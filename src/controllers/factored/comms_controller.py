from torch import Tensor
from torch.nn.parameter import Parameter

from src.components.action_selectors import CommsActionSelector
from src.components.episode_buffer import EpisodeBatch
from src.controllers import REGISTRY as mac_REGISTRY
from src.modules.agents import REGISTRY as agent_REGISTRY
from src.modules.agents.rnn_agent import RNNAgent


class CommsMAC:
    def __init__(self, scheme, groups, args) -> None:

        # mac to interact with low-level environment
        self.env_mac = mac_REGISTRY[args.mac](scheme, groups, args)

        # mac to select comms allocation actions and stuff
        self.action_selector = CommsActionSelector(args=args)

        self.n_agents: int = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.env_action_key: str = "env_actions"
        self._dependent_policy_bank: dict[tuple[int, int, float, int], dict] = {}
        self._dependent_policy_seeds: tuple[int, ...] = ()
        self._fixed_dependent_policy_seed: int | None = None
        self._episode_index = 0
        self._active_dependent_policy_seed: int | None = None
        self._active_hl_state: int | None = None
        self._active_hl_actions: dict | None = None
        self._active_policy_key: tuple[int, int, float, int] | None = None

        # self.agent_output_type = args.agent_output_type

    def _build_agents(self, input_shape) -> None:
        # you would add your model-based ILP agent as an agent in the registry to be grabbed here
        self.comms_agent = agent_REGISTRY[self.args.comms_agent](self.args)

    @property
    def env_agent(self) -> RNNAgent:
        return self.env_mac.agent

    def init_hidden(self, batch_size) -> None:
        self.env_mac.init_hidden(batch_size=batch_size)
        self._active_hl_state = None
        self._active_hl_actions = None
        self._active_policy_key = None
        if self._dependent_policy_seeds:
            if self._fixed_dependent_policy_seed is not None:
                self._active_dependent_policy_seed = (
                    self._fixed_dependent_policy_seed
                )
            else:
                seed_index = self._episode_index % len(self._dependent_policy_seeds)
                self._active_dependent_policy_seed = self._dependent_policy_seeds[
                    seed_index
                ]
                self._episode_index += 1

    def cuda(self) -> None:
        self.comms_agent.cuda()
        self.env_agent.cuda()

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def parameters(self) -> list[Parameter]:
        params = list(self.env_agent.parameters())
        if hasattr(self.comms_agent, "parameters"):
            params += list(self.comms_agent.parameters())

        return params

    def select_actions(
        self,
        ep_batch: EpisodeBatch,
        t_ep,
        t_env,
        bs=slice(None),
        test_mode: bool = False,
    ) -> dict:
        # get the model of the env + optimize
        # update comms allocation to low-level agents + choose next task to complete (if relevant)
        hl_state = int(ep_batch["hl_state"][:, t_ep, :].flatten()[0].item())
        policy = getattr(self.comms_agent, "policy", None)
        task_policy = getattr(policy, "task_policy", None)
        state_has_actions = (
            task_policy is None
            or task_policy.empty
            or bool((task_policy.state == hl_state).any())
        )
        terminal_hl_state = (
            not state_has_actions
            and self._active_hl_actions is not None
        )
        if hl_state != self._active_hl_state:
            if state_has_actions:
                self._active_hl_actions = self.comms_agent.select_actions(
                    ep_batch, t_ep, t_env, test_mode=test_mode
                )
                self._active_hl_state = hl_state
                self._active_policy_key = None
            elif terminal_hl_state:
                self._active_hl_state = hl_state
            else:
                raise ValueError(
                    f"The high-level policy has no action for nonterminal state "
                    f"{hl_state}."
                )

        high_level_actions = self._active_hl_actions
        if high_level_actions is None:
            raise RuntimeError("High-level action selection returned no action.")

        self.msg_budget_per_agent = int(high_level_actions["comms_budget"])
        if self._dependent_policy_seeds and not terminal_hl_state:
            if self._active_dependent_policy_seed is None:
                raise RuntimeError("Dependent policy seed was not selected.")
            policy_key = (
                hl_state,
                int(high_level_actions["chosen_next_state"]),
                float(high_level_actions["comms_budget"]),
                self._active_dependent_policy_seed,
            )
            if policy_key != self._active_policy_key:
                try:
                    policy = self._dependent_policy_bank[policy_key]
                except KeyError as error:
                    raise KeyError(
                        f"No trained dependent low-level policy for "
                        f"{policy_key[:-1]} at seed {policy_key[-1]}."
                    ) from error
                self.env_mac.load_models(policy)
                self.env_mac.init_hidden(batch_size=ep_batch.batch_size)
                self._active_policy_key = policy_key

        # NDArray or Tensor of size (1, n_env_agents)
        env_actions = self.env_mac.select_actions(ep_batch, t_ep, t_env, bs, test_mode)

        # following the format from the parallel episode runner
        if isinstance(env_actions, Tensor):
            env_actions = env_actions.cpu().numpy()

        actions: dict = {
            "env_actions": env_actions,
            "hl_actions": high_level_actions,
        }

        return actions

    def set_dependent_policy_bank(
        self,
        policy_bank: dict[tuple[int, int, float, int], dict],
        seeds: list[int],
    ) -> None:
        if not policy_bank:
            raise ValueError("Dependent low-level policy bank cannot be empty.")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Dependent policy seeds must be non-empty and unique.")
        self._dependent_policy_bank = policy_bank
        self._dependent_policy_seeds = tuple(int(seed) for seed in seeds)
        self._fixed_dependent_policy_seed = None

    def set_dependent_policy_seed(self, seed: int | None) -> None:
        if seed is not None and seed not in self._dependent_policy_seeds:
            raise ValueError(f"Unknown dependent policy seed {seed}.")
        self._fixed_dependent_policy_seed = seed

    @property
    def msg_budget_per_agent(self):
        return self.env_mac.msg_budget_per_agent

    @msg_budget_per_agent.setter
    def msg_budget_per_agent(self, value) -> None:
        self.env_mac.msg_budget_per_agent = value

    def forward(
        self, ep_batch: EpisodeBatch, t, test_mode: bool = False, **kwargs
    ) -> int:
        return 1
        # agent_inputs = self._build_inputs(ep_batch, t)
        # avail_actions = ep_batch["avail_actions"][:, t]

        # agent_outs, self.hidden_states, losses = self.agent.forward(
        #     agent_inputs,
        #     self.hidden_states,
        #     ep_batch.batch_size,
        #     test_mode=test_mode,
        #     **kwargs,
        # )
        # pass

        # return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), losses

    def _build_inputs(self, batch, t) -> None:

        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        pass
        # bs = batch.batch_size
        # inputs = []
        # inputs.append(batch["obs"][:, t])  # b1av
        # if self.args.obs_last_action:
        #     if t == 0:
        #         inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
        #     else:
        #         inputs.append(batch["actions_onehot"][:, t - 1])
        # if self.args.obs_agent_id:
        #     inputs.append(
        #         th.eye(self.n_agents, device=batch.device)
        #         .unsqueeze(0)
        #         .expand(bs, -1, -1)
        #     )

        # inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        # return inputs

    def save_models(self, path: str) -> None:
        self.env_mac.save_models(path)
