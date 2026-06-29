from torch import Tensor

from src.components.episode_buffer import EpisodeBatch
from src.controllers import REGISTRY as mac_REGISTRY
from src.modules.agents import REGISTRY as agent_REGISTRY
from src.components.action_selectors import CommsActionSelector


class CommsMAC:
    def __init__(self, scheme, groups, args):

        # mac to interact with low-level environment
        self.env_mac = mac_REGISTRY[args.mac](scheme, groups, args)

        # mac to select comms allocation actions and stuff
        self.action_selector = CommsActionSelector(args=args)

        self.n_agents: int = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.env_action_key: str = "env_actions"

        # self.agent_output_type = args.agent_output_type

    def _build_agents(self, input_shape):
        # you would add your model-based ILP agent as an agent in the registry to be grabbed here
        self.comms_agent = agent_REGISTRY[self.args.comms_agent](self.args)

    @property
    def env_agent(self):
        return self.env_mac.agent

    def init_hidden(self, batch_size):
        self.env_mac.init_hidden(batch_size=batch_size)

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

    def parameters(self):
        params = list(self.env_agent.parameters())
        if hasattr(self.comms_agent, "parameters"):
            params += list(self.comms_agent.parameters())

        return params

    def select_actions(
        self, ep_batch: EpisodeBatch, t_ep, t_env, bs=slice(None), test_mode=False
    ) -> dict:
        # get the model of the env + optimize
        # update comms allocation to low-level agents + choose next task to complete (if relevant)
        high_level_actions = self.comms_agent.select_actions(ep_batch, t_ep, t_env, test_mode=test_mode)

        # TODO assume this happens in an external control loop for now, we only change comms when doing different evaluation subtasks
        # self.update_comms_value(high_level_actions["comms_allocation"])

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

    @property
    def comms_value(self):
        return self.env_mac.comms_value

    @comms_value.setter
    def comms_value(self, value):
        self.comms_value = value

    def forward(self, ep_batch: EpisodeBatch, t, test_mode=False, **kwargs):
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

    def _build_inputs(self, batch, t):

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

    def save_models(self, path: str):
        self.env_mac.save_models(path)

    @property
    def comms_value(self):
        return self.env_mac.comms_value