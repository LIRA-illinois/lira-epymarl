from math import prod
from numbers import Integral
from typing import Literal

import torch as th

from src.components.action_selectors import REGISTRY as action_REGISTRY
from src.modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class MAICMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._obs_last_action = args.obs_last_action
        self._obs_agent_id = args.obs_agent_id
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self._agent_ids = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(
            ep_batch, t_ep, test_mode=test_mode, train_mode=False
        )
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs) -> tuple[th.Tensor, dict]:
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_kwargs = kwargs.copy()
        if getattr(self.args, "condition_on_msg_budget", False):
            agent_kwargs["msg_budget_per_agent"] = (
                self._get_contextual_msg_budgets(ep_batch, t)
            )
        agent_outs, self.hidden_states, agent_info = self.agent.forward(
            agent_inputs,
            self.hidden_states,
            ep_batch.batch_size,
            test_mode=test_mode,
            **agent_kwargs,
        )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        # resize agent outputs (Q value or action probability tensor)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

        return agent_outs, agent_info

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def to(self, device: int | Literal["cuda"]):
        self.agent.to(device)

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path: str | dict):
        if isinstance(path, str):
            self.agent.load_state_dict(
                th.load(
                    "{}/agent.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        elif isinstance(path, dict):
            state_dict = path
            self.agent.load_state_dict(state_dict)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        obs = batch["obs"][:, t].reshape(bs * self.n_agents, -1)

        if self._obs_last_action:
            if t == 0:
                last_action = th.zeros_like(batch["actions_onehot"][:, t])
            else:
                last_action = batch["actions_onehot"][:, t - 1]
            last_action = last_action.reshape(bs * self.n_agents, -1)

        if self._obs_agent_id:
            if self._agent_ids is None or self._agent_ids.device != batch.device:
                self._agent_ids = th.eye(self.n_agents, device=batch.device)
            agent_ids = self._agent_ids.unsqueeze(0).expand(bs, -1, -1)
            agent_ids = agent_ids.reshape(bs * self.n_agents, -1)

        inputs = [obs]
        if self._obs_last_action:
            inputs.append(last_action)
        if self._obs_agent_id:
            inputs.append(agent_ids)
        if getattr(self.args, "condition_on_msg_budget", False) or getattr(
            self.args, "condition_on_subtask", False
        ):
            policy_context = batch["policy_context"][:, t]
            policy_context = (
                policy_context.unsqueeze(1)
                .expand(-1, self.n_agents, -1)
                .reshape(bs * self.n_agents, -1)
            )
            inputs.append(policy_context)
        return th.cat(inputs, dim=1)

    def _get_contextual_msg_budgets(self, batch, t) -> th.Tensor:
        context_budget = batch["policy_context"][:, t, 0]
        return th.round(context_budget * max(self.n_agents - 1, 0)).long()

    def _get_input_shape(self, scheme):
        obs_shape = scheme["obs"]["vshape"]
        input_shape = (
            int(obs_shape) if isinstance(obs_shape, Integral) else int(prod(obs_shape))
        )
        if self._obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self._obs_agent_id:
            input_shape += self.n_agents
        if getattr(self.args, "condition_on_msg_budget", False):
            input_shape += 1
        if getattr(self.args, "condition_on_subtask", False):
            input_shape += len(self.args.subtask_contexts)

        return input_shape

    @property
    def msg_budget_per_agent(self):
        return self.agent.msg_budget_per_agent

    @msg_budget_per_agent.setter
    def msg_budget_per_agent(self, value: int):
        self.agent.msg_budget_per_agent = value
