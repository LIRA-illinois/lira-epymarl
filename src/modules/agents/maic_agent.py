from typing import Callable
from typing import Literal
from collections import defaultdict

import torch as th
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.types import Tensor

from torch import topk as hard_topk
from softtorch import topk as soft_topk


class STEFunction(th.autograd.Function):
    """_summary_
    straight-through estimation of gradients: Pass the gradient through unchanged
    one way of estimating gradients thru a binary threshold
    grad_output corresponds to gradient wrt x
    """

    @staticmethod
    def forward(ctx, x: Tensor, thres: float) -> Tensor:
        # Binary threshold: 1 if x > thres, else 0
        return (x >= thres).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        need to return a tuple with gradients for each forward input (x, thres). Pass the gradient through for `x` and return None for `thres since it is a constant
        """
        return grad_output, None


@th.compile
class MAICAgent(nn.Module):
    """class for a team of agents that communicate using MAIC
    updated implementation w/ better encapsulation of functions and different options for filtering messages
    """

    def __init__(self, input_shape, args) -> None:
        super(MAICAgent, self).__init__()
        self.args = args
        self.n_agents: int = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions
        self._msg_budget: int = getattr(args, "msg_budget_per_agent", self.n_agents - 1)

        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(args.hidden_dim, args.nn_hidden_size),
            nn.BatchNorm1d(args.nn_hidden_size),
            activation_func,
            nn.Linear(args.nn_hidden_size, args.n_agents * args.latent_dim * 2),
        )

        self.variational_dist_net = nn.Sequential(
            nn.Linear(args.hidden_dim + args.n_actions, args.nn_hidden_size),
            nn.BatchNorm1d(args.nn_hidden_size),
            activation_func,
            nn.Linear(args.nn_hidden_size, args.latent_dim * 2),
        )

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.msg_net = nn.Sequential(
            nn.Linear(args.hidden_dim + args.latent_dim, args.nn_hidden_size),
            activation_func,
            nn.Linear(args.nn_hidden_size, args.n_actions),
        )

        self.w_query = nn.Linear(args.hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

        self.msg_filter_type: Literal[
            "hard_topk", "soft_topk", "attention_thres", "attention_thres_only_test"
        ] = getattr(args, "msg_filter_type", "attention_thres_only_test")

        self._msg_filter: Callable

        match self.msg_filter_type:
            case "hard_topk":
                self._msg_filter = self._msg_filter_hard_topk
            case "soft_topk":
                self._msg_filter = self._msg_filter_soft_topk
            case "attention_thres":
                self._msg_filter = self._msg_filter_attention_thres
            case "attention_thres_only_test":
                self._msg_filter = self._msg_filter_attention_thres_only_test

        if hasattr(args, "msg_drop_rate"):
            self._msg_dropout = nn.Dropout(p=args.msg_drop_rate)

    def _unscaled_dropout(self, msg: Tensor):
        # dropout scales input by by 1 / (1-p), so undo that operation
        return (1 - self.args.msg_drop_rate) * self._msg_dropout(msg)

    def init_hidden(self) -> Tensor:
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    @property
    def msg_budget(self):
        return self._msg_budget

    @msg_budget.setter
    def msg_budget(self, value: int):
        self._msg_budget = value

    def forward(
        self,
        inputs: Tensor,
        hidden_state: Tensor,
        bs: int,
        test_mode: bool = False,
        **kwargs,
    ):
        # aux losses and any other info to be logged
        agent_info: dict = defaultdict(dict)

        q_local, hidden_state = self._get_local_q_value(
            inputs=inputs, hidden_state=hidden_state
        )

        if self.n_agents > 1:
            latent, latent_embed = self._get_teammate_embedding(
                hidden_state=hidden_state, bs=bs, test_mode=test_mode
            )

            gated_msg, msg_weights = self._get_messages(
                hidden_state=hidden_state, bs=bs, latent=latent, test_mode=test_mode
            )

            # average over the incoming messages for each agent to get a measure of "importance" for that agent for a given episode
            # log_data = th.sum(log_data, axis=0) / self.args.n_agents
            msg_weights = msg_weights.detach().to(device="cpu")

            agent_info["logs"]["msg_weights_in_mean"] = msg_weights.mean(axis=1)
            agent_info["logs"]["msg_weights_out_mean"] = msg_weights.mean(axis=2)

            # update estimated Q-value using incentive messages from other agents
            msg_tot = th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
            q_out: Tensor = q_local + msg_tot

            if kwargs.get("train_mode", False):
                if (
                    hasattr(self.args, "mi_loss_weight")
                    and self.args.mi_loss_weight > 0
                ):
                    agent_info["losses"]["mi_loss"] = self._get_action_mi_loss(
                        hidden_state, bs, latent_embed, q_out
                    )

                if (
                    hasattr(self.args, "entropy_loss_weight")
                    and self.args.entropy_loss_weight > 0
                ):
                    alpha = self._get_attention_weights(
                        hidden_state=hidden_state,
                        latent=latent,
                        bs=bs,
                        compute_loss=True,
                    )
                    agent_info["losses"]["entropy_loss"] = self._get_entropy_loss(alpha)

        else:
            # same as VDN, QMIX, or whatever mixer you're using
            q_out = q_local

        return q_out, hidden_state, agent_info

    def _get_local_q_value(
        self, inputs: Tensor, hidden_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """get each agent's estimated Q-value based only on their local observation"""
        x: Tensor = F.relu(self.fc1(inputs))

        # shape: (self.n_agents, self.args.hidden_dim)
        h_in: Tensor = hidden_state.reshape(-1, self.args.hidden_dim)

        # hidden_state (\tau_i^t) summarizes the agent's trajectory to the current time
        hidden_state = self.rnn(x, h_in)
        q: Tensor = self.fc2(hidden_state)

        return q, hidden_state

    def _get_teammate_embedding(
        self, hidden_state: Tensor, bs: int, test_mode: bool = False
    ) -> tuple[Tensor, Tensor]:

        # get each agent's local representation of its teammates
        latent_parameters = self.embed_net(hidden_state)

        latent_parameters[:, -self.n_agents * self.latent_dim :] = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]),
            min=self.args.var_floor,
        )

        latent_embed: Tensor = latent_parameters.reshape(
            bs * self.n_agents, self.n_agents * self.latent_dim * 2
        )

        latent: Tensor
        if test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            teammate_embed_dist = D.Normal(
                loc=latent_embed[:, : self.n_agents * self.latent_dim],
                scale=(latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2),
            )

            # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
            latent = teammate_embed_dist.rsample()
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        return latent, latent_embed

    def _get_messages(
        self, hidden_state: Tensor, bs: int, latent: Tensor, test_mode: bool
    ) -> tuple[Tensor, Tensor]:
        """generate each agent's incentive messages it sends to its teammates"""
        # get base incentive messages
        h_repeat = (
            hidden_state.view(bs, self.n_agents, -1)
            .repeat(1, self.n_agents, 1)
            .view(bs * self.n_agents * self.n_agents, -1)
        )

        msg = self.msg_net(th.cat([h_repeat, latent], dim=-1)).view(
            bs, self.n_agents, self.n_agents, self.n_actions
        )

        alpha = self._get_attention_weights(
            hidden_state=hidden_state,
            latent=latent,
            bs=bs,
            test_mode=test_mode,
        )

        # apply weighting to messages
        gated_msg: Tensor = alpha * msg
        return gated_msg, alpha

    def _get_attention_weights(
        self,
        hidden_state: Tensor,
        latent: Tensor,
        bs: int,
        compute_loss: bool = False,
        test_mode: bool = False,
    ):
        """
        compute attention weights for the messages
        """
        # we only want this auxiliary loss to affect the params for the message-generator network, so
        # detach to prevent backprop from from using this loss to change the
        # teammate latent network or the agent's RNN
        if compute_loss:
            hidden_tmp = hidden_state.detach()
            latent_tmp = latent.detach()
            scaling = 1.0
        else:
            hidden_tmp = hidden_state
            latent_tmp = latent
            scaling = self.args.attention_dim ** (1 / 2)

        query = self.w_query(hidden_tmp).unsqueeze(1)
        key = (
            self.w_key(latent_tmp)
            .reshape(bs * self.n_agents, self.n_agents, -1)
            .transpose(1, 2)
        )

        alpha = th.bmm(query / scaling, key).view(bs, self.n_agents, self.n_agents)

        if not compute_loss:
            # set each agent's attention weight for its message to itself to a large negative value
            # so the softmax sets it to 0
            for i in range(self.n_agents):
                alpha[:, i, i] = -1e9

        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        # always run filtering if unique_policy_per_msg_budget
        # if unique_policy_per_msg_budget false, always run filtering during test mode
        print(self.args.msg_budget_per_agent)
        if getattr(self.args, "unique_policy_per_msg_budget", False) or test_mode:
            alpha = self._msg_filter(alpha, compute_loss)

        # random dropping of messages to model comms network reliability
        if hasattr(self.args, "msg_drop_rate"):
            alpha = self._unscaled_dropout(alpha)

        return alpha

    def _msg_filter_hard_topk(self, alpha: Tensor, _):
        # keep only the top-k message weights and zero out the rest
        _, top_k_idx = hard_topk(alpha, k=self._msg_budget, dim=2)

        mask = th.zeros_like(alpha, dtype=th.bool)
        mask.scatter_(2, top_k_idx, True)
        alpha = th.where(mask, alpha, th.zeros_like(alpha))

        return alpha

    def _msg_filter_soft_topk(self, alpha: Tensor, compute_loss: bool):
        # only keep K messages with the highest attention values, 0 out the rest
        # based on the current message budget for the team
        top_k_vals, top_k_idx = soft_topk(alpha, k=self._msg_budget, dim=2)

        # Build a soft mask over the selected entries
        mask = top_k_idx.sum(dim=2)
        mask = mask.reshape(alpha.shape)

        # Apply the mask so only the top-k entries remain active.
        # An approximation since using a soft version of top-K
        alpha = alpha * mask

        if not compute_loss:
            # set each agent's attention weight for its message to itself to a large negative value
            # so the softmax sets it to 0
            for i in range(self.n_agents):
                alpha[:, i, i] = -1e9

        # this is a way of approximately enforcing the message budget since we're using a soft verion of "top K" to allow gradients to flow backward
        # smaller temperature encourages the distribution to concentrate more probability mass in fewer outputs
        # higher temperature leads to a flatter distribution
        # 0.025 seems about right to get the sharpening you want for the 5-agent case
        # 0.05 allows more messages than you want
        # probably needs be tuned for other n_agents too
        temperature = self.args.soft_topk_base_temperature * self._msg_budget
        alpha = F.softmax(alpha / temperature, dim=2)

        return alpha

    def _msg_filter_attention_thres(self, alpha: Tensor, _):
        # filter message weights based on a threshold on the weights, can be used during training as part of gradient computations
        msg_filter = STEFunction.apply(alpha, self.msg_filter_thres)
        alpha = msg_filter * alpha
        return alpha

    def _msg_filter_attention_thres_only_test(self, alpha: Tensor, _):
        # original approach: only filters messages during evaluation where they aren't part of any gradient calculation for learning
        # set attention weights to 0 if they are below a pre-defined threshold
        # slightly different from the paper, if we let \delta = msg_filter_thres, then we do not scale
        # \delta by n_agents when filtering out the messages
        # This gives more control over the sparsity of messages. In the original implementation, it did
        # not feel like there was a good reason to scale by the number of agents.
        alpha[alpha < 0.25] = 0

        # original implementation, has weird scaling
        # alpha[alpha < (0.25 * 1 / self.n_agents)] = 0
        return alpha

    def _get_action_mi_loss(
        self, hidden_state: Tensor, bs: int, latent_embed: Tensor, q: Tensor
    ):
        """mutual information loss to train each agent's teammate model"""
        # get the conditional distribution which approximates the variational distribution
        # p(z_{ij} | \tau_i, d_j)
        latent_embed = latent_embed.view(
            bs * self.n_agents, 2, self.n_agents, self.latent_dim
        )

        teammate_conditional_dist = D.Normal(
            loc=latent_embed[:, 0, :, :].reshape(-1, self.latent_dim),
            scale=latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1 / 2),
        )

        # get the variational distribution which includes the other agent's action
        # q_{\xi}(z_{ij} | \tau_i, a_j, d_j)
        hi = (
            hidden_state.view(bs, self.n_agents, 1, -1)
            .repeat(1, 1, self.n_agents, 1)
            .view(bs * self.n_agents * self.n_agents, -1)
        )

        selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = (
            th.zeros(selected_action.shape[0], self.n_actions)
            .to(self.args.device)
            .scatter(1, selected_action, 1)
        )
        one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(
            1, self.n_agents, 1, 1
        )
        one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

        latent_infer = self.variational_dist_net(th.cat([hi, one_hot_a], dim=-1)).view(
            bs * self.n_agents * self.n_agents, -1
        )

        latent_infer[:, self.latent_dim :] = th.clamp(
            th.exp(latent_infer[:, self.latent_dim :]), min=self.args.var_floor
        )

        teammate_variational_dist = D.Normal(
            loc=latent_infer[:, : self.latent_dim],
            scale=latent_infer[:, self.latent_dim :] ** (1 / 2),
        )

        mi_loss = (
            kl_divergence(teammate_conditional_dist, teammate_variational_dist)
            .sum(-1)
            .mean()
        )

        return mi_loss * self.args.mi_loss_weight

    def _get_entropy_loss(self, alpha):
        """compute entropy loss to learn attention weights for
        "sparse yet effective communication" (quote from the MAIC paper)
        """
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = -(alpha * th.log2(alpha)).sum(-1).mean()

        return entropy_loss * self.args.entropy_loss_weight

    # @property
    # def comms_value(self) -> float:
    #     """_summary_
    #     comms_value in [0, 1], larger comms_value means "more" communication between agents
    #     = 0 means no comms between agents, attention weights for all messages are set to 0, equivalent to not using the MAIC module
    #     = 1 means unrestricted comms between agents, no attention weights are filtered out
    #     """
    #     return self._comms_value

    # @comms_value.setter
    # def comms_value(self, value) -> None:
    #     self._comms_value = value

    # @property
    # def msg_filter_thres(self) -> float:
    #     """based on definition of comms_value,
    #     if comms = 1.0, that means no restriction on communication, so the threshold should be set to 0
    #     if comms = 0.0, that means all communication is restricted, so the threshold should be set to 1.0
    #     """
    #     return 1.0 - self._comms_value

    # def _approx_threshold_sigmoid(
    #     self, x: Tensor, steepness: float = 50.0, bias: float = 0.5
    # ) -> Tensor:
    #     """uses a steep sigmoid to approximate a hard threshold set at "bias",
    #     using the sigmoid allows gradients to pass through the thresholding operation
    #     Setting steepness to 50 means values < 0.4 are set to 0 and > 0.6 are set to 1, which is probably good enough for this use case. Setting it steeper may lead to gradient issues.

    #     Parameters
    #     ----------
    #     x : Tensor
    #         data to be thresholded
    #     steepness : float, optional
    #         by default 50.
    #     bias : float, optional
    #         location of the threshold, by default 0.5

    #     Returns
    #     -------
    #     Tensor
    #         tensor with approximately binary values
    #     """
    #     return th.sigmoid(steepness * (x - bias))


# original implementation
# @th.compile
# class MAICAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(MAICAgent, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.latent_dim = args.latent_dim
#         self.n_actions = args.n_actions

#         NN_HIDDEN_SIZE = args.nn_hidden_size
#         activation_func = nn.LeakyReLU()

#         self.embed_net = nn.Sequential(
#             nn.Linear(args.hidden_dim, NN_HIDDEN_SIZE),
#             nn.BatchNorm1d(NN_HIDDEN_SIZE),
#             activation_func,
#             nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
#         )

#         self.inference_net = nn.Sequential(
#             nn.Linear(args.hidden_dim + args.n_actions, NN_HIDDEN_SIZE),
#             nn.BatchNorm1d(NN_HIDDEN_SIZE),
#             activation_func,
#             nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
#         )

#         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
#         self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
#         self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

#         self.msg_net = nn.Sequential(
#             nn.Linear(args.hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
#             activation_func,
#             nn.Linear(NN_HIDDEN_SIZE, args.n_actions),
#         )

#         self.w_query = nn.Linear(args.hidden_dim, args.attention_dim)
#         self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

#     def init_hidden(self):
#         return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

#     def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.hidden_dim)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)

#         latent_parameters = self.embed_net(h)
#         latent_parameters[:, -self.n_agents * self.latent_dim :] = th.clamp(
#             th.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]),
#             min=self.args.var_floor,
#         )

#         latent_embed = latent_parameters.reshape(
#             bs * self.n_agents, self.n_agents * self.latent_dim * 2
#         )

#         if test_mode:
#             latent = latent_embed[:, : self.n_agents * self.latent_dim]
#         else:
#             gaussian_embed = D.Normal(
#                 latent_embed[:, : self.n_agents * self.latent_dim],
#                 (latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2),
#             )
#             latent = (
#                 gaussian_embed.rsample()
#             )  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
#         latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

#         h_repeat = (
#             h.view(bs, self.n_agents, -1)
#             .repeat(1, self.n_agents, 1)
#             .view(bs * self.n_agents * self.n_agents, -1)
#         )
#         msg = self.msg_net(th.cat([h_repeat, latent], dim=-1)).view(
#             bs, self.n_agents, self.n_agents, self.n_actions
#         )

#         query = self.w_query(h).unsqueeze(1)
#         key = (
#             self.w_key(latent)
#             .reshape(bs * self.n_agents, self.n_agents, -1)
#             .transpose(1, 2)
#         )
#         alpha = th.bmm(query / (self.args.attention_dim ** (1 / 2)), key).view(
#             bs, self.n_agents, self.n_agents
#         )
#         for i in range(self.n_agents):
#             alpha[:, i, i] = -1e9
#         alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

#         if test_mode:
#             alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

#         gated_msg = alpha * msg

#         return_q = q + th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)

#         returns = {}
#         if "train_mode" in kwargs and kwargs["train_mode"]:
#             if hasattr(self.args, "mi_loss_weight") and self.args.mi_loss_weight > 0:
#                 returns["mi_loss"] = self.calculate_action_mi_loss(
#                     h, bs, latent_embed, return_q
#                 )
#             if (
#                 hasattr(self.args, "entropy_loss_weight")
#                 and self.args.entropy_loss_weight > 0
#             ):
#                 query = self.w_query(h.detach()).unsqueeze(1)
#                 key = (
#                     self.w_key(latent.detach())
#                     .reshape(bs * self.n_agents, self.n_agents, -1)
#                     .transpose(1, 2)
#                 )
#                 alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(
#                     bs, self.n_agents, self.n_agents
#                 )
#                 returns["entropy_loss"] = self.calculate_entropy_loss(alpha)

#         return return_q, h, returns

#     def calculate_action_mi_loss(self, h, bs, latent_embed, q):
#         latent_embed = latent_embed.view(
#             bs * self.n_agents, 2, self.n_agents, self.latent_dim
#         )
#         g1 = D.Normal(
#             latent_embed[:, 0, :, :].reshape(-1, self.latent_dim),
#             latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1 / 2),
#         )
#         hi = (
#             h.view(bs, self.n_agents, 1, -1)
#             .repeat(1, 1, self.n_agents, 1)
#             .view(bs * self.n_agents * self.n_agents, -1)
#         )

#         selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
#         one_hot_a = (
#             th.zeros(selected_action.shape[0], self.n_actions)
#             .to(self.args.device)
#             .scatter(1, selected_action, 1)
#         )
#         one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(
#             1, self.n_agents, 1, 1
#         )
#         one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

#         latent_infer = self.inference_net(th.cat([hi, one_hot_a], dim=-1)).view(
#             bs * self.n_agents * self.n_agents, -1
#         )
#         latent_infer[:, self.latent_dim :] = th.clamp(
#             th.exp(latent_infer[:, self.latent_dim :]), min=self.args.var_floor
#         )
#         g2 = D.Normal(
#             latent_infer[:, : self.latent_dim],
#             latent_infer[:, self.latent_dim :] ** (1 / 2),
#         )
#         mi_loss = kl_divergence(g1, g2).sum(-1).mean()
#         return mi_loss * self.args.mi_loss_weight

#     def calculate_entropy_loss(self, alpha):
#         alpha = th.clamp(alpha, min=1e-4)
#         entropy_loss = -(alpha * th.log2(alpha)).sum(-1).mean()
#         return entropy_loss * self.args.entropy_loss_weight
