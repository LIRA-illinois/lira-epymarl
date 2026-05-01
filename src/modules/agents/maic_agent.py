import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor

import torch.distributions as D
from torch.distributions import kl_divergence


# updated implementation w/ better encapsulation of functions and an explicit comms value parameter
# also has a new method to change the comms value during training
# @th.compile
class MAICAgent(nn.Module):
    """class for a team of agents that communicate using MAIC"""

    def __init__(self, input_shape, args):
        """
        # args.comms_val \in (0, 1], larger comms_val means "more" communication between agents
        # = 0 means no comms between agents, attention weights for all messages are set to 0, equivalent to not using the MAIC module
        # = 1 means unrestricted comms between agents, no attention weights are filtered out
        """
        super(MAICAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions

        self.comms_value: float

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

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def update_comms_value(self, new_comms_value: float):
        self.comms_value = new_comms_value

    def forward(
        self,
        inputs: Tensor,
        hidden_state: Tensor,
        bs: int,
        test_mode: bool = False,
        **kwargs,
    ):
        q_local, hidden_state = self._get_local_q_value(
            inputs=inputs, hidden_state=hidden_state
        )

        latent, latent_embed = self._get_teammate_embedding(
            hidden_state=hidden_state, bs=bs, test_mode=test_mode
        )

        gated_msg = self._get_messages(
            hidden_state=hidden_state, bs=bs, latent=latent, test_mode=test_mode
        )

        # update estimated Q-value using incentive messsages from other agents
        msg_tot = th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
        q_out: Tensor = q_local + msg_tot

        # verify that changing comms value affects q_out
        # if test_mode:
        #     print("msg_filter_thres\n", 1.0 - self.comms_value)
        #     print("comms_value\n", self.args.comms_value)
        #     print("msg_tot\n", msg_tot)
        #     print("q_local\n", q_local)
        #     print("q_out\n", q_out)

        # get auxiliary losses
        returns = {}

        if "train_mode" in kwargs and kwargs["train_mode"]:
            if hasattr(self.args, "mi_loss_weight") and self.args.mi_loss_weight > 0:
                returns["mi_loss"] = self._get_action_mi_loss(
                    hidden_state, bs, latent_embed, q_out
                )

            if (
                hasattr(self.args, "entropy_loss_weight")
                and self.args.entropy_loss_weight > 0
            ):
                alpha = self._get_attention_weights(
                    hidden_state=hidden_state, latent=latent, bs=bs, compute_loss=True
                )
                returns["entropy_loss"] = self._get_entropy_loss(alpha)

        return q_out, hidden_state, returns

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
    ) -> Tensor:
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
            hidden_state=hidden_state, latent=latent, bs=bs
        )

        # filter out messages with attention weights below a pre-defined threshold
        if test_mode:
            # slightly different from the paper, if we let \delta = msg_filter_thres, then we do not scale \delta by n_agents when filtering out the messages
            # This gives more control over the sparsity of messages. It feels like there was not a good reason to scale by the number of agents in the original implementation.
            msg_filter_thres = 1.0 - self.comms_value
            alpha[alpha < msg_filter_thres] = 0

            # original implementation
            # alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg: Tensor = alpha * msg

        return gated_msg

    def _get_attention_weights(
        self, hidden_state: Tensor, latent: Tensor, bs: int, compute_loss: bool = False
    ):
        # compute attention weights for the messages
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

        return alpha

    def _get_action_mi_loss(self, hidden_state, bs, latent_embed, q):
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
