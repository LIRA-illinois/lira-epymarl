import numpy as np
from numpy.typing import NDArray
from torch.types import Tensor

from components.episode_buffer import EpisodeBatch

# import torch as th
# import torch.nn as nn

# import gurobipy as gp


class ILPModel():
    """integer linear program, takes an MDP model with transition probabilities and solves for the optimal policy
    works great for small problems, but this model + solution method does not scale super well with the number of states
    """
    def __init__(self, input_shape, args):
        # self.input_shape = input_shape
        self.args = args

    def select_actions(self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, test_mode: bool) -> dict:
        # return action with shape (n_hierarchical_actions)

        # state is a scalar in the HLMDP
        hl_state = ep_batch["hl_state"][:, t_ep, :].cpu().numpy().flatten()
        mdp_state = int(hl_state[0])
        task_completed = bool(hl_state[1])

        chosen_next_mdp_state: int
        comms_allocation: float

        # can have a null action while the low-level policy is executing its task
        if test_mode:
            # actually choose the comms value to be used during evaluation of MAIC
            comms_allocation = 0.0
        else:
            comms_allocation = 0.0

        if (t_env == 0) and (t_ep == 0):
            print("solving model-based MDP problem (ILP) to get full tabular policy")
            # assume the high-level model never changes for a given env during training
            # you can pre-solve this one time for each env and then save the tabular policy to a dict or something
            # walk thru each state in the HLMDP
            # construct avail actions given the current state using the avail_actions method from the env itself
            # solve the optimization problem to populate a tabular policy
            # grab an action for the current HLMDP state
            ## idk how I want to handle comms values yet, but its only currently relevant for evaluation so don't worry about it for now
            chosen_next_mdp_state = mdp_state

        else:
            if task_completed:
                # if you just completd a task, low-level agents need a new task assigned to them
                # then once we're running this in the loop, just grab the correct action from the tabular policy based on the current state
                chosen_next_mdp_state = mdp_state + 1

            else:
                # if you didn't just complete a task, take a null, "stay" action to keep pursuing the current goal
                chosen_next_mdp_state = mdp_state

        actions: dict = {
            "chosen_next_state": chosen_next_mdp_state,
            "comms_allocation": comms_allocation,
        }

        return actions

    def cuda(self):
        pass

