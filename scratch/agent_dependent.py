import copy
import logging
import os
import pickle
import pprint
import time
from dataclasses import dataclass

import dill
import multiprocess as mp
import numpy as np
import pandas as pd
import torch as th
from agents.low_level_agents.components.action_selectors import (
    REGISTRY as action_REGISTRY,
)
from agents.low_level_agents.components.episode_buffer import ReplayBuffer
from agents.low_level_agents.controllers import REGISTRY as mac_REGISTRY

# from envs import COMMS_ENVS as comms_envs
from agents.low_level_agents.learners import REGISTRY as le_REGISTRY
from envs.high_level_envs.finite_automaton import FiniteAutomaton

from experiments import REGISTRY as r_REGISTRY
from utils.logging import Logger, get_console_logger
from utils.timehelper import time_left, time_str
from utils.utils import EmptyClass, OneHot

from .agent import CMAgent


@dataclass
class StateData:
    outgoing_init_state_dist: list
    success_condition: str


@dataclass
class ActionData:
    agents: list
    comms_vals: list
    success_probs: list
    t_eval: list
    seeds: list


@dataclass
class EvalData:
    eval_success_prob: float
    t_eval: int
    unique_successful_final_states: np.ndarray
    final_state_counts: np.ndarray


class CMAgentDependent(CMAgent):
    def __init__(
        self,
        exp_config,
        env_config,
        low_level_agent_data,
        cm_seq,
        args,
        config,
        success_prob_spec=0.85,
    ):
        super().__init__(exp_config, env_config)
        mp.set_start_method("spawn")
        self.low_level_agent_data = low_level_agent_data
        self.cm_seq = cm_seq
        self.success_prob_spec = success_prob_spec
        self.n_train_per_iter = args.n_train_per_iter
        self.args = args
        self.config = config
        self._build_outputs()
        self._build_logger(self.args.base_dir)

        # define saving / loading paths
        self.cm_output_dir = os.path.join(self.args.base_dir, "cm_agent_outputs")
        os.makedirs(self.cm_output_dir, exist_ok=True)
        self.state_save_path = os.path.join(self.cm_output_dir, "state_data.pickle")
        self.action_save_path = os.path.join(self.cm_output_dir, "action_data.pickle")
        self.hlm_save_path = os.path.join(self.cm_output_dir, "hlm_data.pickle")

        self.logger.info("Experiment parameters:")
        experiment_params = pprint.pformat(config, indent=4, width=1)
        self.logger.info("\n" + experiment_params + "\n")

        # you need this when running on the cluster to "activate" the CUDA stuff before actually starting the sub-processes
        ## otherwise it will give an error of "No NVIDIA GPUs available"
        self.logger.info(f"Available GPUs: {th.cuda.device_count()}")
        for i in range(th.cuda.device_count()):
            self.logger.info(f"{th.cuda.get_device_properties(i).name}")
        self.devices = [
            f"cuda:{device_idx}" for device_idx in range(th.cuda.device_count())
        ]

        self.data.comms_vals = {}
        self.data.success_prob_vals = {}

        # get the data specific to the formulation of the env used in this problem
        self._build_state_space_data()
        self._build_action_space_data()
        self.prev_state_space_data = {}

    def _build_logger(self, base_dir):
        # log experiment-level info
        # parallel processes info is logged in the individual agent classes
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = get_console_logger()
        file_output_handler = logging.FileHandler(
            os.path.join(log_dir, "high_level_agent_logs.txt")
        )
        self.logger.addHandler(file_output_handler)

    def _build_subtask_seq_logger(self, base_dir, subtask_seq, iter_idx):
        # log experiment-level info
        # parallel processes info is logged in the individual agent classes
        log_dir = os.path.join(base_dir, "logs", f"iter_{iter_idx}__seq_{subtask_seq}")
        os.makedirs(log_dir, exist_ok=True)
        self.subtask_seq_logger = get_console_logger()
        file_output_handler = logging.FileHandler(
            os.path.join(log_dir, "subtask_seq_logs.txt")
        )
        self.subtask_seq_logger.addHandler(file_output_handler)

    def _build_outputs(self):
        self.data.state_output = []
        self.data.action_output = []
        self.data.hlm_output = []

    def _build_data(self):
        # this function is specific to this CM Agent since it has to build the data for the optimization problem by grabbing it from the action space periodically
        for a in self.env.action_space:
            if self.n_comms_vals is None:
                self.n_comms_vals = len(self.low_level_agent_data["comms_vals"])

            # get indices of success probs associated with each level of comms
            for i, comms_val in enumerate(self.low_level_agent_data["comms_vals"]):
                curr_comms_idxs = [
                    j
                    for j, c in enumerate(self.env.action_space[a].comms_vals)
                    if c == comms_val
                ]

                # only make this the agents that have success_prob > threshold
                curr_success_probs = np.array(self.env.action_space[a].success_probs)[
                    curr_comms_idxs
                ]
                curr_seeds = np.array(self.env.action_space[a].seeds)[curr_comms_idxs]

                # these policies are ignored b/c they never got above the success threshold
                curr_success_probs = np.delete(
                    curr_success_probs, np.argwhere(curr_success_probs == -1)
                )

                # within each level of comms, the "effective" success prob is the average across the policy replicates
                self.data.comms_vals[a, i] = comms_val

                if len(curr_success_probs) == 0:
                    self.data.success_prob_vals[a, i] = 0
                else:
                    self.data.success_prob_vals[a, i] = np.mean(curr_success_probs)

    def _build_state_space_data(self):
        """adds data to the state space that is specific to this problem"""
        for u in self.env.state_space:
            try:
                outgoing_init_state_dist = self.env.hlm_data[u][
                    "outgoing_init_state_dist"
                ]
            except KeyError:
                outgoing_init_state_dist = None

            success_condition = None
            for u_pred, a_pred in self.env.predecessors[u]:
                success_condition = self.env.hlm_data[u_pred]["subtask_idx"][a_pred][
                    "success_condition"
                ]

            self.env.state_space[u] = StateData(
                outgoing_init_state_dist, success_condition
            )

    def _build_action_space_data(self, only_update_agents=False):
        """adds data to the action space that is specific to this problem"""
        for u in self.env.state_space:
            for a in self.env.avail_actions[u]:
                init_state_dist = self.env.state_space[u].outgoing_init_state_dist
                success_condition = self.env.state_space[
                    self.env.successor[u, a]
                ].success_condition

                agents = []
                comms_vals = []
                success_probs = []
                t_eval = []
                seeds = []

                # create a low-level agent for each edge that could realize transition (u, u')
                for comms_val in self.low_level_agent_data["comms_vals"]:
                    for seed in self.low_level_agent_data["seeds"]:
                        agent = SubtaskAgentDependent(
                            a,
                            success_condition,
                            self.args,
                            init_state_dist=init_state_dist,
                            seed=seed,
                            comms_val=comms_val,
                        )
                        agents.append(agent)
                        comms_vals.append(comms_val)
                        success_probs.append(0)
                        seeds.append(seed)

                # # upate the success probs of the first subtasks in the CM based on iter 0
                # if prev_action_space is not None:
                #     for subtask_seq in self.cm_seq[0]:
                #         if a == subtask_seq[0]:
                #             success_probs = prev_action_space[a].success_probs
                #             t_eval = prev_action_space[a].t_eval
                if only_update_agents:
                    success_probs = self.env.action_space[a].success_probs
                    t_evals = self.env.action_space[a].t_eval
                    for i, agent in enumerate(agents):
                        agent.update_curr_best_model(success_probs[i], t_evals[i])
                    self.env.action_space[a].agents = agents

                else:
                    self.env.action_space[a] = ActionData(
                        agents, comms_vals, success_probs, t_eval, seeds
                    )

    def eval_cm(self, curr_iter=0):
        # the goal here is to evaluate a policy by running it a bunch of times
        ## our policy happens to be hierarchical, so we need to load both levels

        # curr iter refers to the HLM iter, since we're looking at df_hlm to pick out which iter to load

        # because of how we define the iters, the high-level policy at iter N+1 is associated with the success probs from iter N
        ## however, here we want to evaluate the hierarchical policy at iter N+1, which means taking both the high- and low-level policy data from this iter and evaluting it
        ## this is something not done during training, so the comms cost in df_hlm is not correct
        # load the env data
        low_level_policy_data_iter = curr_iter + 1
        self._load_hlm(f"iter_{low_level_policy_data_iter}")

        # load blank agent classes
        ## this might overwrite the existing data in self.env
        self._build_action_space_data(only_update_agents=True)

        # get the aggregate (mean) success probs from the individual policies to construct the CM transition function
        self._build_data()

        # load mu and nu from disk
        # action_load_path = os.path.join(self.cm_output_dir, "action_data.pickle")
        # df_action = pd.read_pickle(action_load_path)
        hlm_load_path = os.path.join(self.cm_output_dir, "hlm_data.pickle")
        df_hlm = pd.read_pickle(hlm_load_path)
        df_hlm = df_hlm.loc[df_hlm.iter_idx == curr_iter]
        mu, nu = df_hlm.mu.item(), df_hlm.nu.item()

        # df_action = df_action.loc[df_action.iter_idx == low_level_policy_data_iter]

        # success_probs = {}

        # for subtask_idx in np.unique(df_action.subtask_idx):
        #     idx = 0
        #     for comms_idx in range(3):
        #         prob = 0
        #         prob_vals = df_action.loc[df_action.subtask_idx == subtask_idx].success_probs.item()
        #         n_vals = 0
        #         print(prob_vals)
        #         for seed_idx in range(3):
        #             if prob_vals[idx] >= self.args.min_success_prob_threshold:
        #                 prob += prob_vals[idx]
        #                 print(prob_vals[idx])

        #                 n_vals += 1
        #             idx += 1

        #         prob = prob / n_vals
        #         success_probs[subtask_idx, comms_idx] = prob
        print("mu (u, a)\n", mu)
        print("\nnu (a : comms_idx)\n", nu)
        print("\nsuccess_probs (a, comms_idx)\n", self.data.success_prob_vals)
        # I don't actually do the calculation here
        ## it was easier to do it in "checking hierarchical policy spec satisfaction.xlsx"

        # evaluate the CM success prob
        # global_success_prob = 0
        # for u in self.env.state_space:

        # if u is a predecessor of the final state
        ## get the aggregate success prob from u to u_goal
        ## then sum over the comms vals
        ### nu[u, u_goal, i] * success_prob

        # compare the calcualted success prob to p_c

        # evaluate the CM comms cost function

        # run an evaluation pass of the CM
        # self._cm_pass(self.cm_seq, mu, nu, self.n_train_per_iter, curr_iter, eval=True)

    def optimize_cm(self, curr_iter=0, max_iter=20, eps=0.2):
        start_time = time.time()
        iter_idx = curr_iter

        # change in initial state distributions across iterations
        delta_isd = -1
        done = False
        mu, nu = None, None

        while not done:
            if iter_idx == 0:
                objective_value = -1
            else:
                if (self.args.hlm_load_dir != "") and (iter_idx == curr_iter):
                    iter_dir = f"iter_{iter_idx - 1}"
                    self._load_hlm(iter_dir)

                # solve model for first time in this iter
                self.build_model(self.success_prob_spec)
                self.optimize()
                infeasible_status = self.get_model_infeasible_status()

                # if model infeasible, run cm_pass without updating mu or nu
                ## run MARL training until it finds a decent low-level policy for the given mu and nu
                ## or you run out of total training budget
                if infeasible_status == 1:
                    while infeasible_status == 1:
                        self.logger.info(
                            "\n\n Running another CM pass with the same mu and nu since MARL did not find a set of low-level policies that meet the CM performance spec."
                        )
                        self._cm_pass(
                            self.cm_seq, mu, nu, self.n_train_per_iter, iter_idx
                        )
                        self.build_model(self.success_prob_spec)
                        self.optimize()
                        infeasible_status = self.get_model_infeasible_status()

                objective_value = self.get_objective_value()
                self.build_solution_data()

                # delete the high-level optimization model here so we can start up parallel processes
                self.decision_model = None
                self.opt_vars = EmptyClass()

                # start with fresh data for this iter
                # need to do this b/c the avail_actions gets modifed when the decision model is built and solved
                prev_state_space = copy.deepcopy(self.env.state_space)
                self.env = FiniteAutomaton(self.env_config)
                self.env.state_space = prev_state_space
                self._build_action_space_data()

            mu, nu = self.policy, self.chosen_idx

            self.logger.info("\n\n" + f"CM iteration: {iter_idx}")
            self._cm_pass(self.cm_seq, mu, nu, self.n_train_per_iter, iter_idx)

            # check for convergence
            if iter_idx >= 2:
                delta_isd = self._get_distance_metric()
                distance_str = f"d={delta_isd}, eps={eps}"
                if (delta_isd <= eps) or (iter_idx >= max_iter):
                    done = True
                    if delta_isd <= eps:
                        done_str = f"Converged in {iter_idx + 1} iterations"

                    elif iter_idx >= max_iter:
                        done_str = f"Max CM iterations {iter_idx} reached"

                    self.logger.info(
                        "\n\n"
                        + f"CM agent optimization done, {done_str}, {distance_str}"
                    )
                else:
                    self.logger.info("\n\n" + f"CM agent not converged, {distance_str}")

            # we start over with a fresh set of agent class each iter, so don't save the agent classes to disk
            for k, v in self.env.action_space.items():
                v.agents = []

            self._save_outputs(iter_idx, delta_isd, objective_value, mu, nu)
            self.prev_state_space_data = copy.deepcopy(self.env.state_space)
            iter_idx += 1

        run_time_hours = round((time.time() - start_time) / (60 * 60), 2)
        run_time_mins = round((time.time() - start_time) / (60), 2)
        self.logger.info(
            f"CM Optimization Runtime: {run_time_hours} hours ({run_time_mins} minutes)"
        )

    def _load_hlm(self, iter_dir):
        # loads the HLM data
        load_path = os.path.join(self.cm_output_dir, iter_dir, "env.dill")
        with open(load_path, "rb") as f:
            self.env = dill.load(f)

    def _save_outputs(self, iter_idx, delta_isd, objective_value, mu, nu):
        # get outputs in dicts
        state_data = {}
        for u in self.env.state_space.keys():
            if u != self.env.u_fail:
                state_data[u] = {
                    "init_state_dist": self.env.state_space[u].outgoing_init_state_dist,
                    "success_condition": self.env.state_space[u].success_condition,
                }

        action_data = {}
        for a in self.env.action_space.keys():
            for i, comms_val in enumerate(self.low_level_agent_data["comms_vals"]):
                curr_comms_idxs = [
                    j
                    for j, c in enumerate(self.env.action_space[a].comms_vals)
                    if c == comms_val
                ]
                curr_success_probs = np.array(self.env.action_space[a].success_probs)[
                    curr_comms_idxs
                ]
                curr_seeds = np.array(self.env.action_space[a].seeds)[curr_comms_idxs]

            action_data[a] = {
                "best_policy_timesteps": self.env.action_space[a].t_eval,
                "comms_vals": self.env.action_space[a].comms_vals,
                "success_probs": self.env.action_space[a].success_probs,
                "seeds": self.env.action_space[a].seeds,
            }

        hlm_data = {
            "iter_idx": iter_idx,
            "delta_isd": delta_isd,
            "total_comms_cost": objective_value,
            "mu": mu,
            "nu": nu,
            "optimal_comms_vals": self.optimal_comms_vals,
            # "success_prob_vals": self.data.success_prob_vals,
        }

        # convert the dicts to dfs
        df_state = self._convert_to_dataframe(state_data, iter_idx)
        df_state["state_idx"] = df_state.index
        new_col_order = [
            "iter_idx",
            "state_idx",
            "success_condition",
            "init_state_dist",
        ]
        df_state = df_state[new_col_order]

        df_action = self._convert_to_dataframe(action_data, iter_idx)
        df_action["subtask_idx"] = df_action.index
        new_col_order = [
            "iter_idx",
            "subtask_idx",
            "best_policy_timesteps",
            "comms_vals",
            "success_probs",
            "seeds",
        ]
        df_action = df_action[new_col_order]

        df_hlm = self._convert_to_dataframe(hlm_data, iter_idx, hlm_data=True)

        # append the df to the list of output dfs
        self.data.state_output.append(df_state)
        self.data.action_output.append(df_action)
        self.data.hlm_output.append(df_hlm)

        # concat the dfs in the list to form a big output df
        df_state_out = pd.concat((self.data.state_output), axis=0).reset_index(
            drop=True
        )
        df_action_out = pd.concat((self.data.action_output), axis=0).reset_index(
            drop=True
        )
        df_hlm_out = pd.concat((self.data.hlm_output), axis=0).reset_index(drop=True)

        df_state_out.to_pickle(self.state_save_path)
        df_action_out.to_pickle(self.action_save_path)
        df_hlm_out.to_pickle(self.hlm_save_path)

        # save the env to disk for re-starting in the middle of training
        env_save_dir = os.path.join(self.cm_output_dir, f"iter_{iter_idx}")
        os.makedirs(env_save_dir, exist_ok=True)
        env_save_path = os.path.join(env_save_dir, "env.dill")

        with open(env_save_path, "wb") as f:
            dill.dump(self.env, f)

    def _convert_to_dataframe(self, data, iter_idx, hlm_data=False):
        if hlm_data:
            df = pd.DataFrame.from_dict([data], orient="columns")
        else:
            df = pd.concat([pd.DataFrame(data)]).T
        df["iter_idx"] = iter_idx
        return df

    def _cm_pass(self, cm_seq, mu, nu, n_train, iter_idx, eval=False):
        start_time = time.time()

        for step in cm_seq.keys():
            self.logger.info(f"Starting subtask sequence step: {step}")
            subtask_seqs = copy.deepcopy(cm_seq[step])

            # # remove the first subtasks from cm_step 0 since we're reusing the success prob and final state dist data from iter 0
            # if (iter_idx > 0) and (step == 0):
            #     for subtask_seq in subtask_seqs:
            #         subtask_seq.pop(0)

            ## right now there are 3 and 1 process per GPU, so I'm setting these as N+1
            ## the key is the GPU idx, and the nth entry of the list refers to the nth CM step
            # for the Titan computer
            max_parallel_processes_per_gpu = {
                # 0: 3,
                1: 4,
                2: 5,
            }

            # for the A4000 computer
            # n_processes_per_gpu = 4
            # max_parallel_processes_per_gpu = {gpu_idx : n_processes_per_gpu for gpu_idx in self.devices}
            # max_parallel_processes_per_gpu = {
            #     0 : 2,
            #     1 : 5,
            #     2 : 5,
            #     3 : 5,
            # }

            # for cluster
            ## I don't know if this will work
            # n_processes_per_gpu = 40
            # max_parallel_processes_per_gpu = {gpu_idx : n_processes_per_gpu for gpu_idx in self.devices}

            kwarg_inputs = []
            for i, subtask_seq in enumerate(subtask_seqs):
                kwarg_inputs.append(
                    {
                        "subtask_seq": subtask_seq,
                        "mu": mu,
                        "nu": nu,
                        "n_train": n_train,
                        "iter_idx": iter_idx,
                        "gpu_idxs": max_parallel_processes_per_gpu.keys(),
                        "max_parallel_processes_per_gpu": max_parallel_processes_per_gpu,
                        "eval": eval,
                    }
                )

            output_data_lists = []

            # #######################
            # debugging, sequential training with regular returns
            # #######################
            """
            for i, seq in enumerate(subtask_seqs):
                tmp = self._train_subtask_sequence(**kwarg_inputs[i])
                # each entry of eval_data is a list, one for each subtask entering this state
                ## I want it to be a single list
                output_data_lists.append(tmp)
            """

            #######################
            # parallel training with an output queue
            #######################
            # """
            # create one output queue to be shared across all parallel subtask sequences
            output_queue = mp.Queue()
            for i, input_dict in enumerate(kwarg_inputs):
                input_dict["output_queue"] = output_queue

            processes = []

            # start processes
            for i, seq in enumerate(subtask_seqs):
                p = mp.Process(
                    target=self._train_subtask_sequence, kwargs=kwarg_inputs[i]
                )
                processes.append(p)
                p.start()

            # make sure we get everything the queue before allowing join() to be called
            ## get() here can take a few seconds since the action space data can be anywhere from 100 MB to a few GB
            ## this extra read time breaks the standard multiprocessing approach of reading queues and calling join()
            ## https://stackoverflow.com/questions/31708646/process-join-and-queue-dont-work-with-large-numbers
            while True:
                running = any(p.is_alive() for p in processes)
                while not output_queue.empty():
                    self.logger.info("Starting reading data from the queue")
                    start = time.time()
                    output_data_lists.append(output_queue.get(timeout=10))
                    end = time.time()
                    self.logger.info(
                        f"Done reading data from the queue, took {round(end - start, 3)} seconds"
                    )

                if not running:
                    break

            # close up the processes after we have everything from the queues
            for i, p in enumerate(processes):
                p.join()

            self.logger.info("Done with subtask sequences, unpacking data")
            # """

            #######################
            # unpack eval data into a single list
            eval_data = []
            env_data = []

            for data_list in output_data_lists:
                tmp_eval_data = data_list[0]
                for data in tmp_eval_data:
                    eval_data.append(data)

                tmp_env_data = dill.loads(data_list[1])
                env_data.append(tmp_env_data)

            # update the env based on the local envs from the parallel processes
            for env in env_data:
                for state, state_data in env.state_space.items():
                    self.env.state_space[state] = state_data
                for subtask, action_data in env.action_space.items():
                    self.env.action_space[subtask] = action_data

            # # check here that the agents in the action data for subtasks 0 and 2 have runners
            # for k, data in self.env.action_space.items():
            #     self.logger.info(f"subtask {k}")
            #     for i, agent in enumerate(data.agents):
            #         try:
            #             self.logger.info(f"agent idx {i}, {agent.runner}")
            #         except:
            #             self.logger.info(f"agent idx {i}, no runner")

            #######################
            # get the next state in the HLM using the last agent in these edge sequences
            if len(cm_seq) > 1:
                agent = self.env.action_space[cm_seq[step][-1][-1]].agents[0]
                for u in self.env.state_space:
                    if agent.subtask_idx in self.env.avail_actions[u]:
                        next_state = self.env.successor[u, agent.subtask_idx]

                # get set of agents incoming to this state
                incoming_agents = []
                for seq in cm_seq[step]:
                    curr_action_data = self.env.action_space[seq[-1]]
                    for i, agent in enumerate(curr_action_data.agents):
                        agent.curr_best_eval_success_prob = (
                            curr_action_data.success_probs[i]
                        )
                    incoming_agents += self.env.action_space[seq[-1]].agents

                # get agents outgoing from next state
                next_agents = []
                if step < max(cm_seq.keys()):
                    next_step = step + 1
                    for seq in cm_seq[next_step]:
                        next_agents += self.env.action_space[seq[0]].agents

                self._update_initial_state_dist_data(
                    incoming_agents,
                    next_agents,
                    next_state,
                    eval_data,
                    mu,
                    nu,
                    logger=self.logger,
                )

                # print("\n\nState space")
                # for k, v in self.env.state_space.items():
                #     print(k, v)
                # print("\n\nAction space")
                # for k, v in self.env.action_space.items():
                #     print(k, v)

        run_time_hours = round((time.time() - start_time) / (60 * 60), 2)
        run_time_mins = round((time.time() - start_time) / (60), 2)
        self.logger.info(
            f"CM Iter {iter_idx} Runtime: {run_time_hours} hours ({run_time_mins} minutes)"
        )

    def _train_subtask_sequence(
        self,
        subtask_seq,
        mu,
        nu,
        n_train,
        iter_idx,
        gpu_idxs,
        max_parallel_processes_per_gpu,
        output_queue=None,
        eval=False,
    ):
        start_time = time.time()
        self._build_subtask_seq_logger(self.args.base_dir, subtask_seq, iter_idx)

        # create a copy of the env for the purposes of multiprocessing
        ## when this function is run in a separate process, self.env refers to the version of self.env within a separate process and not the global process
        ## therefore, local_env has to be returned to the global process so the global process' self.env can be updated based on trained agent data
        # for debugging in sequential mode
        # local_env = copy.deepcopy(self.env)
        local_env = self.env

        # remove subtasks and states that are not relevant for this subtask sequence
        subtasks_remove = list(set(local_env.action_space.keys()) - set(subtask_seq))
        for subtask in subtasks_remove:
            local_env.action_space.pop(subtask)

        states_keep = []
        for (u, a), u_next in local_env.successor.items():
            for subtask in subtask_seq:
                if (subtask == a) and (subtask != subtask_seq[-1]):
                    states_keep.append(u)
                    states_keep.append(u_next)

        states_remove = list(set(local_env.state_space.keys()) - set(states_keep))

        for state in states_remove:
            local_env.state_space.pop(state)

        # train agents to solve subtasks
        for subtask_idx in subtask_seq:
            self.subtask_seq_logger.info(f"Starting subtask: {subtask_idx}")
            agents_to_train = local_env.action_space[subtask_idx].agents

            agents_to_train, eval_data = self._train_single_subtask(
                agents_to_train,
                n_train,
                iter_idx,
                gpu_idxs,
                max_parallel_processes_per_gpu,
                eval,
            )
            self.subtask_seq_logger.info(f"Done training subtask {subtask_idx}")

            # everything below here is for updating stuff after training
            # update CM's stochastic transition function
            # local_env.action_space[subtask_idx].agents = agents_to_train
            local_env.action_space[subtask_idx].success_probs = [
                agent.curr_best_eval_success_prob for agent in agents_to_train
            ]
            local_env.action_space[subtask_idx].t_eval = [
                agent.curr_best_eval_t for agent in agents_to_train
            ]

            # # update agents for next CM pass
            ## don't need anymore b/c we're using fresh policies in each iter
            # for agent in agents_to_train:
            #     agent.update_for_next_cm_iter()

            # get the successful final state dist from final subtask in this sequence
            if subtask_idx == subtask_seq[-1]:
                final_subtask_eval_data_list = eval_data
            else:
                # update the estimated initial state distributions for the next state in the CM
                # NOTE: all agents have the same next state here
                agent = agents_to_train[0]
                next_subtask_idx = subtask_seq[subtask_seq.index(subtask_idx) + 1]
                next_agents = local_env.action_space[next_subtask_idx].agents

                for u in local_env.state_space.keys():
                    if agent.subtask_idx in local_env.avail_actions[u]:
                        next_state = local_env.successor[u, agent.subtask_idx]

                self._update_initial_state_dist_data(
                    agents_to_train,
                    next_agents,
                    next_state,
                    eval_data,
                    mu,
                    nu,
                    logger=self.subtask_seq_logger,
                    env=local_env,
                )

                local_env.action_space[subtask_idx].agents = []

        self.subtask_seq_logger.info(f"Dilling the env {subtask_seq}")

        dilled_local_env = dill.dumps(local_env)

        self.subtask_seq_logger.info(f"Done dilling the env {subtask_seq}")

        if output_queue is None:
            return final_subtask_eval_data_list, local_env
        else:
            output_queue.put([final_subtask_eval_data_list, dilled_local_env])

            self.subtask_seq_logger.info(f"Objects put into the queue {subtask_seq}")
            run_time_mins = round((time.time() - start_time) / 60, 2)
            self.subtask_seq_logger.info(
                f"Subtask Sequence {subtask_seq} Runtime: {run_time_mins} minutes"
            )

            # self.subtask_seq_logger.info("Waiting a little bit before closing this process to make sure everything gets put into the output queue")
            # time.sleep(2)
            # self.subtask_seq_logger.info("Done waiting, process should end now")

    def _train_single_subtask(
        self,
        agents_to_train,
        n_train,
        iter_idx,
        gpu_idxs,
        max_parallel_processes_per_gpu,
        eval=False,
    ):
        start_time = time.time()

        # train until at least one of the agents' success probs is above the success threshold
        success_probs = []

        while not any(
            prob >= agents_to_train[0].min_success_prob_threshold
            for prob in success_probs
        ):
            # reset exploration so agents fully explore during each loop of training
            for agent in agents_to_train:
                agent.reset_exploration(iter_idx)

            # assign agents to GPUs for training
            n_agents_per_gpu = self._assign_agents_to_gpus(
                agents_to_train, max_parallel_processes_per_gpu
            )

            # fully train + evaluate agents and identify best policies
            agent_kwargs = {
                "n_train": n_train,
                "eval_interval": self.args.eval_interval,
                "cm_iter": iter_idx,
                "only_eval": eval,
            }

            ######################
            # parallel training
            ######################
            # create your batches
            batch_idx = 0
            agent_idx = 0
            agent_counts_per_gpu = {gpu_idx: 0 for gpu_idx in gpu_idxs}
            batch_args = {}

            # assign each agent to a batch based on agents_to_train_per_gpu and max_parallel_processes_per_gpu
            while agent_idx < len(agents_to_train):
                curr_batch_agent_counts_per_gpu = {gpu_idx: 0 for gpu_idx in gpu_idxs}
                curr_batch_args = []

                # loop through all the GPUs for this batch and assign agents if the GPU still has space
                for gpu_idx in gpu_idxs:
                    # loop until we assign the max number of agents to this gpu in this batch
                    for _ in range(max_parallel_processes_per_gpu[gpu_idx]):
                        # (condition 0) ensure the right number of agents are assigned to each GPU across batches
                        # (condition 1) ensure each GPU will not run more parallel processes than is allocated in each batch
                        if (
                            agent_counts_per_gpu[gpu_idx] < n_agents_per_gpu[gpu_idx]
                        ) and (
                            curr_batch_agent_counts_per_gpu[gpu_idx]
                            < max_parallel_processes_per_gpu[gpu_idx]
                        ):
                            # build kwargs
                            tmp_kwargs = copy.copy(agent_kwargs)
                            tmp_kwargs["device"] = gpu_idx
                            agent = agents_to_train[agent_idx]
                            tmp_args = (agent, tmp_kwargs)
                            curr_batch_args.append(tmp_args)

                            agent_idx += 1
                            agent_counts_per_gpu[gpu_idx] += 1
                            curr_batch_agent_counts_per_gpu[gpu_idx] += 1

                batch_args[f"{batch_idx}"] = curr_batch_args
                batch_idx += 1

            output_data_lists = []

            #######################
            # debugging, sequential training, only works for 1 agent
            #######################
            """
            # processes here means number of CPUs you make available for this multiprocessing
            curr_outputs = self._fully_train_and_eval_agent(batch_args[f"{0}"][0][0], batch_args[f"{0}"][0][1])
            output_data_lists.append([curr_outputs])

            """
            #######################
            # parallel training with an output queue
            #######################
            n_cpus_subtask = int(mp.cpu_count() / len(max_parallel_processes_per_gpu))
            p = mp.Pool(n_cpus_subtask)

            for i, args in batch_args.items():
                self.subtask_seq_logger.info(f"Running batch: {i}/{batch_idx}")
                curr_outputs = p.starmap(self._fully_train_and_eval_agent, args)
                output_data_lists.append(curr_outputs)
            #######################

            # unpack outputs into a single list
            output_data = [
                data for data_list in output_data_lists for data in data_list
            ]

            # process your outputs
            eval_data = [output[1] for output in output_data]
            success_probs = [data.eval_success_prob for data in eval_data]

        agents_to_train = [output[0] for output in output_data]
        run_time_mins = round((time.time() - start_time) / 60, 2)
        self.subtask_seq_logger.info(
            f"Subtask {agents_to_train[0].subtask_idx} Runtime: {run_time_mins} minutes"
        )

        return agents_to_train, eval_data

    def _assign_agents_to_gpus(self, agents, max_subprocesses_per_gpu):
        n_agents = len(agents)
        total_processes = sum(max_subprocesses_per_gpu.values())

        # proportionally split the agents across the GPUs
        props = np.array(list(max_subprocesses_per_gpu.values())) / total_processes
        n_agents_per_gpu = np.round(n_agents * props).astype("int")

        # fix over- and under-counting of agents due to rounding
        ## EX: If you have 6 agents, and round n_agents_per_gpu = [2.5, 3.5], you would get [3, 4].
        ## This is an over-count since the result tells us there are 7 agents when there are only 6.
        ## If you had 10 agents, and round n_agents_per_gpu = [3.3333, 3.3333, 3.3333], you would get [3, 3, 3],
        ## which only accounts for 9 agents when there are 10.
        diff = n_agents - np.sum(n_agents_per_gpu)

        # if n_agents_per_gpu under-counts, add agents to the lowest-indexed GPUs
        if diff > 0:
            gpu_idx = 0
            while diff > 0:
                n_agents_per_gpu[gpu_idx] += 1
                diff -= 1
                gpu_idx += 1

        # if n_agents_per_gpu over-counts, remove agents from the highest-indexed GPUs
        elif diff < 0:
            gpu_idx = len(n_agents_per_gpu) - 1
            while diff < 0:
                n_agents_per_gpu[gpu_idx] -= 1
                diff += 1
                gpu_idx -= 1

        n_agents_per_gpu_dict = {}
        for i, gpu_idx in enumerate(list(max_subprocesses_per_gpu.keys())):
            n_agents_per_gpu_dict[gpu_idx] = n_agents_per_gpu[i]

        return n_agents_per_gpu_dict

    def _fully_train_and_eval_agent(self, agent, agent_kwargs=None):
        eval_data = agent.fully_train_and_eval(**agent_kwargs)
        return (agent, eval_data)
        # removed b/c I don't think copy is necessary
        # agent_out = copy.copy(agent)
        # return (agent_out, eval_data)

    def _update_initial_state_dist_data(
        self, agents, next_agents, next_state, eval_data, mu, nu, logger, env=None
    ) -> None:
        # eval_data is a list of length len(agents) where each entry is an EvalData object
        if env is None:
            env = self.env

        # update init state dists

        # filter out agents that don't meet the success threshold
        agents_filtered = [
            agent
            for agent in agents
            if agent.curr_best_eval_success_prob >= agent.min_success_prob_threshold
        ]
        self.logger.info(
            f"{len(agents_filtered)}/{len(agents)} agents achieved the minimum success threshold"
        )

        new_init_state_dist = self._aggregate_incoming_final_state_dists(
            agents, agents_filtered, eval_data, mu, nu, env=env
        )

        env.state_space[next_state].outgoing_init_state_dist = new_init_state_dist

        for i, next_agent in enumerate(next_agents):
            next_agent.update_init_state_dist(new_init_state_dist)

    def _aggregate_incoming_final_state_dists(
        self, agents, agents_filtered, eval_data, mu, nu, env
    ):
        # ok so here's the context of this function
        ## we are aiming to update the ISD at a state

        ## we have all of the edges incoming to that state in the list "agents"
        ## we have all of the SFSD data associated with the agents in "eval_data"

        ## we need to compute the new ISD based on "agents", "eval_data", "mu", and "nu"

        # setup stuff
        subtask_idxs = list(set([agent.subtask_idx for agent in agents]))
        comms_vals = list(set([agent.comms_val for agent in agents]))
        comms_vals.sort()
        idx_to_comms_val = {i: comms_val for i, comms_val in enumerate(comms_vals)}

        # for each (u, u'), for each comms value, average over all the seeds to get the average SFSD
        ## we can have multiple predecessor u's for u', so we need to keep them separate here
        final_states = {}
        state_counts = {}
        for i, agent in enumerate(agents):
            if (agent.subtask_idx, agent.comms_val) not in final_states.keys():
                final_states[agent.subtask_idx, agent.comms_val] = []
                state_counts[agent.subtask_idx, agent.comms_val] = []

            # go through and pick out each item, the result of final_states[agent.subtask_idx, agent.comms_val] should be a list where each entry is an np array
            final_states[agent.subtask_idx, agent.comms_val] += [
                state for state in eval_data[i].unique_successful_final_states
            ]
            state_counts[agent.subtask_idx, agent.comms_val] += [
                count for count in eval_data[i].final_state_counts
            ]

        # compute the average SFSD for each subtask index and comms value by averaging across the seeds
        unique_states = {}
        total_state_counts = {}

        for subtask_idx in subtask_idxs:
            for comms_val in comms_vals:
                tmp_state_counts = state_counts[subtask_idx, comms_val]

                if (subtask_idx, comms_val) not in unique_states.keys():
                    unique_states[subtask_idx, comms_val] = []
                    total_state_counts[subtask_idx, comms_val] = []

                for i, state in enumerate(final_states[subtask_idx, comms_val]):
                    in_array, idx = self._array_in_list(
                        state, unique_states[subtask_idx, comms_val]
                    )
                    if in_array:
                        total_state_counts[subtask_idx, comms_val][idx] += (
                            tmp_state_counts[i]
                        )
                    else:
                        # add state as a new entry
                        unique_states[subtask_idx, comms_val].append(state)
                        total_state_counts[subtask_idx, comms_val].append(
                            tmp_state_counts[i]
                        )

        # once you have total_state_counts, divide by the number of seeds to get the average count
        ## then you can divide by the sum of state_counts to normalize to a prob dist
        avg_state_probs = {}
        for subtask_idx in subtask_idxs:
            for comms_val in comms_vals:
                avg_state_probs[subtask_idx, comms_val] = total_state_counts[
                    subtask_idx, comms_val
                ] / np.sum(total_state_counts[subtask_idx, comms_val])

        # use nu to pick out the single comms val we're gonna use
        ## or if nu is uniform, just average across all the comms vals
        unique_states_nu = {}
        avg_state_probs_nu = {}

        prob_idx_to_state = {}
        for subtask_idx in subtask_idxs:
            for comms_val in comms_vals:
                state_idx = 0
                for state in unique_states[subtask_idx, comms_val]:
                    prob_idx_to_state[subtask_idx, comms_val, state_idx] = state
                    state_idx += 1

        # i want each state to be associated with a unique index for a given subtask_idx and comms_val
        ## I then want to grab all of the probs associated with that state across the comms vals and put them in 1 list

        # get unique states across comms values for a given subtask
        unique_states_subtask = {}
        for subtask_idx in subtask_idxs:
            tmp_state_list = []
            for comms_val in comms_vals:
                tmp_state_list += [
                    state for state in unique_states[subtask_idx, comms_val]
                ]
            unique_states_subtask[subtask_idx] = np.unique(tmp_state_list, axis=0)

        # get the prob for each final state for a given subtask
        state_idx_to_prob = {}
        for subtask_idx in subtask_idxs:
            # uniform aggregation
            if len(nu) == 0:
                scaling = 1 / len(comms_vals)
                for unique_state_idx, state in enumerate(
                    unique_states_subtask[subtask_idx]
                ):
                    tmp_state_prob = 0
                    for comms_val in comms_vals:
                        for (
                            tmp_subtask_idx,
                            tmp_comms_val,
                            prob_idx,
                        ) in prob_idx_to_state.keys():
                            if (tmp_subtask_idx == subtask_idx) and (
                                tmp_comms_val == comms_val
                            ):
                                prob_state = prob_idx_to_state[
                                    tmp_subtask_idx, tmp_comms_val, prob_idx
                                ]
                                if np.array_equal(state, prob_state):
                                    tmp_state_prob += (
                                        scaling
                                        * avg_state_probs[subtask_idx, comms_val][
                                            prob_idx
                                        ]
                                    )

                    state_idx_to_prob[subtask_idx, unique_state_idx] = tmp_state_prob

                # the format I want is a list of states and a list of probs
                tmp_final_states = unique_states_subtask[subtask_idx]
                tmp_state_probs = []
                for unique_state_idx, _ in enumerate(
                    unique_states_subtask[subtask_idx]
                ):
                    tmp_state_probs.append(
                        state_idx_to_prob[subtask_idx, unique_state_idx]
                    )

            else:
                # use the data from the single comms val
                for u, mu_subtask_idx in mu.keys():
                    if subtask_idx == mu_subtask_idx:
                        comms_val = idx_to_comms_val[nu[subtask_idx]]
                        tmp_final_states = unique_states[subtask_idx, comms_val]
                        tmp_state_probs = avg_state_probs[subtask_idx, comms_val]

            unique_states_nu[subtask_idx] = tmp_final_states
            avg_state_probs_nu[subtask_idx] = tmp_state_probs

        # use the high-level policy to weight the state aggregation across incoming edges
        # get unique states across subtasks
        tmp_state_list = []
        for subtask_idx in subtask_idxs:
            tmp_state_list += [state for state in unique_states_subtask[subtask_idx]]
        unique_states_final = np.unique(tmp_state_list, axis=0)

        # scale the prob based on the policy probs assigned to the incoming edges to the state
        total_incoming_prob = 0
        for subtask_idx in subtask_idxs:
            for u, mu_subtask_idx in mu.keys():
                if subtask_idx == mu_subtask_idx:
                    total_incoming_prob += mu[u, mu_subtask_idx]

        prob_scaling = {}
        for subtask_idx in subtask_idxs:
            for u, mu_subtask_idx in mu.keys():
                if subtask_idx == mu_subtask_idx:
                    if total_incoming_prob == 0:
                        prob_scaling[mu_subtask_idx] = 1
                        # prob_scaling[mu_subtask_idx] = mu[u, mu_subtask_idx]
                    else:
                        prob_scaling[mu_subtask_idx] = (
                            mu[u, mu_subtask_idx] / total_incoming_prob
                        )

        #  aggregate the probs now using the prob scaling
        avg_state_probs_final = []
        state_idx_to_prob = {}
        for subtask_idx in subtask_idxs:
            scaling = prob_scaling[subtask_idx]
            for unique_state_idx, state in enumerate(unique_states_final):
                if unique_state_idx not in state_idx_to_prob.keys():
                    state_idx_to_prob[unique_state_idx] = 0

                for prob_idx, prob_state in enumerate(unique_states_nu[subtask_idx]):
                    if np.array_equal(prob_state, state):
                        state_idx_to_prob[unique_state_idx] += (
                            prob_scaling[subtask_idx]
                            * avg_state_probs_nu[subtask_idx][prob_idx]
                        )

        # define distribution using the same format as the env config file
        incoming_final_state_dist = []
        for i, state in enumerate(unique_states_final):
            incoming_final_state_dist.append(
                [state_idx_to_prob[i], unique_states_final[i].tolist()]
            )

        return incoming_final_state_dist

    def _get_distance_metric(self):
        distances = []
        for u in self.env.state_space.keys():
            if u != self.env.u_fail:
                # get distance between distributions
                curr_dist = self.env.state_space[u].outgoing_init_state_dist
                prev_dist = self.prev_state_space_data[u].outgoing_init_state_dist
                curr_vec, prev_vec = self._state_dists_to_vectors(curr_dist, prev_dist)
                d = np.linalg.norm(curr_vec - prev_vec, ord=2)
                distances.append(d)

        max_dist = max(distances)
        return max_dist

    def _state_dists_to_vectors(self, curr_dist, prev_dist):
        curr_states, curr_probs = self._unpack_state_dist(curr_dist)
        prev_states, prev_probs = self._unpack_state_dist(prev_dist)

        # index in the final vectors
        state_idx = 0
        idx_to_state = {}
        for state in curr_states + prev_states:
            if state not in idx_to_state.values():
                idx_to_state[state_idx] = state
                state_idx += 1

        # these hold the prob data at the correct index
        # both vectors will end up being the max of the unique indices in length
        curr_vec = np.zeros(state_idx)
        prev_vec = np.zeros(state_idx)

        for state_idx, state in idx_to_state.items():
            if state in curr_states:
                curr_vec[state_idx] = curr_probs[curr_states.index(state)]
            if state in prev_states:
                prev_vec[state_idx] = prev_probs[prev_states.index(state)]

        return curr_vec, prev_vec

    def _unpack_state_dist(self, state_dist):
        states, probs = [], []

        for item in state_dist:
            probs.append(item[0])
            states.append(item[1])
        return states, probs

    def _array_in_list(self, my_arr, list_arrays):
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        in_array, idx = False, None
        for i, arr in enumerate(list_arrays):
            if np.array_equal(arr, my_arr):
                in_array, idx = True, i

        return in_array, idx
        # return next((True for elem in list_arrays if np.array_equal(elem, my_arr)), False)


class SubtaskAgentDependent(object):
    """
    Abstract class representing a team of low-level agents in the PYMARL framework that learn to accomplish goal-oriented sub-tasks.
    """

    def __init__(
        self,
        subtask_idx,
        success_condition,
        args,
        init_state_dist=None,
        seed=None,
        comms_val=None,
        load_path=None,
        load_step=None,
        _log=None,
    ):
        self.subtask_idx = subtask_idx
        self.success_condition = success_condition
        self.args = args
        self.min_success_prob_threshold = self.args.min_success_prob_threshold
        self.seed = seed
        self.init_state_dist = init_state_dist
        self.final_state_dist = None
        self.comms_val = comms_val
        self.load_path = load_path
        self.load_step = load_step

        self.agent_setup_complete = False

        # current-best model data
        self.curr_best_eval_success_prob = -1
        self.curr_best_eval_t = None

        self.subtask_dir = os.path.join(
            args.base_dir,
            f"subtask_{self.subtask_idx}_seed_{self.seed}_comms_{self.comms_val}",
        )
        self.eval_save_dir = os.path.join(
            args.checkpoint_dir,
            "eval",
            f"subtask_{self.subtask_idx}_seed_{self.seed}_comms_{self.comms_val}",
        )
        self.data = {
            "total_training_steps": 0,
            "episode": 0,
        }

    def _build_logger(self, base_dir, cm_iter):
        # log experiment-level info
        # parallel processes info is logged in the individual agent classes
        log_dir = os.path.join(base_dir, f"iter_{cm_iter}", "logs")
        os.makedirs(log_dir, exist_ok=True)
        console_logger = get_console_logger(
            f"logger_subtask_{self.subtask_idx}_seed_{self.seed}_comms_{self.comms_val}"
        )
        file_output_handler = logging.FileHandler(
            os.path.join(log_dir, "low_level_optimization_logs.txt")
        )
        console_logger.addHandler(file_output_handler)
        self.logger = Logger(console_logger, log_dir)
        self.runner.logger = self.logger
        self.learner.logger = self.logger

    def _agent_process_setup(self):
        # sets up stuff needed for the agent to train in an independent process.
        # I want the agent's seed to be used to set a global seed (but within the scope of an independent process)
        # for pytorch to initialize the agent's weights.
        # This should  only be called the first time the agent is trained in a separate process
        # after the first time, the randomness is in numpy rng generators, so we don't need to reset the randomness for pytorch each time
        # th.manual_seed(self.seed)
        th.manual_seed(self.seed)
        if self.load_path is not None:
            self.load(self.load_path, self.load_step)
        else:
            if self.init_state_dist is not None:
                if isinstance(self.init_state_dist, list):
                    self._reformat_init_state_dist()
            self.init_learning_alg_and_env(self.args)

        self.agent_setup_complete = True

    def fully_train_and_eval(
        self, n_train, eval_interval, cm_iter, device=None, only_eval=False
    ):
        if not self.agent_setup_complete:
            self._agent_process_setup()

        # set up logger for this individual agent for this process
        self._build_logger(self.subtask_dir, cm_iter)
        self.to(device)

        if not eval:
            # run training loop
            t_max = self.runner.t_env + n_train
            while self.runner.t_env <= t_max:
                self.train(eval_interval)
                # evaluate to estimate p_success
                ## this eval_data is not used later on, since it would only be returned if success isn't above the threshold
                eval_data = self.evaluate(
                    self.args.n_eval_ep_success, cm_iter, save=True
                )
        else:
            eval_data = self.evaluate(self.args.n_eval_ep_success, cm_iter, save=True)

        # evaluate best policy from training to estimate successful final state dist
        if self.curr_best_eval_success_prob >= self.min_success_prob_threshold:
            self.logger.console_logger.info("Evaluating best model")
            eval_data = self.load_and_eval_best_model(
                self.args.n_eval_ep_successful_final_state, cm_iter, device
            )

        self.logger.console_logger.info(
            "Done evaluating best model, fully_train_and_eval done"
        )

        self.to("cpu")
        return eval_data

    def train(self, train_timesteps=1e4):
        # start training
        t_start = self.data["total_training_steps"]
        t_end = t_start + train_timesteps

        self.last_test_time = -self.args.test_interval - 1
        last_log_time = t_start
        model_save_time = self.runner.t_env

        self.start_time = time.time()
        self.last_time = self.start_time
        self.logger.console_logger.info(
            f"Beginning training agent {self.subtask_idx}, {self.seed}, {self.comms_val} for {train_timesteps} timesteps"
        )

        while self.runner.t_env <= t_end:
            # self.logger.console_logger.info(f"global pytorch seed: {th.seed()}")
            # self.logger.console_logger.info(f"agent seed: {self.seed}")
            # Run for a whole episode at a time
            episode_batch = self.runner.run(test_mode=False)
            self.buffer.insert_episode_batch(episode_batch)

            if self.buffer.can_sample(self.args.batch_size):
                episode_sample = self.buffer.sample(self.args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != self.args.device:
                    episode_sample.to(self.args.device)

                self.learner.train(
                    episode_sample, self.runner.t_env, self.data["episode"]
                )

            # log the time and remaining time every once in a while
            if (
                self.runner.t_env - self.last_test_time
            ) / self.args.test_interval >= 1.0:
                # self.logger.console_logger.info(
                #     "t_env: {} / {}".format(self.runner.t_env, self.args.t_max)
                # )
                self.logger.console_logger.info(
                    "Estimated time left: {}. Time passed: {}".format(
                        time_left(
                            self.last_time,
                            self.last_test_time,
                            self.runner.t_env,
                            self.args.t_max,
                        ),
                        time_str(time.time() - self.start_time),
                    )
                )

                self.last_time = time.time()
                self.last_test_time = self.runner.t_env

            self.data["episode"] += self.args.batch_size_run

            if (self.runner.t_env - last_log_time) >= self.args.log_interval:
                self.logger.log_tb("episode", self.data["episode"], self.runner.t_env)
                self.logger.print_recent_stats()
                last_log_time = self.runner.t_env

        self.data["total_training_steps"] = self.runner.t_env

        self.runner.close_env()
        self.logger.console_logger.info(
            f"Finished training subtask {self.subtask_idx}, {self.seed}, {self.comms_val}"
        )
        self.logger.writer.flush()

    def run_env_manual(self):
        self._agent_process_setup()
        self.init_learning_alg_and_env(self.args)
        self.runner.run()

    def evaluate(self, n_eval_ep, cm_iter, save=False, eval_best=False):
        n_test_success = 0
        successful_final_states = []

        if (not eval_best) and self.args.save_replay:
            replay_dir = os.path.join(
                self.subtask_dir,
                f"iter_{cm_iter}",
                "replays",
                f"{self.runner.t_env}",
            )

            self.logger.console_logger.info(f"Saving replay gifs to {replay_dir}")

        for ep in range(n_eval_ep):
            if ep % 250 == 0:
                self.logger.console_logger.info(f"Eval Ep.: {ep}/{n_eval_ep}")

            _, ep_info = self.runner.run(
                test_mode=True, eval_best=eval_best, n_eval_ep=n_eval_ep
            )

            # save the replays of the first few test episodes
            if ep in range(0, 10) and self.args.save_replay:
                if self.args.train and save:
                    self.runner.save_replay(replay_dir, ep)

            if ep_info["success"]:
                n_test_success += 1
                successful_final_states.append(ep_info["final_obs"])

        eval_success_prob = n_test_success / n_eval_ep

        unique_final_states, counts = np.unique(
            np.array(successful_final_states), return_counts=True, axis=0
        )

        eval_data = EvalData(
            eval_success_prob, ep_info["t_env"], unique_final_states, counts
        )

        # only save the current-best model weights
        ## this saves a ton of computer disk space compared to saving every time we evaluate
        if (
            (not eval_best)
            and (eval_data.eval_success_prob > self.curr_best_eval_success_prob)
            and (eval_data.eval_success_prob >= self.min_success_prob_threshold)
        ):
            self.update_curr_best_model(eval_data.eval_success_prob, eval_data.t_eval)

            # save model weights for evaluated run
            if self.args.train and save:
                self.save(cm_iter)

        return eval_data

    def update_curr_best_model(self, eval_success_prob: float, t_eval: int):
        self.curr_best_eval_success_prob = eval_success_prob
        self.curr_best_eval_t = t_eval

    def load_and_eval_best_model(self, n_eval_ep, cm_iter, device):
        """
        second stage of evaluation, samples successful evaluation episodes to construct accurate estimates of the successful final state distributions
        """
        # same dir as in save()
        load_path = os.path.join(self.subtask_dir, f"iter_{cm_iter}", "models")
        self.load(load_path, self.curr_best_eval_t)
        self.to(device)
        eval_data = self.evaluate(n_eval_ep, cm_iter, save=False, eval_best=True)

        return eval_data

    def update_init_state_dist(self, init_state_dist):
        # if self.init_state_dist is None:
        #     # first time setup
        #     self.init_state_dist = init_state_dist
        #     self._reformat_init_state_dist()
        #     self.init_learning_alg_and_env(self.args)

        # else:
        # all other times
        self.init_state_dist = init_state_dist
        self._reformat_init_state_dist()

        tmp_args = copy.copy(self.args)
        env_args = tmp_args.env_args
        env_args.update({"init_state_dist": self.init_state_dist})
        self.args = tmp_args

        # init the runner
        self.init_learning_alg_and_env(self.args)

        # update init state dist in the env
        self.runner.update_init_state_dist(self.args)

    def init_learning_alg_and_env(self, args, env=None):
        # define environment for this subtask in the format needed by the environment class
        tmp_args = copy.copy(args)

        env_args = tmp_args.env_args
        env_args.update({"init_state_dist": self.init_state_dist})
        env_args.update({"success_condition": self.success_condition})
        env_args.update({"subtask_idx": self.subtask_idx})
        env_args.update({"seed": self.seed})

        # envs that use comms and take comms_val as an arg
        # if args.env in comms_envs:
        #     env_args.update({"comms_val": self.comms_val})

        try:
            del tmp_args.comms_val
        except:
            pass
        tmp_args.comms_val = self.comms_val

        if tmp_args.manual_control:
            # Set up schemes and groups here
            self.runner = r_REGISTRY["manual"](args=tmp_args)

            # bunch of stuff to define the mac
            env_info = self.runner.get_env_info()
            tmp_args.n_agents = env_info["n_agents"]
            tmp_args.n_actions = env_info["n_actions"]
            tmp_args.state_shape = env_info["state_shape"]
            tmp_args.obs_shape = env_info["obs_shape"]

            # Default/Base scheme
            scheme = {
                "state": {"vshape": env_info["state_shape"]},
                "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {
                    "vshape": (env_info["n_actions"],),
                    "group": "agents",
                    "dtype": th.int,
                },
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            self.groups = {"agents": tmp_args.n_agents}
            self.preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=tmp_args.n_actions)])
            }

            self.buffer = ReplayBuffer(
                scheme,
                self.groups,
                tmp_args.buffer_size,
                env_info["episode_limit"] + 1,
                preprocess=self.preprocess,
                device="cpu" if tmp_args.buffer_cpu_only else tmp_args.device,
                seed=self.seed,
            )

            # setup multiagent controller here
            mac = mac_REGISTRY[tmp_args.mac](self.buffer.scheme, self.groups, tmp_args)
            self.runner.setup(scheme, self.groups, self.preprocess, mac)

        else:
            # create the runner and the training env with the specified args in args.env_args
            self.runner = r_REGISTRY[tmp_args.runner](args=tmp_args)

            # Set up schemes and groups here
            env_info = self.runner.get_env_info()
            tmp_args.n_agents = env_info["n_agents"]
            tmp_args.n_actions = env_info["n_actions"]
            tmp_args.state_shape = env_info["state_shape"]
            tmp_args.obs_shape = env_info["obs_shape"]
            tmp_args.epsilon_anneal_time = self.args.epsilon_anneal_time_per_iter

            # Default/Base scheme
            self.scheme = {
                "state": {"vshape": env_info["state_shape"]},
                "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {
                    "vshape": (env_info["n_actions"],),
                    "group": "agents",
                    "dtype": th.int,
                },
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            self.groups = {"agents": tmp_args.n_agents}
            self.preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=tmp_args.n_actions)])
            }

            self.buffer = ReplayBuffer(
                self.scheme,
                self.groups,
                tmp_args.buffer_size,
                env_info["episode_limit"] + 1,
                preprocess=self.preprocess,
                device="cpu" if tmp_args.buffer_cpu_only else tmp_args.device,
                seed=self.seed,
            )

            # setup multiagent controller here
            self.mac = mac_REGISTRY[tmp_args.mac](
                self.buffer.scheme, self.groups, tmp_args
            )

            # Give runner the scheme
            self.runner.setup(
                scheme=self.scheme,
                groups=self.groups,
                preprocess=self.preprocess,
                mac=self.mac,
            )

            # Learner
            self.learner = le_REGISTRY[tmp_args.learner](
                self.mac, self.buffer.scheme, logger=None, args=tmp_args
            )

            # if self.args.use_cuda:
            #     self.learner.cuda()

    def reset_exploration(self, cm_iter: int):
        try:
            # resets exploration so the agent continues to explore more
            ## used when some minimum performance threshold is not met
            self.runner.t_exploration_start = self.runner.t_env

            # create the new action selector
            tmp_args = copy.copy(self.args)
            if cm_iter == 0:
                tmp_args.epsilon_anneal_time = self.args.epsilon_anneal_time_per_iter
            else:
                tmp_args.epsilon_anneal_time = self.args.epsilon_anneal_time_stage_1

            new_action_selector = action_REGISTRY[self.args.action_selector](tmp_args)
            self.mac.action_selector = new_action_selector
        except:
            # agent has not yet initialized the runner, so doesn't make sense
            pass

    def _reformat_init_state_dist(self):
        tmp = {}
        for i, init_state_dist in enumerate(self.init_state_dist):
            tmp[i] = {"prob": init_state_dist[0], "init_state": init_state_dist[1]}
        self.init_state_dist = tmp

    # def update_for_next_cm_iter(self):
    #     self.curr_best_eval_success_prob = -1
    #     self.curr_best_eval_t = None

    def save(self, cm_iter):
        if self.load_path is None:
            save_path = os.path.join(
                self.subtask_dir,
                f"iter_{cm_iter}",
                "models",
                str(self.runner.t_env),
            )
        else:
            save_path = os.path.join(self.load_path, str(self.runner.t_env))

        os.makedirs(save_path, exist_ok=True)
        self.logger.console_logger.info(f"Saving model to {save_path}")

        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        self.learner.save_models(save_path)

        # save agent class to disk
        agent_file = os.path.join(save_path, "agent_data.p")

        agent_data = {
            "subtask_idx": self.subtask_idx,
            "init_state_dist": self.init_state_dist,
            "success_condition": self.success_condition,
            "data": self.data,
        }

        with open(agent_file, "wb") as pickle_file:
            pickle.dump(agent_data, pickle_file)

    def load(self, load_path, load_step):
        """
        load_path: checkpoint_dir, path to the "models" folder for this model and environment
        load_step: int, time step of model to load
        """
        # pick timestep to load
        timestep_to_load = load_step

        # # TODO this doesn't catch what I care about. If the load directory is a directory, but the subtask_agent isn't saved there, it just returns and fails later
        # if not os.path.isdir(load_path):
        #     self.logger.console_logger.info(
        #         f"Checkpoint directory {load_path} doesn't exist"
        #     )
        #     return

        # load subtask agent class
        agent_file = os.path.join(load_path, str(timestep_to_load), "agent_data.p")
        with open(agent_file, "rb") as pickle_file:
            agent_data = pickle.load(pickle_file)

        self.subtask_idx = agent_data["subtask_idx"]
        self.init_state_dist = agent_data["init_state_dist"]
        self.success_condition = agent_data["success_condition"]
        self.data = agent_data["data"]

        agent_data = {
            "subtask_idx": self.subtask_idx,
            "init_state_dist": self.init_state_dist,
            "success_condition": self.success_condition,
            "data": self.data,
        }

        # load environment with previously used info
        self.init_learning_alg_and_env(self.args)

        # load model
        self.model_path = os.path.join(load_path, str(timestep_to_load))
        self.logger.console_logger.info(f"Loading model from {self.model_path}")
        self.runner.logger = self.logger
        self.learner.logger = self.logger
        self.learner.load_models(self.model_path)
        self.runner.t_env = timestep_to_load

    def to(self, device):
        # adds this agent to a specified device
        self.learner.to(device)
        self.buffer.to(device)
        self.runner.args.device = device

        try:
            self.runner.batch.to(device)
        except:
            pass

        self.args.device = device
