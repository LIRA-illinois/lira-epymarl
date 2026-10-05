# Literature Research Handoff: Dependent Subtask Case

## Research Goal
Identify established methods for hierarchical planning and multi-agent reinforcement learning when a successor subtask's initial-state distribution depends on the empirical successful terminal states of predecessor subtasks.

The key question is whether the current iterative dependent-subtask design corresponds to an existing method, and which principled alternatives should replace the heuristic CM update.

## Repository Context

This is a Python/PyTorch EPyMARL project with a tabular high-level project MDP and learned low-level navigation policies.

Target DAG:

```text
0 -> 1 -> 3 -> 4
 \-> 2 -> 3
```

The concrete edge sequence is:

```python
[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
```

For each edge `(u, v)`:

1. Train a fresh low-level policy.
2. Evaluate successful episodes.
3. Estimate the empirical distribution of terminal joint-agent positions.
4. Use that distribution as the spawn distribution for outgoing subtasks from `v`.

At merge state `3`, predecessor terminal distributions are combined according to high-level policy/occupancy weights.

## Current Implementation

Relevant files:

- [src/simulation/run.py](src/simulation/run.py): `Simulation.train_dependent_subtasks`
- [src/simulation/evaluate.py](src/simulation/evaluate.py): `collect_successful_final_state_dist`
- [submodules/gym-multigrid/gym_multigrid/envs/team_navigation.py](submodules/gym-multigrid/gym_multigrid/envs/team_navigation.py): dynamic navigation spawn distributions and terminal `final_state`
- [src/envs/basic_gymnasium_wrappers.py](src/envs/basic_gymnasium_wrappers.py): reset-option forwarding
- [src/modules/agents/ilp_model.py](src/modules/agents/ilp_model.py): existing Gurobi high-level occupancy/policy optimizer
- `agent_dependent.py`: reference implementation from LIRA-illinois/marl

The current code now supports bounded CM iterations:

- Each iteration retrains all edges.
- Edge success rates and terminal distributions are measured.
- A high-level edge policy is derived from downstream success probabilities.
- End-to-end hierarchical success is evaluated analytically after each iteration.
- The policy from iteration `k` weights successor initial-state distributions in iteration `k+1`.
- Convergence is based on the maximum total-variation distance between successive successor-state distributions.

This is deliberately a lightweight replacement for the reference implementation's Gurobi CM optimization. The current policy helper chooses the outgoing edge(s) with the highest estimated downstream success and splits ties uniformly.

## Reference Method

`agent_dependent.py::CMAgentDependent.optimize_cm` repeatedly:

1. Solves the high-level optimization model using the previous low-level success data.
2. Extracts high-level occupancy/policy variables `mu` and `nu`.
3. Runs a CM pass that trains low-level subtask policies.
4. Aggregates successful final-state distributions from incoming edges using `mu`/`nu` weights.
5. Checks convergence of initial-state distributions.
6. Repeats until convergence or `max_iter`.
7. Evaluates the resulting hierarchical policy after each iteration.

Important reference concepts to investigate:

- CM agent / coordination-manager optimization
- occupancy measures `mu`
- communication/action selection `nu`
- stochastic shortest-path MDPs
- empirical successor-state distributions
- distribution shift caused by chaining learned skills
- hierarchical policy iteration with learned transition models

## Literature Questions

1. What established framework best describes this problem?
   - Options may include options/SMDP planning, skill chaining, hierarchical RL, stochastic shortest-path planning, successor-feature methods, distributional RL, compositional task planning, or multi-agent coordination.

2. Is the empirical terminal-state distribution a known object?
   - Search for terms such as `successor state distribution`, `terminal state distribution`, `subtask transition distribution`, `skill effect model`, `option termination distribution`, `occupancy measure`, and `abstract state transition model`.

3. What is the principled update at a merge state?
   - Current implementation uses a weighted mixture of predecessor terminal distributions.
   - Compare this with Bayesian model averaging, occupancy-measure weighting, mixture-of-options models, flow conservation, belief-state updates, and maximum-entropy policy selection.

4. How should high-level policy optimization use low-level empirical success rates?
   - Compare the current greedy downstream-success policy with LP/ILP occupancy optimization, stochastic shortest-path dynamic programming, policy iteration, value iteration, constrained MDPs, and risk-sensitive planning.

5. How should uncertainty and finite evaluation samples be handled?
   - Look for confidence bounds, robust MDPs, Bayesian transition estimation, Dirichlet posterior updates, distributionally robust optimization, and safe policy improvement.

6. Does the iterative loop have an established interpretation?
   - Possible descriptions: generalized policy iteration, approximate policy iteration, block coordinate descent, bilevel optimization, expectation-maximization-like alternating optimization, or fixed-point iteration over induced state distributions.

7. How should communication budgets be integrated?
   - The project has per-agent message budgets and an existing `ILPModel` that selects high-level transitions and communication allocations.
   - Search for constrained communication MARL, communication-aware options, coordination graphs, decentralized execution with centralized training, and resource-constrained task planning.

8. What methods explicitly handle branching DAGs rather than simple chains?
   - Prioritize papers with task graphs, AND/OR graphs, precedence constraints, workflow planning, multi-agent task allocation, or compositional skills under stochastic outcomes.

## Research Deliverables

Return a concise technical report containing:

1. A taxonomy of 3-5 relevant method families.
2. The closest prior method to this repository's dependent-subtask loop.
3. At least 5 high-quality references, preferably primary papers or authoritative books/tutorials.
4. For each reference:
   - problem setting,
   - how it represents subtask effects,
   - how it updates high-level policy,
   - whether it handles stochastic terminal distributions,
   - whether it handles branching task graphs,
   - whether it handles multi-agent communication/resource constraints.
5. A recommendation for replacing or formalizing `_build_cm_policy`, `_evaluate_cm_policy`, and `_merge_state_dists`.
6. A recommendation for uncertainty handling when successful terminal-state samples are sparse.
7. A mapping from the recommended method to concrete repository changes.

## Search Terms

Use combinations of:

- `hierarchical reinforcement learning stochastic subtask transition distribution`
- `options framework termination state distribution`
- `skill chaining successor state distribution reinforcement learning`
- `multi-agent task graph stochastic outcomes occupancy measure`
- `stochastic shortest path occupancy measure branching task graph`
- `hierarchical reinforcement learning empirical transition model skills`
- `compositional reinforcement learning task graph options`
- `constrained MDP communication budget multi-agent reinforcement learning`
- `coordination manager multi-agent reinforcement learning task decomposition`
- `robust planning learned skills distribution shift`
- `Bayesian estimation terminal state distribution options`
- `approximate policy iteration hierarchical reinforcement learning`

## Important Caveats

- Do not assume that `ILPModel` is automatically the correct replacement. It currently contains hard `sys.exit()` behavior on infeasibility and is not yet wired into the iterative dependent loop.
- Distinguish an analytically evaluated high-level success probability from an actual end-to-end environment rollout. The current implementation evaluates the former because separately trained edge policies are not yet composed into one executable high-level controller.
- Check whether a cited method assumes a fixed abstract state after a skill, while this project needs a distribution over joint low-level agent positions.
- Check whether methods assume Markovian abstract states. Here, the same high-level state may induce different low-level spawn distributions depending on predecessor history and policy iteration.
- Account for the fact that each edge currently trains a fresh policy from scratch on each CM iteration.
- Keep literature findings separate from implementation recommendations, and do not modify repository code during research.
