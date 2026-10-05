# Handoff: Dependent-Subtask CM Training

## Goal and current state
The user is implementing iterative dependent-subtask training for team navigation. Successor-task spawn distributions should be estimated from successful terminal joint states of trained predecessor tasks. Two experiment configs exist:

- Branching DAG: `experiments/exp_44/exp_config.yaml`, using `navigation_branching_small_hall.yaml` and edges `0->1, 0->2, 1->3, 2->3, 3->4`.
- Hallway chain: `experiments/exp_45/exp_config.yaml`, using `navigation_hallway_learned.yaml` and edges `0->1->2->3->4`. Only the first task declares an external initial distribution; later tasks intentionally omit it.

The code includes bounded CM iterations, empirical successful-final-state collection, weighted merging of predecessor distributions, an ILP-backed high-level policy attempt, and analytical fallback. Python compilation and YAML-shape checks have passed in prior work. **No end-to-end dependent training run or successful live Gurobi solve has been verified.**

## Immediate blocker
The hallway experiment currently cannot load its task catalog. In `submodules/gym-multigrid/gym_multigrid/utils/subtasks.py`, `NavigationTaskData.init_state_dist` has been made optional, and the catalog parser uses `config.get("init_state_dist")`, but `NavigationTaskData.__post_init__` still unconditionally executes:

```python
if len(self.init_state_dist.states) == 0:
```

When loading `1->2`, `2->3`, or `3->4` this dereferences `None`. First fix: only validate non-empty states when `init_state_dist is not None`; keep a clear error in `_spawn_navigation_agents` if a task is actually started without either configured or learned distribution. Then load the hallway config through `NavigationTaskCatalog.from_configs` (not YAML parsing alone) and verify only the root task has a distribution.

## Implementation map

- `src/simulation/run.py`
  - `run()` dispatches on `hl_task_sequence` to `train_dependent_subtasks`.
  - Training walks edges in given topological order, rebuilds simulation components for each edge, trains an independent low-level policy per edge, collects successful terminal-state distributions, and passes learned distributions to successor resets.
  - Multiple predecessor distributions are merged using estimated upstream occupancy, edge policy, and predecessor success rate.
  - CM passes are bounded by `cm_max_iterations`, with convergence checked by initial-distribution distance.
  - `_optimize_dependent_cm_policy` attempts `ILPModel` and falls back to `_build_cm_policy` when the solve fails.
- `src/simulation/evaluate.py`
  - `collect_successful_final_state_dist` evaluates a trained subtask and returns `{"states": ..., "probs": ...}` plus success rate.
- `submodules/gym-multigrid/gym_multigrid/utils/subtasks.py`
  - Defines `NavigationTaskData` and parses catalog YAML. It currently contains the optional-distribution bug above.
- `submodules/gym-multigrid/gym_multigrid/envs/team_navigation.py`
  - Learned `navigation_init_state_dist` overrides task catalog spawn distribution; missing both produces a clear `ValueError`.
- `src/modules/agents/ilp_model.py`
  - Existing Gurobi occupancy optimizer; infeasibility now raises `RuntimeError` rather than calling `sys.exit`.
- `experiments/exp_44/exp_config.yaml`
  - Branching case, 1M test/save cadence, `success_rate_spec: 0.85`, CM limit 5, epsilon 0.2.
- `experiments/exp_45/exp_config.yaml`
  - Hallway case, same general training settings, dependent chain `[[0,1],[1,2],[2,3],[3,4]]`.
- `submodules/gym-multigrid/gym_multigrid/envs/maps/team_navigation/navigation_hallway_learned.yaml`
  - Chain catalog. Only `0->1` has `init_state_dist`; successors omit it intentionally.
- `submodules/gym-multigrid/gym_multigrid/envs/maps/team_navigation/navigation_branching_small_hall.yaml`
  - Branching catalog; all tasks currently specify distributions, preserving old catalog behavior.

## High-level policy caveats
1. The dependent loop measures one success rate per edge and copies that same rate to every configured communication budget. The ILP's budget choice is therefore not supported by budget-specific empirical success data.
2. `_optimize_dependent_cm_policy` accesses `self.runner.env.hlmdp`. This may fail for process-backed runners such as `SharedMemoryParallelRunner`, which do not expose `.env`; the exception is caught and silently causes analytical fallback (with a log message). To actually use ILP there, retain/build an MDP reference independently or retrieve it through runner APIs.
3. Policy extraction uses edge task occupancy, normalizing outgoing occupancy to probabilities. The extracted communication-budget policy is not used by the dependent rollout/training path.
4. The analytical fallback is greedy maximum downstream success, with uniform tie splitting.
5. The reference implementation's `mu`/`nu` semantics and convergence metric have not been fully matched or reviewed.
6. The ILP helper was not successfully smoke-tested: an import/API inspection attempt exceeded a 30-second execution window before Gurobi was reached.

## Known modeling/runtime considerations
- Every edge gets a fresh low-level policy and independent training budget; this is not continual fine-tuning or warm-starting.
- CM recomputes edge policies and learned distributions each iteration. No full composed hierarchical rollout has been validated.
- Evaluation and full training are expensive with current experiment parameters (`t_max=5,050,000`, 50 eval episodes per edge, up to 5 CM iterations). For first integration checks, create a temporary minimal config or use a short programmatic test; do not overwrite the main experiments.
- Check `episode_runner.py` and `shared_memory_parallel_runner.py` evaluation return contracts if `log_stats` / terminal info issues recur.
- Confirm the topological ordering and DAG assumptions when testing other task graphs.

## Next actions
1. Fix the `NavigationTaskData.__post_init__` optional handling and add/run a narrow catalog-load test for both navigation YAML files.
2. Run a tiny navigation reset with `navigation_init_state_dist` omitted on a successor and supplied through reset options; also confirm a reset with no supplied distribution fails clearly.
3. Run a minimal `train_dependent_subtasks` smoke test with short `t_max` and small evaluation counts, first with the episode runner, then process-backed runner if that is an intended target.
4. Address ILP MDP access for the selected runner, and run a minimal Gurobi solve on the branching DAG. Confirm whether the log says `policy source: ILP` or fallback.
5. Decide whether policy fidelity requires budget-specific edge success evaluations and actual `mu`/`nu` extraction/rollout.
6. Update this handoff and the existing `scratch/agent_handoffs/HANDOFF_dependent_subtasks.md` after runtime evidence is available.

## Validation already reported
- Python compile checks passed for the files touched in the prior conversation, including `run.py`, `ilp_model.py`, `team_navigation.py`, and `utils/subtasks.py`.
- YAML validation passed for `exp_44` and `exp_45`, checking navigation-config paths, the five-edge branching DAG and four-edge chain, and parameter values.
- The new hallway YAML was checked for a root-only configured initial distribution; however, only YAML shape was tested, not successful catalog instantiation. The `__post_init__` bug above was noticed while preparing this handoff.
- Full dependent training and a live ILP solve remain unverified.

## Worktree note
The latest `git status --short` returned no output in the current environment. On the other computer, inspect status before editing; do not assume earlier unrelated Ray-launcher/reference-file changes are still present. The user-provided `agent_dependent.py` is a reference, not intended as a production module.
