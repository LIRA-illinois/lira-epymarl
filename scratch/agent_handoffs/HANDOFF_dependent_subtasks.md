# Handoff: Dependent-Subtask Training Loop

## Goal
Implement a training loop where, for a DAG of HL subtasks, each subtask's initial
state distribution is learned from the empirical distribution of *successful*
terminal states of its predecessor subtask(s)' trained policies. Modeled on
`agent_dependent.py` from `LIRA-illinois/marl`. Target example is the 5-subtask
branching DAG in [experiments/exp_36/exp_config.yaml](experiments/exp_36/exp_config.yaml)
with edges `0→1, 0→2, 1→3, 2→3, 3→4` (node 3 has two predecessors: 1 and 2).

## Status: iterative CM integration implemented, runtime smoke test still pending
The dependent-subtask plumbing, bounded CM iteration loop, and runnable example
config are present. The six touched Python files compile, and the new YAML
parses to the expected DAG. Pure CM-helper validation passes on the target DAG.
**No actual multi-iteration training run has been executed yet.**

## Files changed

### `submodules/gym-multigrid/gym_multigrid/envs/team_navigation.py` — clean
- `_gen_grid(..., navigation_init_state_dist: PositionDist | None = None)`
- `_spawn_navigation_agents(..., init_state_dist=None)`: uses
  `spawn_dist = init_state_dist or task.init_state_dist`
- `reset()`: parses `options["navigation_init_state_dist"]` dict into a
  `PositionDist`
- `_get_info()`: adds `info["final_state"] = tuple(tuple(agent.pos) for agent in self.agents)`

### `src/envs/basic_gymnasium_wrappers.py`
- `HLMDPEnvWrapper.reset()` forwards `navigation_init_state_dist` through to
  `ll_options`. One-line addition. Remaining `get_errors` output is all
  pre-existing (lbforaging import resolution, Env/Wrapper signature
  mismatches, render/pad/vstack typing) — unrelated to this change.

### `src/runners/episode_runner.py`
- Added `out["final_info"] = env_info` under `if test_mode:` before `return out`.
- **Please re-verify** the earlier `return_log_stats` / test_returns length-check
  bug fix (`elif len(self.test_returns) >= ...` vs `==`) is still in place — it
  was seen reverted once mid-session for unclear reasons and has not been
  re-confirmed since.
- Remaining `get_errors` output is pre-existing `MultiAgentEnv` attribute-access
  noise, unrelated to the new line.

### `src/runners/shared_memory_parallel_runner.py` — now clean except pre-existing issue
- `run()` test-mode return now:
  ```python
  if test_mode:
      result: dict[str, Any] = {"batch": self.batch, "final_infos": final_infos}
      if return_log_stats:
          result["log_stats"] = log_stats
      return result
  return self.batch
  ```
- Just fixed (this session) the type-annotation error on `result`. Only
  remaining `get_errors` finding is pre-existing/unrelated
  (`env.action_space` on `MultiAgentEnv`, line ~460).

### `src/simulation/evaluate.py` — clean
- New `collect_successful_final_state_dist(runner, n_eval_eps, reset_options=None) -> tuple[dict, float]`:
  runs eval episodes (snapshots/restores `runner.test_returns`/`test_stats` so
  it doesn't pollute normal eval logging), collects successful episodes'
  `final_state` from terminal infos, computes unique states + counts via
  `np.unique(..., return_counts=True, axis=0)`, returns
  `({"states": [...], "probs": [...]}, success_rate)`.

### `src/simulation/run.py` — clean (new code checked against error list)
- `run()` dispatch: if config has `hl_task_sequence`, calls
  `self.train_dependent_subtasks(hl_task_sequence)` and returns early.
- `train_single_task(self, reset_options=None, close_env: bool = True)`:
  now conditionally closes the env (needed so dependent-subtask loop can keep
  reusing config/build machinery without tearing down prematurely — check this
  still behaves correctly when called in isolation, i.e. `close_env` defaults
  to `True` for backward compatibility).
- New `train_dependent_subtasks(self, hl_task_sequence: list) -> None`:
  - Topologically walks edges in `hl_task_sequence` (e.g. `[[0,1],[0,2],[1,3],[2,3],[3,4]]`).
  - For each edge, rebuilds `self.args` / `self.runner` / `self.buffer` /
    `self.learner` from scratch via `build_sim()` (i.e. **every edge gets a
    fully independent freshly-initialized policy**, not a continuously
    fine-tuned one — this was a deliberate simplification, flag to user if
    they expected continual/warm-started training).
  - Reset options: root edges (no predecessor) use `hl_task: [from, to]` as
    today; dependent edges use `navigation_task_transition` +
    `navigation_init_state_dist` derived from predecessor(s)' learned dist.
  - After training each edge, calls `collect_successful_final_state_dist` to
    get that edge's terminal-state distribution for use by successor edges.
  - For nodes with multiple predecessors (e.g. node 3 ← {1, 2} in exp_36),
    merges predecessor distributions via `_merge_state_dists`.
- New static `_merge_state_dists(dists, weights) -> dict`: weighted
  union/merge of state distributions — **not** based on `ILPModel`/Gurobi.
- Fixed shared-memory compatibility by deriving the unique DAG root from the
  edge list instead of accessing `runner.env`, which does not exist on the
  process-backed runner.
- Fixed runner lifecycle: the initial runner is closed before the first
  per-edge rebuild, and `train_single_task(close_env=False)` now leaves the
  trained runner alive for terminal-state collection.
- `train_dependent_subtasks` now performs bounded CM iterations, controlled by
  `cm_max_iterations` and `cm_convergence_epsilon`. Each iteration derives a
  high-level edge policy from measured downstream success, evaluates analytical
  end-to-end success, and uses the previous iteration's policy to weight the
  next iteration's merge distributions.
- Each CM iteration now attempts the existing `ILPModel`/Gurobi occupancy
  optimization using the measured edge success rates, converts solved edge
  occupancies into normalized state policies, and logs whether ILP or the
  analytical fallback was used. The fallback remains available for missing
  Gurobi installations, malformed models, and infeasible success targets.
- The current dependent-subtask evaluation measures one success rate per edge,
  so that rate is assigned to every configured communication budget. The ILP
  can still choose the least-cost budget, but budget-specific low-level success
  measurements are not yet available in this loop.

### `experiments/exp_44/exp_config.yaml` — new example
- Mirrors exp_36's branching navigation environment.
- Uses one dependent sequence:
  `[[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]]`.
- Includes `n_eval_eps_dependent_subtask: 50`.
- Includes `cm_max_iterations: 5` and `cm_convergence_epsilon: 0.2`.

## Design decisions made but NOT yet confirmed with the user
Surface these explicitly next time you talk to the user:

1. **Budget-specific measurements.** The dependent loop currently has one
  measured success rate for each edge and applies it to every communication
  budget. To make the high-level budget policy fully empirical, evaluate each
  edge separately for each configured budget and populate the ILP transition
  table with those rates.
2. **Fresh policy per edge.** Each subtask edge trains an independent policy
   from scratch rather than continuing/fine-tuning a shared one. Simpler and
   matches "trains predecessor to convergence, then uses its terminal states"
   semantics, but uses more compute than warm-starting.
3. **wandb step-curve overlap.** Each edge's `train_single_task` call restarts
   `t_env` at 0, so if all edges log to wandb under one run, their step curves
   will overlap/reset repeatedly on the x-axis. Accepted as a cosmetic
   limitation, not fixed. Could be addressed later with a per-edge step offset
   or separate wandb runs per edge.

## ⚠️ Unrelated pending changes already in the working tree
`git status --short` shows several modified/untracked files that are **not**
part of this dependent-subtask work and were already present/in-progress
before/alongside this session. Do not confuse them with the feature above, and
do not revert or "clean up" them without checking with the user first:

- `Makefile`, `pyproject.toml`, `src/config/default.yaml`,
  `src/experiments/grid_search_experiment.py`, `src/utils/logging.py`,
  `src/experiments/ray_runner.py` (new, untracked) — an in-progress **Ray-based
  distributed experiment launcher** (`make run_experiment_ray`, `--computer ray`
  option in `grid_search_experiment.py`, W&B local-dir override, node-local
  W&B sync via `ray_runner.py`). Entirely separate feature from dependent
  subtasks. `src/utils/logging.py` also has an unrelated-looking bugfix in it
  (`group_name=args.wandb_group` instead of `args.run_name`, and
  `dir=config.get("wandb_local_dir", RESULTS_DIR)`).
- `agent_dependent.py` (untracked, repo root) — this is the **reference file**
  from `LIRA-illinois/marl` that the user pasted in earlier in the session as
  the design template for `train_dependent_subtasks`. Keep it around for
  reference but it is not meant to be committed as-is; consider moving it
  under `scratch/` or deleting once the feature is verified working.
- `experiments/exp_43/` (untracked) — a normal grid-search experiment config,
  unrelated to the dependent-subtask DAG feature.
- `submodules/gym-multigrid` shows as modified (dirty submodule) — this is
  expected since `team_navigation.py` was edited in place inside that
  submodule for this feature; will need a submodule commit/push eventually
  once changes are verified.

Before committing anything, separate these into distinct commits/branches:
(1) dependent-subtask feature, (2) Ray launcher work, (3) exp_43 config,
(4) reference file cleanup.

## Remaining TODOs (in priority order)

1. **Smoke test.** `py_compile` and pure CM-helper tests pass; still do a
  short, fast multi-iteration dry run of `train_dependent_subtasks` (e.g. tiny
   `n_eval_eps`, tiny `t_max`) against a minimal config to catch runtime
   issues: dict-shape mismatches between `collect_successful_final_state_dist`'s
   output and what `reset()`/`_merge_state_dists` expect, missing attribute
   access on the HLMDP env, reset-options schema mismatches
   (`navigation_init_state_dist` dict shape vs. what `PositionDist` parsing
   expects in `team_navigation.py`).
2. **Re-verify `episode_runner.py`'s eval-cadence fix is still in place** —
   it was seen reverted once before for no clear reason; check
   `return_log_stats` / the `test_returns` length comparison logic again.
3. **Produce an example dependent config.** Create e.g.
   `experiments/exp_44/exp_config.yaml` mirroring exp_36's env args
   (`env_args.map_name: 3ga_1r_small_hall`,
   `navigation_config: navigation_branching_small_hall.yaml`) but replacing
   the independent `hl_task` grid search with:
   ```yaml
   hl_task_sequence: [[0,1],[0,2],[1,3],[2,3],[3,4]]
   ```
4. **Communicate the 3 design tradeoffs above to the user** and get
   confirmation/adjust as needed.
5. Consider whether `_merge_state_dists`'s weighting scheme (how it weights
   multiple predecessors, e.g. nodes 1 and 2 both feeding into node 3) matches
   what the user actually wants — this hasn't been discussed in detail, only
   implemented as a reasonable default (likely proportional to each
   predecessor's success rate / episode count — verify against the actual
   implementation in `run.py` before assuming).
6. Run a short runtime smoke test in an environment where the full simulation
  import and Gurobi startup fit the execution budget, then validate the ILP
  policy source and selected communication budgets on the target DAG.

## How to resume
- Read [src/simulation/run.py](src/simulation/run.py) for
  `train_dependent_subtasks` / `_merge_state_dists` first — that's the core
  orchestration logic and the place most likely to need debugging once a
  smoke test is run.
- Cross-reference [experiments/exp_36/exp_config.yaml](experiments/exp_36/exp_config.yaml)
  for the exact env/navigation config shape to reuse for the new example config.
- Repo memory at `/memories/repo/epymarl-integration-points.md` was empty at
  handoff time — consider populating it with confirmed facts (build/test
  commands, config schema conventions) as they're verified, to help future
  sessions.
