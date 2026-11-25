# Unified Hierarchical Policy Implementation TODO

This tracker is temporary coordination state. Independent and dependent projects use the same option-calibration and policy-synthesis pipeline; they differ only in initial-state support and successor-state propagation.

## Current status

- **Next task:** Verify lab/Ray/Slurm seed isolation and per-context independent/dependent behavior.
- **In progress:** Scheduler isolation tests and per-context fixture validation.
- **Blocked:** None.
- **Deferred:** Per-context policies and final cross-seed sensitivity post-processing until shared-context synthesis is stable.
- **Completed:** Added typed contracts, finite-data calibration/mixing primitives, project propagation, iterative candidate screening core, and resumable option-artifact manifests with focused tests.
- **Completed experiment:** Added `src/scripts/mock_hierarchical_experiment.py`, a 5-state merge fixture (`0 -> {1,2} -> 3 -> 4`) that writes/reloads mock calibration and option artifacts, then runs the iterative $p_g+\varepsilon_p$ process.
- **Parameterized experiment:** `n_agents`, `n_final_positions`, `n_comms_values`, and `n_subtasks` now drive mock joint-state, communication, edge, calibration, and artifact generation.
- **Validation:** New core and mock experiment tests pass; existing evaluation, dependent-CM, and high-level-policy targeted tests pass.
- **Grid integration:** Unified config/DAG/margin validation, budget preservation, run identity, and exp49 grid shape pass focused tests.
- **Mock pipeline:** exp49 completed one seed through the registered mock environment, runner calibration, success-vs-communication plots, iterative `$p_g` synthesis, candidate CSVs, offline W&B, and seed artifact output.
- **Utility extraction:** Moved reward totals, training stabilization, success-rate LCB, message-budget parsing, $p_g$ spec parsing, task-edge/config parsing, test-return loading, W&B project parsing, high-level env rendering, and state-distribution merging out of `Simulation`/`run.py` into utility modules.
- **Run split:** `run.py` is now a 17-line public compatibility entry; legacy implementation is isolated in `run_legacy.py`, and unified mock/real seed orchestration is in `unified_hierarchical_runner.py`.
- **Separate unified runner:** `UnifiedHierarchicalSeedRunner` now owns unified seed dispatch and real/mock orchestration, while `Simulation.run()` only validates and delegates unified configs.
- **Five-checkpoint mock figures:** exp49 and the standalone script now emit five mock evaluations at 20k, 40k, 60k, 80k, and 100k environment steps per subtask/capacity, with training success-vs-time and success-vs-comms plots.
- **Reliability-cost frontier:** Added ours-versus-fixed-comms PNG/PDF/CSV plotting. Exp49 now sweeps ten $p_g$ values and uploads the frontier to W&B.
- **Real snapshot/artifact pass:** Real option training now exposes 50/100-style evaluation snapshots, writes `real_training_eval_seed_*` CSV plus comms plots, produces seed-local fixed-comms frontier plots, and writes per-$p_g$ candidate JSON manifests.
- **Exp50 passing-candidate run:**  Added a 3-agent, 2-task MAIC tiny-hall experiment. A 1000-step seed produced 1.0 empirical success and 0.0 cost for all three `$p_g` values, with 20/20 exact held-out project success and candidate/terminal-state CSVs.
- **Real exp49 smoke:** Completed one seed of 10 real `(edge, capacity)` 100-step option trainings, 100-episode calibration, categorical calibration reload, and 10-$p_g$ synthesis. Real calibration is below $m_{\mathrm{est}}$ (`n=100`, `m_est=1476`) and yielded explicit infeasibility for tested thresholds rather than a finite-data certificate.
- **Real exp50 smoke:** Completed a 3-agent, 2-task, 1000-step MAIC seed in the very-small hallway. All four `(edge, capacity)` options trained; all three `$p_g$` values were certified with empirical success 1.0 and cost 0.0; 20 held-out episodes per `$p_g$` passed with exact terminal-state rows.
- **Iterative verification plots:** Added mock and real $p_g$-vs-iteration plots of empirical success and policy communication cost. Mock W&B: https://wandb.ai/jheglun2-phd-projects/comm_experiments/runs/b94w3wnq. Real W&B: https://wandb.ai/jheglun2-phd-projects/comm_experiments/runs/6i4apol4.
- **Replay/batch verification:** Exp49 uses `batch_size_run=10`, records 2 replays once per statewise calibration setting, and no longer records replays during held-out or snapshot-only outcome collection. This reduces exp49 from 1830 replay MP4 files to an expected 20 for 10 statewise settings plus normal evaluation replay calls. Previous batch-10 artifacts: https://wandb.ai/jheglun2-phd-projects/comm_experiments/runs/ljihzf29.
- **Greedy high-level baselines:** Added task-greedy and comms-greedy policy evaluators over the propagated calibrated DAG model. Task greedy maximizes edge success at maximum communication; comms greedy uses the same edge choice and the minimum capacity meeting the per-edge reliability threshold. Both curves are added to frontier CSV/PNG/PDF post-processing and mock experiment outputs.
- **Exp50/exp51 sequential-parallel validation:** Exp50 uses `batch_size_run=1`; exp51 uses `batch_size_run=10`. Both 1000-step real MAIC runs certified all three `$p_g$` values and held out 20/20 episodes. Exp51 generated 293 replay media files and uploaded its iteration/comparison artifacts to https://wandb.ai/jheglun2-phd-projects/comm_experiments/runs/wtfx765j.
- **Exp52 team-delay sweep:** Added a 5000-step, batch10, 3-agent smoke sweep for `team_delay_prob=[0.05,0.10,0.20]`. At 0.05, edge `0->1` success is 0.94 and `$p_g=0.55,0.65$` are certified. At 0.10, edge success is 0.935 and only `$p_g=0.55$` is certified. At 0.20, edge success is 0.83 and no tested `$p_g` is feasible. Results and plots are under `results/hierarchical_artifacts/exp_52_26-11-20_16-31-26/team_delay_comparison.{csv,png}`.
- **Exp53 mixed-distribution development proposal:** Two-task 3-agent 5000-step batch10 experiment on `navigation_mixing_two_task.yaml`, with communication budgets 0/1/2 and team delay probabilities 0.05/0.10. The edge 0->1 and 1->2 initial distributions each have two equal-probability joint states. It uses `epsilon_est=0.01`, `epsilon_p=0.05`, and `n_max=2`, with 50 held-out episodes and 10 synthesis iterations. Add exact mixed-start-distribution CSV/PNG/PDF output per `$p_g` and iteration, plus a 4000-sample uniform-support sampling test.

## 0. Coordination

- [x] Create and maintain this temporary work tracker.
- [x] Before Python commands, activate `.venv` with `source .venv/bin/activate`.
- [x] Preserve existing uncommitted and untracked work.
- [x] Keep independent/dependent behavior inside one implementation rather than separate algorithms.
- [ ] Add focused tests before expensive environment changes.
- [ ] Keep seed runs isolated until final post-processing.

## 0.1 Outer grid and execution model

The existing pipeline in `src/experiments/grid_search_experiment.py` remains the scheduler for coarse-grained work.

The outer Cartesian grid is:

`hyperparameter scenario x seed`

Each resulting `(experiment time ID, scenario, seed)` is an independent end-to-end run. It must own:

- low-level policy training or resume state,
- frozen low-level option library,
- statewise calibration rows,
- high-level policy or policies,
- all $p_g$ frontier candidates,
- held-out evaluation rows,
- manifests and summary outputs.

The outer grid must not mix scenarios or seeds. The Ray, lab, and Slurm runners may execute these units concurrently, but each unit must produce complete, independently resolvable artifacts.

### Outer grid parameters

The unified hierarchical experiment should keep scenario-varying parameters such as:

- map and environment
- number of agents
- number of communication values
- subtasks and project DAG
- low-level training hyperparameters
- policy seed set
- $p_g$ frontier list
- $\varepsilon_{\mathrm{est}}$, $\beta$, $\varepsilon_p$, $N_{\max}$
- number of calibration episodes
- max high-level iterations
- shared versus per-context policy scope

The grid search should assign a stable scenario ID before seed expansion. Seed expansion remains a cross-product of `(scenario, seed)`.

Unified hierarchical runs use the outer experiment `seed` for the complete low-level, calibration, synthesis, and held-out pipeline. Multiple seeds are separate grid runs, not nested `policy_seeds`.

## 0.2 Unified hierarchical config contract

Add one validated top-level config block, for example:

```yaml
hierarchical:
  policy_scope: shared_across_contexts
  task_sequence: [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]]
  pg_values: [0.60, 0.65, 0.70]
  epsilon_est: 0.05
  beta: 0.95
  epsilon_p: 0.20
  n_max: 4
  n_calibration_episodes: 100
  max_iterations: 20
  resume: latest
```

Configuration rules:

- `task_sequence` or its equivalent defines the project DAG.
- Communication capacities stay as one complete list during unified synthesis.
- `policy_scope` must be `shared_across_contexts` or `per_context`.
- `pg_values` is a frontier sweep input, not a cross-averaged result column.
- Per-seed list values expand inside the run or frontier stage without mixing seeds.
- `epsilon_p >= Delta_p = 1-(1-epsilon_est)^N_max` must be validated.
- `n_calibration_episodes >= m_est` must be validated against $K$, $L$, $\beta$, and $\varepsilon_{\mathrm{est}}`.
- Legacy `hl_task_sequence`, per-budget expansion, and `dependent_policy_seeds` must not silently control unified runs.

The preferred entry point is a single branch in `src/simulation/run.py` or a dedicated orchestration mixin invoked by `run.py`.

## 0.3 Required interfaces

Define stable interfaces between the existing framework and the new pipeline:

1. **Grid interface**
   - `_get_scenarios()` creates hyperparameter scenarios.
   - `_get_run_setups()` adds `(experiment, time_id, scenario, seed)` and builds one command per atomic run.
   - Unified hierarchical scenarios keep their full budget list.

2. **Entrypoint interface**
   - `src/main.experiment_main()` continues to construct `Simulation`.
   - `Simulation.run()` dispatches unified hierarchical work from the validated config.
   - A unified pipeline mode must not fall through to legacy dependent or independent branches.

3. **Low-level training interface**
   - Inputs: seed, subtask/edge, communication capacity, initial-state distribution, environment config.
   - Outputs: frozen policy artifact, checkpoint manifest entry, training timestep.
   - Resume modes: latest, specific timestep, no resume.

4. **Calibration interface**
   - Inputs: frozen library, setting $(i,c,s)$, episode count.
   - Outputs: one `CalibrationRow` containing categorical outcomes, success rate, terminal-state distribution, seed, setting key, and episode count.

5. **Synthesis interface**
   - Inputs: calibration rows, task DAG, root distribution, communication capacities, $p_g$, $\varepsilon_p$, $N_{\max}$, max iterations.
   - Outputs: seed-specific candidates, selected candidate, certificate status, propagated starts, empirical cost, empirical success.

6. **Frontier interface**
   - Inputs: one frozen library, calibration manifest, and $p_g$ sequence.
   - For each $p_g$, run synthesis independently and emit one status row.
   - Keep candidate and held-out outputs separate by $p_g$.

7. **Held-out evaluation interface**
   - Inputs: selected or all candidates, seed, context, edge, budget, start state, held-out episode count.
   - Outputs: exact project-success and terminal-state rows keyed as
     `(seed, context, candidate/p_g, edge, budget, start state, episode)`.

8. **Artifact interface**
   - Option manifest: low-level settings and checkpoints.
   - Calibration manifest: statewise rows.
   - Policy manifest: selected candidate(s) and $p_g$.
   - Held-out manifest/table: exact episode-level and summary results.

9. **Post-processing interface**
   - Final aggregation is read-only over completed seed artifacts.
   - The frontier remains per `$p_g` and per seed until cross-seed sensitivity reporting.

## 0.4 Parallelism levels

Use four distinct parallelism levels:

| Level | Unit | Mechanism |
|---|---|---|
| Outer hyperparameter grid | `(scenario, fixed seed)` | Existing grid search: lab runners, Ray, or Slurm |
| $p_g$ frontier | `(scenario, seed, $p_g$)` | Independent synthesis jobs inside or after a seed run |
| Low-level stage | `(edge, communication capacity, start-state support, seed)` | Independent option-training/calibration jobs |
| Held-out evaluation | `(seed, candidate, $p_g$, context, edge, budget, start state)` | Independent episode batches |

Do not perform seed-level parallelism by splitting one low-level run arbitrarily across different seed combinations. Each seed run should retain an immutable library and one matched synthesis/evaluation lineage.

For dependent finite-support DAGs:

- Parallelize settings within each DAG stage.
- Wait for one stage's frozen library and calibration rows.
- Propagate terminal-state distributions.
- Advance downstream stages.

For independent/singleton distributions:

- All settings with singleton support are independent.
- They use the same stage and artifact interface as finite-support cases.

## 0.5 $p_g$ frontier execution

The frontier is a required implementation, not only a late optional plot.

For every outer `(scenario, seed)`:

1. Freeze and calibrate the option library once.
2. Load or compute the finite-data margin $\varepsilon_p$.
3. Run iterative synthesis once for each configured $p_g$.
4. At each iteration:
   - mix statewise estimates,
   - solve with empirical threshold $p_g+\varepsilon_p$,
   - propagate the candidate,
   - retain candidates with empirical success at least $p_g+\varepsilon_p$.
5. Keep the lowest empirical-cost passing candidate for that `$p_g` and seed.
6. Run the held-out project check for that candidate.
7. Record certificate status, held-out success, held-out cost, and search failure separately.

Rules:

- `$p_g` candidates never share calibration data across seeds.
- Reuse the one frozen calibration manifest within all `$p_g` solves for a seed.
- Do not replace “no certified candidate found” with infeasibility.
- Do not aggregate frontier points across seeds until post-processing.
- A missing or failed `$p_g` solve must be represented as search status, not as held-out cost zero.

## 1. Schemas and contracts

- [x] Define `OptionSettingKey` as `(subtask edge, communication capacity, physical start state)`.
- [x] Define a categorical calibration outcome: `failure` or `(success, terminal state)`.
- [x] Define `CalibrationRow` with setting key, episode outcomes, success rate, terminal-state distribution, and seed.
- [x] Define `LowLevelPolicyArtifact` with seed, edge, capacity, initial distribution, config hash, timestep, and checkpoint path.
- [x] Define `HighLevelCandidate` with seed, $p_g$, cost, task policy, communication policy, induced distributions, and certificate status.
- [x] Define `HeldOutResult` with seed, context, project success, communication cost, and terminal outcome.
- [x] Define `CertificateStatus`: `certified`, `no-certified-candidate`, or `search-failed`.
- [x] Define one exact context key and one exact evaluation-row key.

## 2. Finite-state calibration

- [x] Implement $m_{\mathrm{est}}$ calculation.
- [x] Validate $\varepsilon_{\mathrm{est}}$, $\beta$, $K$, $L$, $N_{\max}$, and $\varepsilon_p$.
- [x] Enumerate $\mathcal{L}=\{(i,c,s)\}$.
- [x] Reuse the runner to collect independent categorical episodes for one statewise setting.
- [x] Store empirical categorical outcome distributions.
- [x] Test total variation against small known distributions.
- [x] Test the mixing lemma for arbitrary initial-state distributions.
- [x] Report calibration error, episode count, and confidence settings separately.

## 3. Outcome mixing and propagation

- [x] Mix statewise success rates using $b_{0,\operatorname{src}(i)}(s)$.
- [x] Mix success-conditioned terminal-state distributions.
- [x] Implement mixing for singleton initial-state supports.
- [x] Implement mixing for finite-support dependent initial-state distributions.
- [x] Propagate predecessor terminal states into downstream initial-state distributions.
- [x] Preserve failure as an absorbing project-failure outcome in project-flow calculations.
- [x] Treat $p_{ic}=0$ with an explicitly empty terminal-state distribution.
- [x] Add small chain, branching, and merging-DAG fixtures.

## 4. Low-level checkpoint pipeline

- [x] Replace independent `(edge, capacity, start state)` low-level training with one shared conditional policy.
- [x] Initialize the shared policy once per seed.
- [x] Randomly sample task edge and communication capacity for each batch rollout.
- [x] Sample initial joint state from the selected task edge's pre-specified initial-state distribution.
- [x] Add one batch of mixed-context trajectories to the replay buffer.
- [x] Include `(subtask, communication capacity, initial state)` as policy context.
- [x] Freeze the shared policy once after training.
- [x] Calibrate every statewise setting using the frozen shared policy.
- [x] Compare exp50 batch1 and exp51 batch10 shared-policy training results.

- [x] Implement a shared conditional option policy $\pi(s,i,c;\theta)$ in the new run pipeline.
- [x] Train and freeze one option library per seed.
- [x] Add an immutable checkpoint path and manifest entry for each policy/library setting.
- [x] Generate a policy manifest with seed, edge, capacity, distribution, config hash, timestep, and path.
- [x] Support loading the latest checkpoint.
- [x] Support loading a specific checkpoint/timestep.
- [x] Support resume-from-checkpoint lookup through the manifest.
- [x] Support calibration mode with no retraining.
- [x] Support evaluation mode with no retraining.
- [x] Support atomic manifest writes for seed jobs.
- [x] Decide and document checkpoint retention policy: preserve latest and specific timestep manifests; never overwrite another seed/artifact key.

## 5. Shared-context synthesis

- [x] Replace the legacy scalar edge-success-rate interface for the first environment implementation slice.
- [x] Synthesize with fixed mixed initial-state distributions.
- [x] Solve with threshold $p_g+\varepsilon_p$ through the repository ILPModel.
- [x] Propagate every candidate using the same frozen calibration data.
- [x] Evaluate each candidate under its own induced distributions.
- [x] Retain passing candidates and choose the lowest empirical cost.
- [x] Break the iteration cleanly on infeasible or non-progressing candidates.
- [x] Return “no certified candidate found” when no candidate passes.
- [x] Preserve the no-convergence and no-global-optimality caveats in synthesis results and candidate manifests.

## 6. Held-out project checks

- [x] Summarize project success separately from calibration success.
- [x] Summarize project communication cost separately from model cost.
- [x] Record held-out terminal-state and failure outcomes.
- [x] Distinguish held-out empirical performance from the finite-data certificate.
- [x] Run the held-out check for each selected `$p_g` candidate.
- [x] Store selected candidate policy artifacts separately by `$p_g`.

## 6.1 Low-level success-vs-communication plots

After a low-level policy finishes training and is evaluated, produce success vs communication plots for every `(seed, subtask)` combination.

- [x] Reuse `plot_context_evaluation()` and `plot_success_vs_comms_by_subtask()` through the unified runner.
- [x] Generate one curve per low-level policy seed and subtask edge.
- [x] Plot `test_task_completed_mean` on the y-axis against `msg_budget_per_agent` on the x-axis.
- [x] Include episode count and finite-data/calibration context in real/mock snapshot rows.
- [x] Preserve exact `(seed, subtask edge, communication capacity, start-state distribution)` row keys.
- [x] Keep seed and subtask panels separate; do not pool seed or subtask success curves.
- [x] Save PNG/PDF and data CSV artifacts in the seed-isolated output root.
- [x] Log iteration plots to W&B and preserve replay flags for exp49/exp50 eval episodes.
- [x] Generate training-time snapshots at every low-level evaluation interval.
- [x] Generate a final frozen-policy success-vs-communication plot after training stops.
- [x] Include all configured communication values even if an individual row has no successful episodes.
- [x] Mark missing/invalid evaluation rows as NaN/missing rather than silently interpolating or using a default success rate.

## 7. Independent/singleton implementation slice

- [x] Create a small two-node project fixture.
- [x] Create a small branching project fixture.
- [x] Create a 5-state merge fixture: `0 -> {1,2} -> 3 -> 4`.
- [x] Verify singleton initial-state support through the shared calibration and synthesis pipeline.
- [ ] Verify frozen-library reuse without retraining.
- [x] Verify $p_g+\varepsilon_p$ screening.
- [ ] Verify held-out success/cost reporting.
- [ ] Run focused unit tests before environment runs.

## 8. Dependent/finite-support implementation

- [x] Extend the singleton implementation to finite-support distributions.
- [x] Propagate starts from predecessor success terminal states.
- [x] Implement merging at project states with incoming flows.
- [x] Test chain and branch/merge DAG structures.
- [x] Confirm independent and dependent cases use the same core mixing and propagation APIs.

## 9. Evaluation redesign

- [x] Separate cross-budget and cross-seed result aggregation during raw evaluation.
- [x] Key option-calibration rows by `(subtask, capacity, start state, seed)`.
- [x] Key project rows by `(context, high-level policy/seed, edge, capacity, episode)`.
- [x] Preserve raw goal/failure outcomes and terminal states.
- [x] Track episode counts exactly.
- [x] Update logs and tables to use exact keys.
- [x] Add context-specific plots and avoid implicit seed/subtask averaging.
- [x] Add tests for missing, duplicate, or mixed evaluation keys.

## 10. Policy scope modes

- [x] Implement `shared_across_contexts` policy scope.
- [x] Implement `per_context` policy scope in config and real runner loop.
- [ ] Verify per-context mode for independent and dependent projects.
- [x] Keep shared mode first.
- [x] Synthesize one high-level policy per seed in shared mode.
- [x] Synthesize one high-level policy per `(seed, context)` in per-context mode.
- [x] Never average contexts or seeds inside a policy solve.

## 11. Parallelization

- [x] Treat `(scenario, seed)` as the atomic existing grid-search run.
- [x] Keep communication-budget expansion disabled for unified hierarchical scenarios.
- [x] Add scenario and seed identifiers to every grid-search run setup and output path.
- [ ] Parallelize low-level work over `(edge, capacity, start state, seed)` only within the atomic run.
- [ ] Parallelize stages of dependent DAG training stage-by-stage.
- [ ] Parallelize `$p_g$` synthesis over separate seed-isolated frontier jobs.
- [ ] Parallelize held-out evaluation over `(seed, policy, $p_g$, context, edge, capacity)`.
- [ ] Use disjoint manifests and output paths for every scenario and seed.
- [ ] Verify resumed/rerun jobs do not overwrite existing immutable seed artifacts.
- [ ] Confirm lab, Ray, and Slurm runners can submit the atomic units without relying on legacy tmux grouping semantics.
- [ ] Add tests for scenario expansion, seed uniqueness, budget preservation, runner command generation, and Ray/Slurm output paths.

## 12. Frontier sweep

- [x] Define a validated ordered `pg_values` or `postprocess_success_rate_specs` sequence in the unified config.
- [x] Add a naïve independent solve for each $p_g$.
- [x] Plot success vs normalized communication cost for ours and fixed-comms baseline.
- [x] Save PNG, PDF, and CSV for each reliability-cost frontier.
- [x] Add exp49 with ten `$p_g` values.
- [x] Reuse one frozen option library and one calibration manifest per seed across every $p_g$.
- [x] Record certificate, held-out success, held-out cost, and search status per `(seed, $p_g)` row.
- [x] Retain one lowest-cost empirical passing candidate per `(seed, $p_g)`.
- [x] Distinguish infeasible synthesis from no-certified-candidate and solver/search failure.
- [x] Add per-$p_g$ candidate policy CSV artifacts and candidate JSON manifests.
- [x] Add per-$p_g$ held-out terminal-state/failure data.
- [x] Add a reliability-cost frontier table and plot after each seed run.
- [x] Add final cross-seed frontier sensitivity aggregation only in post-processing.
- [ ] Test lower, median, high, and unattainable $p_g$ values.

## 13. Final seed post-processing

- [x] Aggregate seed frontier artifacts only after their seed tables are complete; duplicate-seed tables are rejected.
- [x] Report sensitivity of held-out success across independent low-level/high-level seed runs.
- [x] Report sensitivity of certified success and communication cost.
- [x] Report median, min/max, and seed-count for each frontier $p_g$ point.
- [x] Preserve per-seed curves and values in raw post-processing output.
- [x] Do not mix per-seed policies or per-seed calibration data during synthesis.
- [x] Clearly mark missing, infeasible, no-certificate, and search-failed seed/$p_g$ points.
- [x] Add post-processing tests that read multiple seed artifact manifests and cannot silently pool them.

## 14. Validation

- [x] Run targeted unit tests from the repository virtual environment.
- [x] Run small fixtures before environment experiments.
- [x] Test outer grid scenario generation and full scenario count.
- [x] Test seed/scenario Cartesian expansion.
- [x] Test unified budget preservation and legacy budget splitting suppression.
- [x] Test one seed through the mock 5-state merge fixture.
- [x] Test one seed through a small real environment end-to-end for option training/calibration; add passing-candidate held-out execution next.
- [x] Test exp50 tiny-hall MAIC candidate synthesis and exact held-out project episodes end-to-end.
- [x] Validate independent/singleton project behavior.
- [x] Validate dependent/finite-support project behavior.
- [x] Run the `$p_g$` frontier on at least passing, boundary, and unattainable values.
- [x] Run targeted evaluation and checkpoint tests.
- [ ] Run multiple seeds to verify isolation and artifact naming.
- [ ] Run both lab and Ray command-generation tests.
- [ ] Run Slurm file-generation tests only if cluster integration changes.
- [ ] Run final seed post-processing tests with complete and incomplete seed sets.

## 15. Mock merge experiment

Run:

```sh
source .venv/bin/activate
python src/scripts/mock_hierarchical_experiment.py \
  --output-dir tmp/mock_hierarchical_synthesis \
  --seed 17 \
  --n-agents 3 \
  --n-final-positions 4 \
  --n-comms-values 2 \
  --n-subtasks 5 \
  --pg 0.65 \
  --epsilon-est 0.05 \
  --beta 0.95 \
  --epsilon-p 0.10 \
  --n-estimation-episodes 100 \
  --iterations 3
```

The fixture has states `0,1,2,3,4`, task edges `0->1,0->2,1->3,2->3,3->4`, and a final project goal at state `4`. It generates and reloads mock statewise calibration rows and fixed option-policy manifests before running the paper's iterative candidate/screening loop.

Parameter meanings:

- `--n-agents`: number of agents represented in each mock joint state.
- `--n-final-positions`: number of physical positions available to each mock agent.
- `--n-final-joint-states`: not a CLI parameter; it is derived as `n_agents*n_final_positions` for this representative mock model.
- `--n-comms-values`: number of communication capacities, evenly spaced from `0.0` to `1.0`.
- `--n-subtasks`: number of task edges. Five uses the branching/merge DAG; other counts use a chain.
- `l_setting_count`: derived as `n_comms_values*n_subtasks*n_final_joint_states`.

Custom chain example:

```sh
source .venv/bin/activate
python src/scripts/mock_hierarchical_experiment.py \
  --output-dir tmp/mock_hierarchical_synthesis_3c \
  --n-agents 3 \
  --n-final-positions 4 \
  --n-comms-values 3 \
  --n-subtasks 3 \
  --pg 0.55 \
  --epsilon-p 0.10 \
  --iterations 2
```

## 18. exp49 interface test experiment

Create an experiment based on exp48 to test the boundary between existing RL training/evaluation and the outer hierarchical synthesis loop.

### Design

- **Experiment name:** `exp_49`
- **Environment:** team navigation, `3ga_1r_small_hall`
- **Outer grid:** one fixed seed initially, with later seed expansion supported
- **Project DAG:** `0->1, 0->2, 1->3, 2->3, 3->4`
- **Agent count:** 3
- **Communication capacities:** `0, 1`
- **Policy scope:** shared across contexts
- **Low-level training horizon:** `t_max=100`
- **Evaluation cadence:** every 50 training steps
- **Test episodes:** 2 episodes per context/evaluation call
- **Saved models:** disabled for the first implementation pass
- **W&B:** online mock run for this short test
- **Frontier:** 10 values from `p_g=0.55` through `p_g=1.00`
- **High-level iterations:** 3
- **Held-out episodes:** 20
- **Calibration episodes:** 100 for interface testing only

This is an integration/smoke experiment, not a finite-data-theorem verification. It should exercise code paths with short rolls and report that certificate margins are not being claimed unless $m_{\mathrm{est}}$ requirements are met.

### Grid and config TODOs

- [x] Keep exp49 as one scenario × seed run and preserve the full `[0,1]` communication-budget list.
- [x] Verify `_get_scenarios()` does not expand exp49 into separate budget scenarios.
- [x] Verify `_get_run_setups()` generates exactly one command for seed 101.
- [x] Verify exp49's nested `hierarchical` dict reaches `Simulation.run()` unchanged except normalized lists/enums.
- [x] Test fixed `seed=101` with no separate `n_seeds` scenario parameter.

### RL interface TODOs

- [x] Add a unified-run dispatch entry after post-processing and before legacy `hl_task_sequence`.
- [x] Open one seed-isolated artifact root for exp49.
- [x] Train or load real low-level policy candidates for each `(edge, capacity, seed)`; exp49 completed 10 real 100-step settings.
- [x] Evaluate real low-level policies at training steps 50 and 100 for the configured contexts; exp49 smoke ran each setting through 100 episodes after training.
- [x] Exercise mock edge/budget episodes through `EpisodeRunner.run(test_mode=True)`.
- [x] Record success and successful terminal-state distributions separately.
- [x] Convert unified option settings into team-navigation joint-position reset distributions.
- [x] Plot success-vs-communication data for real low-level policies using seed-local snapshot callbacks and CSV/plot output.
- [x] Add exact held-out episode execution when a real candidate is selected; exp50 verified 20/20 success per $p_g$.
- [x] Save real checkpoint/frozen success-vs-comms rows using `plot_context_evaluation` and `plot_success_vs_comms_by_subtask`.
- [x] Generate mock success-vs-communication curves through `plot_context_evaluation`.
- [x] Preserve the 100-step training and 50-step evaluation cadence in exp49 option-trainer logs.
- [x] Snapshot and freeze low-level weights per `(edge, capacity, seed)` option setting.

### Calibration and synthesis TODOs

- [x] Enumerate every singleton `(edge, capacity, start state)` setting for the exp49 DAG.
- [x] Compute mock `$m_{\mathrm{est}}` and report the 100-episode interface run separately from certificate applicability.
- [x] Reuse the existing statewise calibration collector for the exp49 mock settings.
- [x] Save/load one calibration manifest at the exp49 artifact root.
- [x] Mix calibration rows under induced initial-state distributions in the synthesis core.
- [x] Run the existing iterative synthesizer for `p_g=0.60` and `p_g=0.65` in mock mode.
- [x] Propagate starts through the 5-state merge and run final candidate screening.
- [x] Return `certified`, `no_certified_candidate`, `infeasible`, or `search_failed` status.

### Artifact TODOs

- [x] Write a mock run summary containing artifact root, seed, `$p_g$ values`, and status.
- [x] Write separate mock and real low-level/calibration artifacts; per-$p_g$ candidate/held-out artifacts remain for the next passing-candidate slice.
- [x] Keep each exp49 rerun under a deterministic scenario/seed artifact key.
- [x] Keep seed mixing isolated in mock and real seed runner dispatch.
- [x] Print concise mock and real JSON summaries for local debugging.

## 19. Mock low-level environment runner

Before real low-level training, add a normal environment-keyed mock runner that exercises the full grid/run/evaluation/synthesis interface.

- [x] Register a `mock_hierarchical` environment in `src/envs/__init__.py`.
- [x] Implement runner-compatible `reset`, `step`, `state`, `obs`, `avail_actions`, `action_space`, episode limit, and environment info.
- [x] Accept `hl_task`, communication capacity, seed, and mock library parameters through `reset_options`/`env_args`.
- [x] Generate categorical option outcomes from deterministic mock success rates.
- [x] Record successful terminal states and failure outcomes on `final_info`.
- [x] Support evaluation episodes through `EpisodeRunner.run(test_mode=True)`.
- [x] Support parallel/shared-memory runner environment construction.
- [x] Add a grid config switch such as `env=mock-hierarchical-v0` or `env_args.mock_low_level=true`.
- [x] Make `run.py` dispatch unified mock mode before legacy low-level training.
- [x] Reuse `collect_statewise_calibration_row` instead of a synthesis-only mock sampler.
- [x] Save/load mock calibration manifests through the existing artifact helpers.
- [x] Save/load mock option manifests through the existing artifact helpers in every seed.
- [x] Run the existing comms success-vs-budget plot per `(seed, subtask)`.
- [x] Add at least five mock training-evaluation checkpoints to every run.
- [x] Plot realistic success vs environment time per `(seed, subtask)`.
- [x] Plot success vs communication capacity at each mock checkpoint.
- [x] Run one seed through all `$p_g$` values and write candidate frontier artifacts.
- [x] Run held-out result rows for selected mock candidates.
- [x] Log an offline W&B run from exp49 using the mock environment.
- [x] Keep reset/step, terminal-info, and calibration interfaces shared across mock and real runner dispatch.
- [ ] Add W&B table-shape and real runner integration tests.

### Acceptance criteria

- [x] Grid generation reports one scenario for exp49.
- [x] Grid run setup reports exactly one seed 101 run.
- [x] Unified mode never splits `[0,1]` into separate budget scenarios.
- [x] Real training runs to 100 steps and invokes evaluation for the exp49 option settings; latest seed completed all 10 `(edge, capacity)` trainings and 100-episode calibration.
- [x] Mock low-level and high-level interfaces complete or report an explicit search status.

## 20. Remaining run.py extraction

The current goal is to reduce `run.py` further by moving only stateless or coherent utility workloads, not changing legacy behavior.

- [x] Move training reward totals, stabilization, and success-rate LCB helpers to `simulation_utils.py`.
- [x] Move message-budget, $p_g$-spec, task-edge, W&B project, and test-return helpers to `simulation_utils.py`.
- [x] Move merge-state-distribution and high-level environment rendering to `context_utils.py`.
- [x] Move context evaluation and success-vs-comms plotting helpers to `context_utils.py`.
- [x] Move success-vs-cost and fixed-baseline plotting helpers to `context_utils.py`.
- [x] Move unified mock seed execution into `unified_hierarchical_runner.py`.
- [x] Move unified real seed execution into `unified_hierarchical_runner.py`.
- [x] Leave `_run_unified_hierarchical()` as a thin config/real-mock dispatch in `run.py`.
- [ ] Move legacy W&B table/artifact loading into a post-processing utility module.
- [ ] Move legacy task-frame/rendering helpers into a navigation plotting utility.
- [ ] Keep only lifecycle methods, runner/learner wiring, logging dispatch, and thin compatibility wrappers in `Simulation`.
- [ ] Add tests that assert extracted helpers and compatibility wrappers return identical results.
- [x] Mock saved files reload without retraining.
- [x] The mock test run can finish locally within the short environment limits.
- [ ] Existing exp48 and legacy dependent-subtask behavior remain unchanged.

## 16. Implementation order

1. **Grid/config contract**
   - Add and validate `hierarchical` config.
   - Add `_get_scenarios()` tests for no legacy budget splitting.
   - Add stable scenario/seed run IDs and artifact roots.

2. **One unified seed run**
   - Implement one `(scenario, seed)` entry point in `src/simulation/run.py`.
   - Train or resume low-level options.
   - Freeze the library.
   - Enumerate statewise settings.
   - Collect calibration.
   - Persist calibration and option manifests.

3. **One $p_g$ synthesis slice**
   - Use one frozen calibration manifest.
   - Mix statewise estimates.
   - Solve through the existing ILPModel adapter.
   - Propagate candidate-induced starts.
   - Screen and retain the lowest-cost passing candidate.

4. **One held-out seed slice**
   - Evaluate the selected candidate under its own induced distributions.
   - Write exact held-out project and terminal-state rows.
   - Report empirical success separately from finite-data certificate status.

5. **Full `$p_g$` frontier**
   - Repeat synthesis for the configured `$p_g` sequence.
   - Reuse calibration and library data.
   - Store each `$p_g` candidate and result independently.

6. **Outer grid integration**
   - Generate scenario/seed units.
   - Launch through lab, Ray, or Slurm.
   - Confirm resume and rerun behavior.

7. **Seed post-processing**
   - Wait for all seed artifacts for a scenario.
   - Produce sensitivity tables and frontier plots.

8. **Per-context scope**
   - Implement only after shared-context end-to-end work is stable.

## 17. Interface test matrix

| Interface | Required test |
|---|---|
| Grid scenario generation | No legacy budget splitting in unified mode |
| Grid run setup | Unique `(time_id, scenario, seed)` command and artifact key |
| Config validation | Invalid $p_g$, $\varepsilon_p$, episode count, or DAG rejected |
| Low-level artifact save/load | Latest and specific timestep both resolve |
| Calibration save/load | One exact row for every `(edge, capacity, start state)` |
| Synthesis | Passing, no-candidate, infeasible, and solver-failure statuses distinct |
| Iteration | Candidate-specific induced starts and lowest-cost retention |
| Held-out evaluation | Exact key, exact episode count, terminal-state rows |
| Frontier | Reuses library/calibration; independent `$p_g` candidate outputs |
| Ray/lab | Atomic seed run, GPU assignment, rerun isolation |
| Slurm | One or more atomic seed runs per job; no seed mixing |
| Final post-processing | Cross-seed sensitivity only after seed completion |
