# Unified Hierarchical Policy Implementation TODO

This tracker is temporary coordination state. Independent and dependent projects use the same option-calibration and policy-synthesis pipeline; they differ only in initial-state support and successor-state propagation.

## Current status

- **Next task:** Define typed schemas and contracts.
- **In progress:** None yet; implementation has not started.
- **Blocked:** None.
- **Deferred:** Seed post-processing and reliability-cost frontier sweeps until the core pipeline is stable.

## 0. Coordination

- [x] Create and maintain this temporary work tracker.
- [ ] Before Python commands, activate `.venv` with `source .venv/bin/activate`.
- [ ] Preserve existing uncommitted and untracked work.
- [ ] Keep independent/dependent behavior inside one implementation rather than separate algorithms.
- [ ] Add tests before expensive environment changes.
- [ ] Keep seed runs isolated until final post-processing.

## 1. Schemas and contracts

- [ ] Define `OptionSettingKey` as `(subtask edge, communication capacity, physical start state)`.
- [ ] Define a categorical calibration outcome: `failure` or `(success, terminal state)`.
- [ ] Define `CalibrationRow` with setting key, episode outcomes, success rate, terminal-state distribution, and seed.
- [ ] Define `LowLevelPolicyArtifact` with seed, edge, capacity, initial distribution, config hash, timestep, and checkpoint path.
- [ ] Define `HighLevelCandidate` with seed, $p_g$, cost, task policy, communication policy, induced distributions, and certificate status.
- [ ] Define `HeldOutResult` with seed, context, project success, communication cost, episode count, and outcome rows.
- [ ] Define `CertificateStatus`: `certified`, `no-certified-candidate`, or `search-failed`.
- [ ] Define one exact context key and one exact evaluation-row key.

## 2. Finite-state calibration

- [ ] Implement $m_{\mathrm{est}}$ calculation.
- [ ] Validate $\varepsilon_{\mathrm{est}}$, $\beta$, $K$, $L$, $N_{\max}$, and $\varepsilon_p$.
- [ ] Enumerate $\mathcal{L}=\{(i,c,s)\}$.
- [ ] Run independent episodes for every statewise setting.
- [ ] Store empirical categorical outcome distributions.
- [ ] Test total variation against small known distributions.
- [ ] Test the mixing lemma for arbitrary initial-state distributions.
- [ ] Report calibration error, episode count, and confidence settings separately.

## 3. Outcome mixing and propagation

- [ ] Mix statewise success rates using $b_{0,\operatorname{src}(i)}(s)$.
- [ ] Mix success-conditioned terminal-state distributions.
- [ ] Implement mixing for singleton initial-state supports.
- [ ] Implement mixing for finite-support dependent initial-state distributions.
- [ ] Propagate predecessor terminal states into downstream initial-state distributions.
- [ ] Preserve failure as the absorbing project-failure outcome.
- [ ] Treat $p_{ic}=0$ with an explicitly arbitrary terminal-state distribution.
- [ ] Add small chain, branching, and merging-DAG fixtures.

## 4. Low-level checkpoint pipeline

- [ ] Implement a shared conditional option policy $\pi(s,i,c;\theta)$.
- [ ] Train and freeze one option library per seed.
- [ ] Add one immutable checkpoint per policy/library setting.
- [ ] Generate a policy manifest with seed, edge, capacity, distribution, config hash, timestep, and path.
- [ ] Support loading the latest checkpoint.
- [ ] Support loading a specific checkpoint/timestep.
- [ ] Support resume-from-checkpoint for training.
- [ ] Support calibration mode with no retraining.
- [ ] Support evaluation mode with no retraining.
- [ ] Support atomic archive paths for parallel seed jobs.
- [ ] Decide and document checkpoint retention policy.

## 5. Shared-context synthesis

- [ ] Replace the legacy scalar edge-success-rate interface for the first implementation slice.
- [ ] Synthesize with fixed mixed initial-state distributions.
- [ ] Solve with threshold $p_g+\varepsilon_p$.
- [ ] Propagate every candidate using the same frozen calibration data.
- [ ] Evaluate each candidate under its own induced distributions.
- [ ] Retain passing candidates and choose the lowest empirical cost.
- [ ] Break the iteration cleanly on infeasible or non-progressing candidates.
- [ ] Return “no certified candidate found” when no candidate passes.
- [ ] Preserve the no-convergence and no-global-optimality caveats.

## 6. Held-out project checks

- [ ] Evaluate project success separately from calibration success.
- [ ] Evaluate project communication cost separately from model cost.
- [ ] Record held-out terminal-state and failure outcomes.
- [ ] Distinguish held-out empirical performance from the finite-data certificate.
- [ ] Run the check after each high-level iteration.
- [ ] Store iteration index and selected candidate status.

## 7. Independent/singleton implementation slice

- [ ] Create a small two-node project fixture.
- [ ] Create a small branching project fixture.
- [ ] Verify singleton initial-state support through the shared pipeline.
- [ ] Verify frozen-library reuse without retraining.
- [ ] Verify $p_g+\varepsilon_p$ screening.
- [ ] Verify held-out success/cost reporting.
- [ ] Run focused unit tests before environment runs.

## 8. Dependent/finite-support implementation

- [ ] Extend the singleton implementation to finite-support distributions.
- [ ] Propagate starts from predecessor success terminal states.
- [ ] Implement merging at project states with incoming flows.
- [ ] Test chain and branch/merge DAG structures.
- [ ] Confirm independent and dependent cases use the same core APIs.

## 9. Evaluation redesign

- [ ] Replace cross-budget and cross-seed result aggregation during raw evaluation.
- [ ] Key option-calibration rows by `(subtask, capacity, start state, seed)`.
- [ ] Key project rows by `(context, high-level policy/seed, edge, capacity, episode)`.
- [ ] Preserve raw goal/failure outcomes and terminal states.
- [ ] Track episode counts exactly.
- [ ] Update logs and tables to use exact keys.
- [ ] Update context plots to avoid implicit averaging.
- [ ] Add tests for missing, duplicate, or mixed evaluation keys.

## 10. Policy scope modes

- [ ] Implement `shared_across_contexts` policy scope.
- [ ] Implement `per_context` policy scope.
- [ ] Support both scopes for independent and dependent projects.
- [ ] Keep shared mode first.
- [ ] Synthesize one high-level policy per seed in shared mode.
- [ ] Synthesize one high-level policy per `(seed, context)` in per-context mode.
- [ ] Never average contexts or seeds inside a policy solve.

## 11. Parallelization

- [ ] Parallelize low-level work over `(edge, capacity, start state, seed)`.
- [ ] Parallelize stages of dependent DAG training stage-by-stage.
- [ ] Parallelize high-level synthesis over `(seed, p_g)`.
- [ ] Parallelize held-out evaluation over `(seed, policy, context, edge, capacity)`.
- [ ] Use disjoint manifests and output paths for each seed.
- [ ] Verify resumed/rerun jobs do not overwrite existing seed artifacts.

## 12. Frontier sweep

- [ ] Add a naïve independent solve for each $p_g$.
- [ ] Reuse frozen option libraries and calibration rows.
- [ ] Record certificate, held-out success, cost, and search status per frontier point.
- [ ] Distinguish infeasible synthesis from no-certified-candidate.
- [ ] Add this only after the core pipeline is stable.

## 13. Final seed post-processing

- [ ] Aggregate only after training/synthesis/evaluation are complete.
- [ ] Report sensitivity of held-out success across low-level/high-level seed pairs.
- [ ] Report sensitivity of certified success and communication cost.
- [ ] Preserve per-seed curves and values in raw post-processing output.
- [ ] Do not mix per-seed policies or per-seed calibration data during synthesis.

## 14. Validation

- [ ] Run targeted unit tests from the repository virtual environment.
- [ ] Run small fixtures before environment experiments.
- [ ] Validate independent/singleton project behavior.
- [ ] Validate dependent/finite-support project behavior.
- [ ] Run targeted evaluation and checkpoint tests.
- [ ] Run one seed through a small environment end-to-end.
- [ ] Run multiple seeds to verify isolation and artifact naming.
