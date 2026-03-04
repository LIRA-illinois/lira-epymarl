import sys
import os
import subprocess
import random
import time
import uuid
import json


def cli_to_sacred_args(argv):
    """Split sweep CLI args into Sacred options (--config, --env-config)
    and Sacred updates (everything else, passed after 'with')."""
    options = []
    updates = []

    for arg in argv:
        if not arg.startswith("--"):
            continue
        key_value = arg[2:]
        key = key_value.split("=", 1)[0]
        if key in ["config", "env-config"]:
            options.append(arg)
        else:
            updates.append(key_value)

            # Backward-compatible special case used in sweeps
            if key == "fcn_hidden":
                value = key_value.split("=", 1)[1] if "=" in key_value else ""
                updates.append(f"n_embed={value}")

    return options, updates


def updates_to_config_dict(updates):
    """Convert Sacred-style update strings ('key=value') into a flat dict
    for passing as wandb config so sweep charts can display hyperparameters."""
    config = {}
    for u in updates:
        if "=" in u:
            k, v = u.split("=", 1)
            # Try to parse as Python literal (number, bool, etc.)
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # Check common Python literals that aren't valid JSON
                if v == "True":
                    v = True
                elif v == "False":
                    v = False
        else:
            k, v = u, True
        config[k] = v
    return config


def run_seed(seed, script_path, options, updates, clean_env):
    """Run a single training seed as a subprocess with sweep env vars stripped.
    Returns (run_id, return_code)."""
    run_id = uuid.uuid4().hex[:8]
    env = dict(clean_env)
    env["WANDB_RUN_ID"] = run_id  # so we can query this run later

    cmd = [
        sys.executable, script_path,
        *options,
        "with",
        *updates,
        f"seed={seed}",
    ]

    print(f"[sweep_wrapper] Starting seed={seed}, wandb_run_id={run_id}")
    result = subprocess.run(cmd, env=env)
    print(f"[sweep_wrapper] Seed={seed} finished with exit code {result.returncode}")
    return run_id, result.returncode


def fetch_run_metric(api, entity, project, run_id, metric_name, max_retries=6, delay=10):
    """Fetch a metric from a finished run's summary via wandb API, with retries
    to handle propagation delay."""
    for attempt in range(max_retries):
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            if metric_name in run.summary:
                value = run.summary[metric_name]
                print(f"[sweep_wrapper]   run {run_id}: {metric_name}={value}")
                return value
            else:
                print(f"[sweep_wrapper]   run {run_id}: metric '{metric_name}' not in summary yet (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"[sweep_wrapper]   run {run_id}: API error (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(delay)

    print(f"[sweep_wrapper]   WARNING: could not fetch '{metric_name}' for run {run_id} after {max_retries} attempts")
    return None


def get_sweep_metric_name(api, entity, project, sweep_id):
    """Read the metric name from the sweep's config via wandb API."""
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        return sweep.config.get("metric", {}).get("name", None)
    except Exception as e:
        print(f"[sweep_wrapper] WARNING: could not read sweep metric config: {e}")
        return None


if __name__ == "__main__":
    options, updates = cli_to_sacred_args(sys.argv[1:])
    script_path = "src/main.py"

    num_seeds = int(os.environ.get("SWEEP_NUM_SEEDS", "3"))
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_seeds)]

    # ── Build a clean env: strip sweep-specific vars so each subprocess
    #    creates an independent wandb run (not the sweep-tracked one) ──
    sweep_env_keys = [
        "WANDB_SWEEP_ID",
        "WANDB_SWEEP_PARAM_PATH",
        "WANDB_AGENT_ID",
        "WANDB_RUN_ID",
    ]
    original_env = dict(os.environ)
    clean_env = {k: v for k, v in original_env.items() if k not in sweep_env_keys}

    # ── Generate a shared human-readable run name for all seeds ──
    hparam_tag = uuid.uuid4().hex[:6]
    run_name = f"sweep_{hparam_tag}"
    clean_env["WANDB_NAME"] = run_name  # wandb.init() reads this automatically

    # ── Run all seeds as subprocesses ──
    print(f"[sweep_wrapper] Running {num_seeds} seeds: {seeds} (name={run_name})")
    run_ids = []
    for seed in seeds:
        run_id, rc = run_seed(seed, script_path, options, updates, clean_env)
        if rc != 0:
            print(f"[sweep_wrapper] ABORTING: seed {seed} failed with exit code {rc}")
            sys.exit(rc)
        run_ids.append(run_id)

    # ── Compute averaged metric and report to the sweep ──
    import wandb

    entity = original_env.get("WANDB_ENTITY", "")
    project = original_env.get("WANDB_PROJECT", "")
    sweep_id = original_env.get("WANDB_SWEEP_ID", "")

    api = wandb.Api()

    # Determine the metric name from the sweep config
    metric_name = get_sweep_metric_name(api, entity, project, sweep_id)
    if not metric_name:
        # Fallback: use env var or a sensible default
        metric_name = os.environ.get("SWEEP_METRIC_NAME", "test_return_mean")
        print(f"[sweep_wrapper] Using fallback metric name: {metric_name}")
    else:
        print(f"[sweep_wrapper] Sweep metric: {metric_name}")

    # Fetch metric values from each seed run
    print(f"[sweep_wrapper] Fetching metrics from {len(run_ids)} seed runs...")
    values = []
    for run_id in run_ids:
        val = fetch_run_metric(api, entity, project, run_id, metric_name)
        if val is not None:
            values.append(float(val))

    if not values:
        print("[sweep_wrapper] ERROR: no metric values retrieved from any seed run")
        sys.exit(1)

    avg_value = sum(values) / len(values)
    print(f"[sweep_wrapper] Average {metric_name} across {len(values)} seeds: {avg_value}")

    # ── Create the sweep-tracked run and report the averaged metric ──
    # This wandb.init() picks up WANDB_RUN_ID and WANDB_SWEEP_ID from the
    # original (unmodified) parent process env, associating it with the sweep.
    hparam_config = updates_to_config_dict(updates)
    hparam_config["num_seeds"] = num_seeds
    hparam_config["seed_run_ids"] = run_ids
    hparam_config["seeds"] = seeds

    wandb.init(
        entity=entity,
        project=project,
        config=hparam_config,
        name=f"{run_name}-sweep",
        tags=["sweep_summary"],
    )

    # Log averaged metric + per-seed breakdown
    log_data = {metric_name: avg_value}
    for i, (seed, val) in enumerate(zip(seeds, values)):
        log_data[f"{metric_name}_seed{i}"] = val
        log_data[f"seed_{i}"] = seed
    wandb.log(log_data)

    wandb.finish()
    print(f"[sweep_wrapper] Done. Sweep summary run logged with {metric_name}={avg_value}")

