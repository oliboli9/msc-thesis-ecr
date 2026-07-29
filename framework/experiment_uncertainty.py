import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Reuse the regex parsers from the delay experiment harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_delay import FILL_RE, LATENESS_RE  # noqa: E402

REPO_ROOT   = Path(__file__).resolve().parent.parent
RUNNER      = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR = REPO_ROOT / "results" / "uncertainty-experiments"

MODEL_FILE             = "models/max-demand-fulfilment.py"  # rejection model (default)
SCENARIO               = "moderate"                          # +10%, as in §5
NUM_ITERATIONS         = 5
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1

# (year, week) start instances — matched to the §5 rolling-horizon table.
INSTANCES = [(2023, 1), (2024, 1), (2025, 1), (2026, 1)]

# Per-week growth of the forecast-noise std. 0.0 = perfect foresight.
SIGMA_VALUES = [0.0, 0.1, 0.2, 0.4]
SEEDS        = [0]

SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock"]


def run_one(label: str, extra_env: dict, keep_snapshot: bool = False) -> dict:
    """Run one rolling-horizon execution and parse its summary metrics."""
    # Clean the shared per-run snapshot dirs that runner.py writes into.
    for sub in SNAPSHOT_DIRS:
        d = RESULTS_DIR.parent / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "RUNNER_MODEL_FILE":             MODEL_FILE,
        "RUNNER_SCENARIO":               SCENARIO,
        "RUNNER_NUM_ITERATIONS":         str(NUM_ITERATIONS),
        "RUNNER_PLANNING_HORIZON_WEEKS": str(PLANNING_HORIZON_WEEKS),
        "RUNNER_DECISION_HORIZON_WEEKS": str(DECISION_HORIZON_WEEKS),
        **extra_env,
    })

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, str(RUNNER)], cwd=str(REPO_ROOT),
                         env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.stdout:
        print(res.stdout[-2000:])
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"runner failed: {label} (exit {res.returncode})")

    if keep_snapshot:
        target = RESULTS_DIR / label
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for sub in SNAPSHOT_DIRS:
            src = RESULTS_DIR.parent / sub
            if src.exists():
                shutil.copytree(src, target / sub, dirs_exist_ok=True)

    fill_match     = None
    total_lateness = 0.0
    for line in res.stdout.splitlines():
        m = FILL_RE.search(line)
        if m:
            fill_match = m
        lm = LATENESS_RE.search(line)
        if lm:
            total_lateness += float(lm.group(1))

    if fill_match is None:
        print(f"WARNING: no fill-rate summary line found for {label}")
        fulfilled, demand, pct = 0, 0, 0.0
    else:
        fulfilled = int(fill_match.group(1))
        demand    = int(fill_match.group(2))
        pct       = float(fill_match.group(3))

    return {
        "label": label,
        "fill_rate_pct": pct,
        "fulfilled": fulfilled,
        "demand": demand,
        "total_lateness_cd": total_lateness,
        "wall_seconds": round(wall, 1),
    }


def run_sweep(instances, sigmas, seeds) -> pd.DataFrame:
    rows = []
    for (year, week) in instances:
        for sigma in sigmas:
            # sigma == 0 is deterministic: a single seed suffices.
            seed_list = [0] if sigma == 0.0 else seeds
            for seed in seed_list:
                label = f"{year}w{week}_s{sigma}_seed{seed}"
                extra = {
                    "RUNNER_START_YEAR":     str(year),
                    "RUNNER_START_WEEK":     str(week),
                    "RUNNER_FORECAST_SIGMA": str(sigma),
                    "RUNNER_FORECAST_SEED":  str(seed),
                }
                # Keep one snapshot per (instance, sigma) for later inspection.
                row = run_one(label, extra, keep_snapshot=(seed == seed_list[0]))
                row.update({"instance": f"{year}w{week}", "sigma": sigma, "seed": seed})
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RESULTS_DIR / "exp_uncertainty.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results -> {raw_path}")

    # Per-(instance, sigma) mean +/- std.
    by_inst = (df.groupby(["instance", "sigma"])["fill_rate_pct"]
                 .agg(["mean", "std", "count"]).reset_index())
    by_inst.to_csv(RESULTS_DIR / "exp_uncertainty_by_instance.csv", index=False)

    # Overall per-sigma mean +/- std (across instances and seeds).
    overall = (df.groupby("sigma")
                 .agg(fill_mean=("fill_rate_pct", "mean"),
                      fill_std=("fill_rate_pct", "std"),
                      lateness_mean=("total_lateness_cd", "mean"),
                      n=("fill_rate_pct", "count"))
                 .reset_index())
    overall.to_csv(RESULTS_DIR / "exp_uncertainty_by_sigma.csv", index=False)

    print("\nPer-sigma summary (fill rate %, mean +/- std):")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick check: 1 instance, 2 sigmas, 2 seeds.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        df = run_sweep([(2025, 1)], [0.0, 0.2], [0, 1])
    else:
        df = run_sweep(INSTANCES, SIGMA_VALUES, SEEDS)

    summarise(df)
