

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT      = Path(__file__).resolve().parent.parent
RUNNER         = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR    = REPO_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "model-comparison"

# ── Sweep configuration ──────────────────────────────────────────────────────

# 4 problem instances (Jan W1 of each available stock year).
INSTANCES: list[tuple[int, int]] = [
    (2023, 1),
    (2024, 1),
    (2025, 1),
    (2026, 1),
]

# (label, model_file_relpath)
MODELS: list[tuple[str, str]] = [
    ("rejection",  "models/max-demand-fulfilment.py"),
    ("delay",      "models/min-lateness-fulfilment.py"),
    ("sequential", "models/sequential-demand-then-reposition.py"),
]

# Demand scenario (as produced by pipeline/augment_demand.py --all).
SCENARIO = "moderate"  # +10% augmented demand

# Rolling-horizon shape — full study uses 8 iterations; set to 1 for a quick
# sanity check before committing to the full sweep.
NUM_ITERATIONS         = 8
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1

# Subdirs in results/ that runner.py writes to per iteration.
SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock", "vessel-utilisation"]


# ── Run one combination ──────────────────────────────────────────────────────

def run_one(year: int, week: int, model_short: str, model_file: str) -> Path:
    """Run a single (instance, model) combination and snapshot its results."""
    inst_label = f"{year}w{week}"
    target_dir = COMPARISON_DIR / f"{inst_label}_{model_short}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale per-iteration outputs so we snapshot only this run.
    for sub in SNAPSHOT_DIRS:
        d = RESULTS_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RUNNER_SCENARIO"]               = SCENARIO
    env["RUNNER_MODEL_FILE"]             = model_file
    env["RUNNER_NUM_ITERATIONS"]         = str(NUM_ITERATIONS)
    env["RUNNER_PLANNING_HORIZON_WEEKS"] = str(PLANNING_HORIZON_WEEKS)
    env["RUNNER_DECISION_HORIZON_WEEKS"] = str(DECISION_HORIZON_WEEKS)
    env["RUNNER_START_YEAR"]             = str(year)
    env["RUNNER_START_WEEK"]             = str(week)

    print(f"\n{'#'*70}")
    print(f"# Instance {inst_label}  |  model={model_short}  |  scenario={SCENARIO}")
    print(f"# {NUM_ITERATIONS} iter × {DECISION_HORIZON_WEEKS}w decision "
          f"× {PLANNING_HORIZON_WEEKS}w planning")
    print(f"{'#'*70}")

    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"runner.py failed for instance={inst_label}, model={model_short} "
            f"(exit {result.returncode})"
        )

    for sub in SNAPSHOT_DIRS:
        src = RESULTS_DIR / sub
        dst = target_dir / sub
        if src.exists():
            shutil.copytree(src, dst)

    return target_dir


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_summary() -> pd.DataFrame:
    """Walk the per-instance snapshots and build a tidy per-iteration table."""
    rows = []
    for year, week in INSTANCES:
        inst_label = f"{year}w{week}"
        for model_short, _ in MODELS:
            run_dir = COMPARISON_DIR / f"{inst_label}_{model_short}" / "demand-fulfilment"
            if not run_dir.exists():
                continue
            for csv in sorted(run_dir.glob("*.csv")):
                df = pd.read_csv(csv)
                demand    = int(df["Demand"].sum())
                fulfilled = int(df["Fulfilled"].sum())
                unfilled  = int((df["Demand"] - df["Fulfilled"]).sum())
                fill_rate = fulfilled / demand if demand else 0.0
                if "LatenessDays" in df.columns:
                    cd_late     = int(df["LatenessDays"].sum())
                    late_groups = int((df["LatenessDays"] > 0).sum())
                    mean_late   = (df.loc[df["LatenessDays"] > 0,
                                          "MeanLatenessDays"].mean()
                                   if late_groups > 0 else 0.0)
                else:
                    cd_late, late_groups, mean_late = 0, 0, 0.0
                rows.append({
                    "instance":              inst_label,
                    "start_year":            year,
                    "start_week":            week,
                    "model":                 model_short,
                    "iteration":             csv.stem,
                    "demand":                demand,
                    "fulfilled":             fulfilled,
                    "unfulfilled":           unfilled,
                    "fill_rate":             fill_rate,
                    "container_days_late":   cd_late,
                    "late_groups":           late_groups,
                    "mean_days_late_of_late": round(float(mean_late), 2),
                })
    return pd.DataFrame(rows)


def aggregate_by_instance(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (instance, model) summing across iterations."""
    if summary.empty:
        return summary
    grouped = summary.groupby(["instance", "start_year", "start_week", "model"],
                              as_index=False).agg(
        iterations=("iteration", "count"),
        demand=("demand", "sum"),
        fulfilled=("fulfilled", "sum"),
        unfulfilled=("unfulfilled", "sum"),
        container_days_late=("container_days_late", "sum"),
        late_groups=("late_groups", "sum"),
    )
    grouped["fill_rate"] = grouped["fulfilled"] / grouped["demand"]
    return grouped.sort_values(["start_year", "start_week", "model"]).reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    for year, week in INSTANCES:
        for model_short, model_file in MODELS:
            run_one(year, week, model_short, model_file)

    summary = aggregate_summary()
    summary_path = COMPARISON_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote per-iteration summary: {summary_path}  ({len(summary)} rows)")

    by_inst = aggregate_by_instance(summary)
    by_path = COMPARISON_DIR / "summary_by_instance.csv"
    by_inst.to_csv(by_path, index=False)
    print(f"Wrote per-instance summary:  {by_path}")
    print()
    if not by_inst.empty:
        print(by_inst.to_string(index=False))


if __name__ == "__main__":
    main()
