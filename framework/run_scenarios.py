

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT     = Path(__file__).resolve().parent.parent
RUNNER        = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR   = REPO_ROOT / "results"
SCENARIOS_DIR = RESULTS_DIR / "scenarios"

# (scenario_name, target_pct_for_reporting, method)
# Baseline is method-independent; uniform/od-weighted variants share it.
SCENARIOS = [
    ("baseline",    0.00, "uniform"),
    ("light",       0.05, "uniform"),
    ("moderate",    0.10, "uniform"),
    ("heavy",       0.20, "uniform"),
    ("light_od",    0.05, "od-weighted"),
    ("moderate_od", 0.10, "od-weighted"),
    ("heavy_od",    0.20, "od-weighted"),
]

# (label, model_file_relpath)
MODELS = [
    ("rejection", "models/max-demand-fulfilment.py"),
    ("delay",     "models/min-lateness-fulfilment.py"),
]

# Subdirs in results/ that runner.py writes to per iteration.
SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock", "vessel-utilisation"]


def run_one(scenario: str, model_short: str, model_file: str) -> Path:
    """Run a single (scenario, model) combination and snapshot its results."""
    target_dir = SCENARIOS_DIR / f"{scenario}_{model_short}"
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
    env["RUNNER_SCENARIO"]   = scenario
    env["RUNNER_MODEL_FILE"] = model_file

    print(f"\n{'#'*70}")
    print(f"# Running scenario={scenario}  model={model_short}")
    print(f"{'#'*70}")

    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"runner.py failed for scenario={scenario}, model={model_short} "
            f"(exit {result.returncode})"
        )

    # Snapshot.
    for sub in SNAPSHOT_DIRS:
        src = RESULTS_DIR / sub
        dst = target_dir / sub
        if src.exists():
            shutil.copytree(src, dst)

    return target_dir


def aggregate_summary() -> pd.DataFrame:
    """Walk the per-scenario snapshots and build a tidy summary table."""
    rows = []
    for scenario, target_pct, method in SCENARIOS:
        for model_short, _ in MODELS:
            run_dir = SCENARIOS_DIR / f"{scenario}_{model_short}" / "demand-fulfilment"
            if not run_dir.exists():
                continue
            for csv in sorted(run_dir.glob("*.csv")):
                df = pd.read_csv(csv)
                demand    = int(df["Demand"].sum())
                fulfilled = int(df["Fulfilled"].sum())
                unfilled  = int((df["Demand"] - df["Fulfilled"]).sum())
                fill_rate = fulfilled / demand if demand else 0.0
                if "LatenessDays" in df.columns:
                    cd_late = int(df["LatenessDays"].sum())
                    late_groups = int((df["LatenessDays"] > 0).sum())
                    mean_late = (df.loc[df["LatenessDays"] > 0, "MeanLatenessDays"].mean()
                                 if late_groups > 0 else 0.0)
                else:
                    cd_late, late_groups, mean_late = 0, 0, 0.0
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "target_pct": target_pct * 100,
                    "model": model_short,
                    "iteration": csv.stem,  # e.g. "2025w1"
                    "demand": demand,
                    "fulfilled": fulfilled,
                    "unfulfilled": unfilled,
                    "fill_rate": fill_rate,
                    "container_days_late": cd_late,
                    "late_groups": late_groups,
                    "mean_days_late_of_late": round(float(mean_late), 2),
                })
    return pd.DataFrame(rows)


def aggregate_by_scenario(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, method, model) summing across iterations."""
    grouped = summary.groupby(["scenario", "method", "target_pct", "model"],
                              as_index=False).agg(
        demand=("demand", "sum"),
        fulfilled=("fulfilled", "sum"),
        unfulfilled=("unfulfilled", "sum"),
        container_days_late=("container_days_late", "sum"),
        late_groups=("late_groups", "sum"),
    )
    grouped["fill_rate"] = grouped["fulfilled"] / grouped["demand"]
    return grouped.sort_values(["target_pct", "method", "model"]).reset_index(drop=True)


def main():
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario, _, _ in SCENARIOS:
        for model_short, model_file in MODELS:
            run_one(scenario, model_short, model_file)

    summary = aggregate_summary()
    summary_path = SCENARIOS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote per-iteration summary: {summary_path}  ({len(summary)} rows)")

    by_scenario = aggregate_by_scenario(summary)
    by_path = SCENARIOS_DIR / "summary_by_scenario.csv"
    by_scenario.to_csv(by_path, index=False)
    print(f"Wrote per-scenario summary:  {by_path}")
    print()
    print(by_scenario.to_string(index=False))


if __name__ == "__main__":
    main()
