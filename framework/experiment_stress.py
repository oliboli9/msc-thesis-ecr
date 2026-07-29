import argparse
import os
import re
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
RESULTS_DIR = REPO_ROOT / "results" / "stress-experiments"

MODEL_FILE             = "models/max-demand-fulfilment.py"  # rejection model
SCENARIO               = "moderate"                          # +10%, as in §6.3.1
NUM_ITERATIONS         = 5
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1

# Single start-week instance — a clean per-iteration recovery curve.
START_YEAR = 2025
START_WEEK = 1

# Vessels to remove, one scenario each (3-char code = first three letters of a
# voyage code). Chosen to span lines/sizes among vessels active in the 2025 w1
# window: DET (Red, 2148 TEU), JOR (Blue, 1853 TEU), BAK (Green, 880 TEU).
VESSELS = ["DET", "JOR", "BAK"]

# Iterations in which the vessel is unavailable — the whole 5-iteration horizon.
DISABLE_ITERS = "1,2,3,4,5"

SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock"]

# Per-iteration rows of the runner's rolling-horizon summary table, e.g.
#   "   2     2025w2    559   1502    2075   2075   0   0   0   98.0"
# Group 1 = 1-based iteration index, group 2 = trailing Rate% column.
ITER_RATE_RE = re.compile(r"^\s*(\d+)\s+\d{4}w\d+\s+.*?\s+([\d.]+)\s*$", re.M)


def scenarios(smoke: bool) -> list[tuple[str, dict]]:
    """(label, extra_env) per scenario: baseline + one per disabled vessel."""
    vessels = VESSELS[:1] if smoke else VESSELS
    out = [("baseline", {})]
    for v in vessels:
        out.append((v, {"RUNNER_DISABLE_VESSEL": v,
                        "RUNNER_DISABLE_ITERS": DISABLE_ITERS}))
    return out


def run_one(label: str, extra_env: dict, keep_snapshot: bool = True) -> tuple[dict, list[dict]]:
    """Run one rolling-horizon execution; return (summary row, per-iter rows)."""
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
        "RUNNER_START_YEAR":             str(START_YEAR),
        "RUNNER_START_WEEK":             str(START_WEEK),
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

    # Overall fill rate (last match) + total lateness.
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

    summary = {
        "scenario": label,
        "fill_rate_pct": pct,
        "fulfilled": fulfilled,
        "demand": demand,
        "total_lateness_cd": total_lateness,
        "wall_seconds": round(wall, 1),
    }

    # Per-iteration fill rate from the rolling-horizon summary table.
    per_iter = [{"scenario": label, "iteration": int(it), "fill_rate_pct": float(rate)}
                for it, rate in ITER_RATE_RE.findall(res.stdout)]
    if len(per_iter) != NUM_ITERATIONS:
        print(f"WARNING: parsed {len(per_iter)} per-iteration rows for {label} "
              f"(expected {NUM_ITERATIONS})")

    return summary, per_iter


def summarise(summary_rows: list[dict], iter_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "exp_stress.csv", index=False)

    di = pd.DataFrame(iter_rows)
    di.to_csv(RESULTS_DIR / "exp_stress_by_iter.csv", index=False)

    print("\nOverall decision-window fill rate by scenario:")
    print(df[["scenario", "fill_rate_pct", "fulfilled", "demand"]].to_string(index=False))
    print("\nPer-iteration fill rate (%):")
    print(di.pivot(index="iteration", columns="scenario",
                   values="fill_rate_pct").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick check: baseline + transient only.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows, iter_rows = [], []
    for label, extra in scenarios(args.smoke):
        s, pi = run_one(label, extra)
        summary_rows.append(s)
        iter_rows.extend(pi)

    summarise(summary_rows, iter_rows)
