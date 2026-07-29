import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
RUNNER      = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR = REPO_ROOT / "results" / "delay-experiments"

MODEL_FILE             = "models/min-lateness-fulfilment.py"
START_YEAR             = 2025
START_WEEK             = 1
NUM_ITERATIONS         = 5
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1
SCENARIO               = "baseline"

SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock"]

FILL_RE = re.compile(
    r"Decision-window fulfillment across \d+ iterations:\s*"
    r"(\d+)/(\d+) fulfilled.*?\(([\d.]+)%\)"
)
LATENESS_RE = re.compile(r"Total lateness:\s*([\d.]+)\s*container-days")

# ── Exp B: mu_lateness sweep ──────────────────────────────────────────────────

EXP_B_MU_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]

# ── Exp C: Reefer vs Dry tiers ────────────────────────────────────────────────
# (mu_dry, mu_reefer)
EXP_C_CONFIGS = [
    ("uniform",  0.1,  0.0),   # baseline — same penalty for all
    ("tier_2x",  0.1,  0.2),   # reefer 2× dry
    ("tier_5x",  0.1,  0.5),   # reefer 5× dry
    ("tier_10x", 0.1,  1.0),   # reefer 10× dry
]

# ── Exp D: maximum delay cap ──────────────────────────────────────────────────
# 0 = no cap (soft penalty only)
EXP_D_MAX_DAYS = [0, 8, 6, 4]


def run_one(label: str, extra_env: dict) -> dict:
    target = RESULTS_DIR / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

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
        print(res.stdout[-3000:])
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"runner failed: {label} (exit {res.returncode})")

    # Copy snapshots into labelled directory
    for sub in SNAPSHOT_DIRS:
        src = RESULTS_DIR.parent / sub
        if src.exists():
            shutil.copytree(src, target / sub, dirs_exist_ok=True)

    # Parse summary metrics from stdout
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

    row = {
        "label": label,
        "fill_rate_pct": pct,
        "fulfilled": fulfilled,
        "demand": demand,
        "total_lateness_cd": total_lateness,
        "wall_seconds": round(wall, 1),
    }
    row.update(extra_env)
    return row


def run_exp_b() -> pd.DataFrame:
    print("\n### Experiment B: mu_lateness sweep ###")
    rows = []
    for mu in EXP_B_MU_VALUES:
        label = f"B_mu{mu}"
        row = run_one(label, {"RUNNER_MU_LATENESS": str(mu)})
        row["mu_lateness"] = mu
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "exp_B_mu_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nExp B results saved to: {out}")
    print(df[["label", "mu_lateness", "fill_rate_pct", "total_lateness_cd"]].to_string(index=False))
    return df


def run_exp_c() -> pd.DataFrame:
    print("\n### Experiment C: Reefer vs Dry time-sensitivity tiers ###")
    rows = []
    for name, mu_dry, mu_reefer in EXP_C_CONFIGS:
        label = f"C_{name}"
        extra = {"RUNNER_MU_LATENESS": str(mu_dry)}
        if mu_reefer > 0:
            extra["RUNNER_MU_REEFER"] = str(mu_reefer)
        row = run_one(label, extra)
        row["mu_dry"]   = mu_dry
        row["mu_reefer"] = mu_reefer
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "exp_C_reefer_tiers.csv"
    df.to_csv(out, index=False)
    print(f"\nExp C results saved to: {out}")
    print(df[["label", "mu_dry", "mu_reefer", "fill_rate_pct", "total_lateness_cd"]].to_string(index=False))
    return df


def run_exp_d() -> pd.DataFrame:
    print("\n### Experiment D: hard maximum delay cap ###")
    rows = []
    for max_days in EXP_D_MAX_DAYS:
        label = f"D_maxdelay{max_days}" if max_days > 0 else "D_nocap"
        row = run_one(label, {"RUNNER_MAX_DELAY_DAYS": str(max_days)})
        row["max_delay_days"] = max_days
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "exp_D_max_delay.csv"
    df.to_csv(out, index=False)
    print(f"\nExp D results saved to: {out}")
    print(df[["label", "max_delay_days", "fill_rate_pct", "total_lateness_cd"]].to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["B", "C", "D"], default=None,
                        help="Run a single experiment (B, C, or D). Default: all.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp is None or args.exp == "B":
        run_exp_b()
    if args.exp is None or args.exp == "C":
        run_exp_c()
    if args.exp is None or args.exp == "D":
        run_exp_d()
