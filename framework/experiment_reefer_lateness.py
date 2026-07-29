import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
RUNNER      = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR = REPO_ROOT / "results" / "delay-experiments"
COMMOD_DIR  = REPO_ROOT / "data" / "processed" / "commodities"

MODEL_FILE             = "models/min-lateness-fulfilment.py"
START_YEAR             = 2025
START_WEEK             = 1
NUM_ITERATIONS         = 5
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1
SCENARIO               = "baseline"

SNAPSHOT_DIRS = ["demand-fulfilment"]

# (mu_dry, mu_reefer) pairs. Equal pair == uniform penalty.
CONFIGS = [
    (0.1,  0.1),
    (0.1,  0.12),
    (0.1,  0.15),
    (0.1,  0.2),
    (0.15, 0.15),
    (0.15, 0.2),
]


def is_reefer_type(ct: str) -> bool:
    return isinstance(ct, str) and len(ct) >= 3 and ct[2].upper() == "R"


def planning_file(iter_idx: int) -> Path:
    """Commodities file for iteration iter_idx (1-based): 3-week planning horizon."""
    start = START_WEEK + (iter_idx - 1) * DECISION_HORIZON_WEEKS
    end   = start + PLANNING_HORIZON_WEEKS - 1
    return COMMOD_DIR / f"{START_YEAR}w{start}-w{end}.csv"


def run_one(label: str, mu_dry: float, mu_reefer: float) -> dict:
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
    env["PYTHONHASHSEED"] = "0"          # pin alternate-optima jitter (see memory)
    env.update({
        "RUNNER_MODEL_FILE":             MODEL_FILE,
        "RUNNER_SCENARIO":               SCENARIO,
        "RUNNER_NUM_ITERATIONS":         str(NUM_ITERATIONS),
        "RUNNER_PLANNING_HORIZON_WEEKS": str(PLANNING_HORIZON_WEEKS),
        "RUNNER_DECISION_HORIZON_WEEKS": str(DECISION_HORIZON_WEEKS),
        "RUNNER_START_YEAR":             str(START_YEAR),
        "RUNNER_START_WEEK":             str(START_WEEK),
        "RUNNER_MU_LATENESS":            str(mu_dry),
        "RUNNER_MU_REEFER":              str(mu_reefer),
    })

    print(f"\n{'='*60}\n  {label}  (mu_dry={mu_dry}, mu_reefer={mu_reefer})\n{'='*60}")
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, str(RUNNER)], cwd=str(REPO_ROOT),
                         env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"runner failed: {label} (exit {res.returncode})")

    # Snapshot demand-fulfilment into the labelled dir.
    src = RESULTS_DIR.parent / "demand-fulfilment"
    shutil.copytree(src, target / "demand-fulfilment", dirs_exist_ok=True)

    # ── Split lateness by cargo type, summed across the 5 iterations ──────────
    # Fill rate is decision-window customer-only (matches the runner summary).
    reefer_cd = dry_cd = total_cd = 0.0
    reefer_containers_late = 0
    dw_demand = dw_fulfilled = 0
    for it in range(1, NUM_ITERATIONS + 1):
        start = START_WEEK + (it - 1) * DECISION_HORIZON_WEEKS
        snap  = target / "demand-fulfilment" / f"{START_YEAR}w{start}.csv"
        ful   = pd.read_csv(snap)

        comm = pd.read_csv(planning_file(it)).set_index("Commodity")
        ful["ContainerType"] = ful["Commodity"].map(comm["ContainerType"])
        missing = ful["ContainerType"].isna().sum()
        if missing:
            print(f"  WARNING: {missing} commodities in {snap.name} unmatched to a type")

        reefer_mask = ful["ContainerType"].apply(is_reefer_type)
        reefer_cd += ful.loc[reefer_mask, "LatenessDays"].sum()
        dry_cd    += ful.loc[~reefer_mask, "LatenessDays"].sum()
        total_cd  += ful["LatenessDays"].sum()
        reefer_containers_late += int((ful.loc[reefer_mask, "LatenessDays"] > 0).sum())

        # Decision-window fill: customer commodities departing within week 1.
        cutoff = DECISION_HORIZON_WEEKS * 168
        dep = ful["Commodity"].map(comm["DepartureTime"])
        syn = ful["Commodity"].astype(str).str.startswith(("CARRY_", "TLEG_", "DLAY_"))
        dw  = ful[(dep < cutoff) & ~syn]
        dw_demand    += int(dw["Demand"].sum())
        dw_fulfilled += int(dw["Fulfilled"].sum())

    fill_pct = dw_fulfilled / dw_demand * 100 if dw_demand else 0.0
    print(f"  reefer={reefer_cd:.0f} cd | dry={dry_cd:.0f} cd | total={total_cd:.0f} cd "
          f"| fill={fill_pct:.2f}% | wall={wall:.0f}s")

    return {
        "label":            label,
        "mu_dry":           mu_dry,
        "mu_reefer":        mu_reefer,
        "fill_rate_pct":    round(fill_pct, 2),
        "reefer_late_cd":   round(reefer_cd),
        "dry_late_cd":      round(dry_cd),
        "total_late_cd":    round(total_cd),
        "reefer_comms_late": reefer_containers_late,
        "wall_seconds":     round(wall, 1),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mu_dry, mu_reefer in CONFIGS:
        tag = "uniform" if mu_reefer == mu_dry else f"{mu_reefer/mu_dry:.2g}x"
        label = f"REEFER_d{mu_dry}_r{mu_reefer}"
        rows.append(run_one(label, mu_dry, mu_reefer))
        rows[-1]["tier"] = tag

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "exp_C_reefer_lateness.csv"
    df.to_csv(out, index=False)
    print(f"\n{'='*60}\nReefer-lateness sweep saved to: {out}\n{'='*60}")
    print(df[["mu_dry", "mu_reefer", "tier", "fill_rate_pct",
              "reefer_late_cd", "dry_late_cd", "total_late_cd"]].to_string(index=False))


if __name__ == "__main__":
    main()
