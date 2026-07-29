

import importlib.util
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
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR  = RESULTS_DIR / "proactive-repositioning"

YEARS                  = [(2023, 1), (2024, 1), (2025, 1)]
NUM_ITERATIONS         = 52
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1
SCENARIO               = "heavy"
MODEL_FILE             = "models/max-demand-fulfilment.py"

DEFICIT_PORTS = "IS REY,FO THO,GB IMM"
HUB_PORTS     = "NL RTM,DK AAR"

# (config, terminal_reward, terminal_ports, idle_penalty, idle_ports)
CONFIGS = [
    ("C0_baseline", 0.0,  "",            0.0,  ""),
    ("C1_terminal", 0.05, DEFICIT_PORTS, 0.0,  ""),
    ("C2_idle",     0.0,  "",            0.01, HUB_PORTS),
    ("C3_combined", 0.05, DEFICIT_PORTS, 0.01, HUB_PORTS),
]

SNAPSHOT_DIRS = ["stock", "model-flows", "demand-fulfilment"]

FILL_RE = re.compile(
    r"Decision-window fulfillment across \d+ iterations:\s*"
    r"(\d+)/(\d+) fulfilled.*?\(([\d.]+)%\)"
)

# Reuse empty-flow helpers from the analysis script.
_eca_path = REPO_ROOT / "analysis" / "empty-container-analysis.py"
_spec = importlib.util.spec_from_file_location("eca", _eca_path)
_eca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eca)


def run_one(config: str, treward: float, tports: str,
            ipen: float, iports: str, year: int, week: int) -> dict:
    label = f"{year}w{week}"
    target = OUTPUT_DIR / config / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for sub in SNAPSHOT_DIRS:
        d = RESULTS_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RUNNER_SCENARIO"]               = SCENARIO
    env["RUNNER_MODEL_FILE"]             = MODEL_FILE
    env["RUNNER_NUM_ITERATIONS"]         = str(NUM_ITERATIONS)
    env["RUNNER_PLANNING_HORIZON_WEEKS"] = str(PLANNING_HORIZON_WEEKS)
    env["RUNNER_DECISION_HORIZON_WEEKS"] = str(DECISION_HORIZON_WEEKS)
    env["RUNNER_START_YEAR"]             = str(year)
    env["RUNNER_START_WEEK"]             = str(week)
    env["RUNNER_SKIP_RESULTS"]           = "vessel-utilisation"
    env["RUNNER_TERMINAL_EMPTY_REWARD"]  = str(treward)
    env["RUNNER_TERMINAL_EMPTY_PORTS"]   = tports
    env["RUNNER_IDLE_EMPTY_PENALTY"]     = str(ipen)
    env["RUNNER_IDLE_EMPTY_PORTS"]       = iports

    print(f"\n{'#'*70}\n# {config} | {label} | treward={treward} ipen={ipen}\n{'#'*70}")

    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, str(RUNNER)], cwd=str(REPO_ROOT),
                         env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.stdout:
        print(res.stdout[-2000:])
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"runner failed: {config} {label} (exit {res.returncode})")

    for sub in SNAPSHOT_DIRS:
        src = RESULTS_DIR / sub
        if src.exists():
            shutil.copytree(src, target / sub)

    m = None
    for line in res.stdout.splitlines():
        mm = FILL_RE.search(line)
        if mm:
            m = mm
    if m is None:
        raise RuntimeError(f"no fill-rate line for {config} {label}")
    fulfilled, demand, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))

    return {"config": config, "instance": label, "year": year,
            "decision_demand": demand, "decision_fulfilled": fulfilled,
            "fill_rate_pct": pct, "wall_seconds": round(wall, 1)}


def empty_metrics(config: str) -> dict:
    """Mean IS REY weekly net flow + NL RTM idle TEU over this config's snapshots."""
    rey_net, rtm_idle, n = [], [], 0
    for year, week in YEARS:
        flow_dir = OUTPUT_DIR / config / f"{year}w{week}" / "model-flows"
        if not flow_dir.exists():
            continue
        for csv in sorted(flow_dir.glob("*.csv")):
            df = pd.read_csv(csv)
            df = df[df["DepHour"] < _eca.DECISION_HOURS]
            e  = _eca.parse_empty_rows(df)
            if e.empty:
                continue
            net  = _eca.per_port_net_flow_sailing(e)
            idle = _eca.idle_empty_inventory(e)
            rey_net.append(float(net.get("IS REY", 0.0)))
            rtm_idle.append(float(idle.get("NL RTM", 0.0)))
            n += 1
    import numpy as np
    return {
        "is_rey_mean_net_flow": round(float(np.mean(rey_net)), 1) if rey_net else None,
        "nl_rtm_mean_idle_teu": round(float(np.mean(rtm_idle)), 1) if rtm_idle else None,
        "n_windows": n,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config, treward, tports, ipen, iports in CONFIGS:
        for year, week in YEARS:
            rows.append(run_one(config, treward, tports, ipen, iports, year, week))
    df = pd.DataFrame(rows)

    # Aggregate per config + attach empty metrics.
    agg = (df.groupby("config", as_index=False)
             .agg(decision_demand=("decision_demand", "sum"),
                  decision_fulfilled=("decision_fulfilled", "sum"),
                  mean_fill_rate_pct=("fill_rate_pct", "mean"),
                  total_wall_s=("wall_seconds", "sum")))
    agg["agg_fill_rate_pct"] = (agg["decision_fulfilled"] / agg["decision_demand"] * 100).round(3)
    em = pd.DataFrame([{"config": c, **empty_metrics(c)} for c, *_ in CONFIGS])
    agg = agg.merge(em, on="config", how="left")

    df.to_csv(OUTPUT_DIR / "summary_by_instance.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\n=== Per-instance ===")
    print(df.to_string(index=False))
    print("\n=== Per-config (vs 99.9% single-block ceiling) ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
