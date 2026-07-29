

import os
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT      = Path(__file__).resolve().parent.parent
RUNNER         = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR    = REPO_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "model-comparison-rolling"

# Start-of-year and mid-year (week 26) start points. Mid-year instances reuse
# the 1 January stock snapshot (the runner reads Stock_0101{year}.csv for
# iteration 0), so their absolute fill rates are biased low — the snapshot no
# longer reflects the real fleet position by July. This bias is identical
# across all three models, so the relative comparison stays valid; mid-year
# instances simply widen the set of network conditions tested. 2026 has no
# mid-year instance (demand data ends at week 9).
INSTANCES: list[tuple[int, int]] = [
    (2023, 1),
    (2024, 1),
    (2025, 1),
    (2026, 1),
    (2023, 26),
    (2024, 26),
    (2025, 26),
]

MODELS: list[tuple[str, str]] = [
    ("rejection",  "models/max-demand-fulfilment.py"),
    ("delay",      "models/min-lateness-fulfilment.py"),
    ("sequential", "models/sequential-demand-then-reposition.py"),
]

SCENARIO               = "moderate"   # +10% augmented demand
NUM_ITERATIONS         = 5
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1

SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock", "vessel-utilisation"]

# Skip instance/model pairs that already have complete results, so re-running
# the script only fills in what's missing (e.g. newly added mid-year instances).
# Set FORCE_RERUN=1 in the environment to recompute everything from scratch.
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"


def _advance_weeks(year: int, week: int, n: int) -> tuple[int, int]:
    """Advance (ISO year, ISO week) by n weeks (mirrors runner.advance_by_weeks)."""
    jan4 = date(year, 1, 4)
    monday = jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1 + n)
    iso = monday.isocalendar()
    return iso[0], iso[1]


def _expected_decision_labels(year: int, week: int) -> list[str]:
    """ISO labels of the NUM_ITERATIONS decision weeks for this instance."""
    return [f"{y}w{w}" for (y, w) in
            (_advance_weeks(year, week, i) for i in range(NUM_ITERATIONS))]


def _is_complete(target_dir: Path) -> bool:
    """A run is complete if its demand-fulfilment snapshot and wall time exist.

    Build-time files are NOT required: dirs produced before build-time
    instrumentation was added still count as complete (their build columns are
    simply left blank in the summary).
    """
    df_dir = target_dir / "demand-fulfilment"
    return (
        df_dir.is_dir()
        and any(df_dir.glob("*.csv"))
        and (target_dir / "wall_time_seconds.txt").exists()
    )


def _read_cached_times(target_dir: Path):
    """Recover (wall, phase1, phase2, gurobi, network, commodity) from saved files.

    The rolling solve_times file stores totals with an iteration count suffix,
    e.g. ``gurobi_total=6.52 (n=5)`` — strip the suffix before parsing.
    """
    def _num(s: str):
        s = s.split("(", 1)[0].strip()
        return None if s == "None" else float(s)

    def _parse(path: Path) -> dict:
        out: dict = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = _num(v)
        return out

    wall = None
    wt = target_dir / "wall_time_seconds.txt"
    if wt.exists():
        wall = float(wt.read_text().strip())
    solve = _parse(target_dir / "solve_times_seconds.txt")
    build = _parse(target_dir / "build_times_seconds.txt")
    return (wall, solve.get("phase1_total"), solve.get("phase2_total"),
            solve.get("gurobi_total"), build.get("network"), build.get("commodity"))


def run_one(year: int, week: int, model_short: str, model_file: str):
    inst_label = f"{year}w{week}"
    target_dir = COMPARISON_DIR / f"{inst_label}_{model_short}"

    if not FORCE_RERUN and _is_complete(target_dir):
        print(f"\n# Skipping {inst_label} | model={model_short} — results already present "
              f"(set FORCE_RERUN=1 to recompute)")
        wall, p1, p2, g, nb, cb = _read_cached_times(target_dir)
        return target_dir, wall, p1, p2, g, nb, cb

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"# rolling: {NUM_ITERATIONS} iter × planning={PLANNING_HORIZON_WEEKS}w "
          f"decision={DECISION_HORIZON_WEEKS}w")
    print(f"{'#'*70}")

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

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

    # Parse and sum runtime + build markers across iterations.
    phase1_total = phase2_total = gurobi_total = 0.0
    network_total = commodity_total = 0.0
    phase1_n = phase2_n = gurobi_n = network_n = commodity_n = 0
    for line in result.stdout.splitlines():
        if "SEQ_PHASE1_RUNTIME_S=" in line:
            phase1_total += float(line.split("=", 1)[1].strip()); phase1_n += 1
        elif "SEQ_PHASE2_RUNTIME_S=" in line:
            phase2_total += float(line.split("=", 1)[1].strip()); phase2_n += 1
        elif "GUROBI_RUNTIME_S=" in line:
            gurobi_total += float(line.split("=", 1)[1].strip()); gurobi_n += 1
        elif "NETWORK_BUILD_S=" in line:
            network_total += float(line.split("=", 1)[1].strip()); network_n += 1
        elif "COMMODITY_BUILD_S=" in line:
            commodity_total += float(line.split("=", 1)[1].strip()); commodity_n += 1

    phase1 = phase1_total if phase1_n else None
    phase2 = phase2_total if phase2_n else None
    gurobi = gurobi_total if gurobi_n else None
    network = network_total if network_n else None
    commodity = commodity_total if commodity_n else None

    (target_dir / "wall_time_seconds.txt").write_text(f"{wall:.3f}\n")
    if phase1 is not None or phase2 is not None or gurobi is not None:
        (target_dir / "solve_times_seconds.txt").write_text(
            f"gurobi_total={gurobi} (n={gurobi_n})\n"
            f"phase1_total={phase1} (n={phase1_n})\n"
            f"phase2_total={phase2} (n={phase2_n})\n"
        )
    if network is not None or commodity is not None:
        (target_dir / "build_times_seconds.txt").write_text(
            f"network={network} (n={network_n})\n"
            f"commodity={commodity} (n={commodity_n})\n"
        )
    return target_dir, wall, phase1, phase2, gurobi, network, commodity


def aggregate(timings, phase_times, gurobi_times, build_times) -> pd.DataFrame:
    rows = []
    for year, week in INSTANCES:
        inst_label = f"{year}w{week}"
        expected = _expected_decision_labels(year, week)
        for model_short, _ in MODELS:
            run_dir = COMPARISON_DIR / f"{inst_label}_{model_short}" / "demand-fulfilment"
            if not run_dir.exists():
                continue
            # Read only this run's own decision-week CSVs. Globbing *.csv would
            # also pick up stray snapshots leaked into results/ by a concurrently
            # running experiment before the copytree ran.
            csvs = [run_dir / f"{lbl}.csv" for lbl in expected
                    if (run_dir / f"{lbl}.csv").exists()]
            if not csvs:
                csvs = sorted(run_dir.glob("*.csv"))
            demand = fulfilled = unfilled = cd_late = late_groups = 0
            mean_lates = []
            for csv in csvs:
                df = pd.read_csv(csv)
                demand    += int(df["Demand"].sum())
                fulfilled += int(df["Fulfilled"].sum())
                unfilled  += int((df["Demand"] - df["Fulfilled"]).sum())
                if "LatenessDays" in df.columns:
                    cd_late     += int(df["LatenessDays"].sum())
                    n_late       = int((df["LatenessDays"] > 0).sum())
                    late_groups += n_late
                    if n_late > 0:
                        mean_lates.append(
                            float(df.loc[df["LatenessDays"] > 0, "MeanLatenessDays"].mean())
                        )
            fill_rate = fulfilled / demand if demand else 0.0
            rows.append({
                "instance":            inst_label,
                "model":               model_short,
                "iterations":          len(csvs),
                "demand":              demand,
                "fulfilled":           fulfilled,
                "unfulfilled":         unfilled,
                "fill_rate":           round(fill_rate, 4),
                "container_days_late": cd_late,
                "late_groups":         late_groups,
                "mean_days_late_of_late": round(sum(mean_lates)/len(mean_lates), 2) if mean_lates else 0.0,
                "wall_seconds":        round(timings.get((inst_label, model_short), float("nan")), 2),
                "gurobi_total_seconds":      (round(gurobi_times[(inst_label, model_short)], 2)
                                              if (inst_label, model_short) in gurobi_times
                                              and gurobi_times[(inst_label, model_short)] is not None
                                              else ""),
                "phase1_total_seconds":      (round(phase_times[(inst_label, model_short)][0], 2)
                                              if (inst_label, model_short) in phase_times
                                              and phase_times[(inst_label, model_short)][0] is not None
                                              else ""),
                "phase2_total_seconds":      (round(phase_times[(inst_label, model_short)][1], 2)
                                              if (inst_label, model_short) in phase_times
                                              and phase_times[(inst_label, model_short)][1] is not None
                                              else ""),
                "network_build_total_seconds":   (round(build_times[(inst_label, model_short)][0], 2)
                                                  if (inst_label, model_short) in build_times
                                                  and build_times[(inst_label, model_short)][0] is not None
                                                  else ""),
                "commodity_build_total_seconds": (round(build_times[(inst_label, model_short)][1], 2)
                                                  if (inst_label, model_short) in build_times
                                                  and build_times[(inst_label, model_short)][1] is not None
                                                  else ""),
            })
    return pd.DataFrame(rows)


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[tuple[str, str], float] = {}
    phase_times: dict[tuple[str, str], tuple[float, float]] = {}
    gurobi_times: dict[tuple[str, str], float] = {}
    build_times: dict[tuple[str, str], tuple[float, float]] = {}

    for year, week in INSTANCES:
        for model_short, model_file in MODELS:
            _, wall, p1, p2, g, nb, cb = run_one(year, week, model_short, model_file)
            timings[(f"{year}w{week}", model_short)] = wall
            if p1 is not None or p2 is not None:
                phase_times[(f"{year}w{week}", model_short)] = (p1, p2)
            if g is not None:
                gurobi_times[(f"{year}w{week}", model_short)] = g
            if nb is not None or cb is not None:
                build_times[(f"{year}w{week}", model_short)] = (nb, cb)

    summary = aggregate(timings, phase_times, gurobi_times, build_times)
    out = COMPARISON_DIR / "summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nWrote summary: {out}")
    print()
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
