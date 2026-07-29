

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT      = Path(__file__).resolve().parent.parent
RUNNER         = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR    = REPO_ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "model-comparison-single-block"

# Start-of-year and mid-year (week 27) start points. Mid-year instances reuse
# the 1 January stock snapshot (load_stock_df always reads Stock_0101{year}.csv
# for iteration 0), so their absolute fill rates are biased low — the snapshot
# no longer reflects the real fleet position by July. This bias is identical
# across all three models, so the *relative* comparison remains valid; mid-year
# instances simply widen the set of network conditions the models are tested on.
# 2026 has no mid-year instance (demand data ends at week 9).
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
NUM_ITERATIONS         = 1
PLANNING_HORIZON_WEEKS = 8            # ~2 months
DECISION_HORIZON_WEEKS = 8            # locks the full block — no rolling

SNAPSHOT_DIRS = ["demand-fulfilment", "model-flows", "stock", "vessel-utilisation"]

# Skip instance/model pairs that already have complete results, so re-running
# the script only fills in what's missing (e.g. newly added instances). Set
# FORCE_RERUN=1 in the environment to recompute everything from scratch.
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"


def _is_complete(target_dir: Path) -> bool:
    """A run is complete if its demand-fulfilment snapshot and timing files exist."""
    df_dir = target_dir / "demand-fulfilment"
    return (
        df_dir.is_dir()
        and any(df_dir.glob("*.csv"))
        and (target_dir / "wall_time_seconds.txt").exists()
        and (target_dir / "build_times_seconds.txt").exists()
    )


def _read_cached_times(target_dir: Path):
    """Recover (wall, phase1, phase2, gurobi, network, commodity) from saved files."""
    def _parse(path: Path) -> dict:
        out: dict = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = None if v.strip() == "None" else float(v.strip())
        return out

    wall = None
    wt = (target_dir / "wall_time_seconds.txt")
    if wt.exists():
        wall = float(wt.read_text().strip())
    solve = _parse(target_dir / "solve_times_seconds.txt")
    build = _parse(target_dir / "build_times_seconds.txt")
    return (wall, solve.get("phase1"), solve.get("phase2"), solve.get("gurobi"),
            build.get("network"), build.get("commodity"))


def run_one(year: int, week: int, model_short: str, model_file: str) -> tuple[Path, float]:
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
    print(f"# single-block: planning={PLANNING_HORIZON_WEEKS}w decision={DECISION_HORIZON_WEEKS}w")
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

    # Echo so the user still sees runner output live-ish (after each run).
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

    # Parse Gurobi runtime + build-time markers.
    phase1 = phase2 = gurobi = None
    network_build = commodity_build = None
    for line in result.stdout.splitlines():
        if "SEQ_PHASE1_RUNTIME_S=" in line:
            phase1 = float(line.split("=", 1)[1].strip())
        elif "SEQ_PHASE2_RUNTIME_S=" in line:
            phase2 = float(line.split("=", 1)[1].strip())
        elif "GUROBI_RUNTIME_S=" in line:
            gurobi = float(line.split("=", 1)[1].strip())
        elif "NETWORK_BUILD_S=" in line:
            network_build = float(line.split("=", 1)[1].strip())
        elif "COMMODITY_BUILD_S=" in line:
            commodity_build = float(line.split("=", 1)[1].strip())

    (target_dir / "wall_time_seconds.txt").write_text(f"{wall:.3f}\n")
    if phase1 is not None or phase2 is not None or gurobi is not None:
        (target_dir / "solve_times_seconds.txt").write_text(
            f"gurobi={gurobi}\nphase1={phase1}\nphase2={phase2}\n"
        )
    if network_build is not None or commodity_build is not None:
        (target_dir / "build_times_seconds.txt").write_text(
            f"network={network_build}\ncommodity={commodity_build}\n"
        )
    return target_dir, wall, phase1, phase2, gurobi, network_build, commodity_build


def aggregate(timings: dict[tuple[str, str], float],
              phase_times: dict[tuple[str, str], tuple[float, float]],
              gurobi_times: dict[tuple[str, str], float],
              build_times: dict[tuple[str, str], tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for year, week in INSTANCES:
        inst_label = f"{year}w{week}"
        for model_short, _ in MODELS:
            run_dir = COMPARISON_DIR / f"{inst_label}_{model_short}" / "demand-fulfilment"
            if not run_dir.exists():
                continue
            # A single-block run produces exactly one decision-window CSV named
            # after the instance. Read only that file — globbing *.csv would also
            # pick up any stray snapshots that leaked into results/ from a
            # concurrently-running experiment before the copytree ran.
            inst_csv = run_dir / f"{inst_label}.csv"
            csvs = [inst_csv] if inst_csv.exists() else sorted(run_dir.glob("*.csv"))
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
                "demand":              demand,
                "fulfilled":           fulfilled,
                "unfulfilled":         unfilled,
                "fill_rate":           round(fill_rate, 4),
                "container_days_late": cd_late,
                "late_groups":         late_groups,
                "mean_days_late_of_late": round(sum(mean_lates)/len(mean_lates), 2) if mean_lates else 0.0,
                "wall_seconds":        round(timings.get((inst_label, model_short), float("nan")), 2),
                "gurobi_seconds":      (round(gurobi_times[(inst_label, model_short)], 2)
                                        if (inst_label, model_short) in gurobi_times
                                        and gurobi_times[(inst_label, model_short)] is not None
                                        else ""),
                "phase1_seconds":      (round(phase_times[(inst_label, model_short)][0], 2)
                                        if (inst_label, model_short) in phase_times
                                        and phase_times[(inst_label, model_short)][0] is not None
                                        else ""),
                "phase2_seconds":      (round(phase_times[(inst_label, model_short)][1], 2)
                                        if (inst_label, model_short) in phase_times
                                        and phase_times[(inst_label, model_short)][1] is not None
                                        else ""),
                "network_build_seconds":   (round(build_times[(inst_label, model_short)][0], 2)
                                            if (inst_label, model_short) in build_times
                                            and build_times[(inst_label, model_short)][0] is not None
                                            else ""),
                "commodity_build_seconds": (round(build_times[(inst_label, model_short)][1], 2)
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
