

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
RUNNER         = REPO_ROOT / "framework" / "runner.py"
RESULTS_DIR    = REPO_ROOT / "results"
OUTPUT_DIR     = RESULTS_DIR / "full-year-rejection"

INSTANCES: list[tuple[int, int]] = [
    (2023, 1),
    (2024, 1),
    (2025, 1),
]

MODEL_FILE             = "models/max-demand-fulfilment.py"
SCENARIO               = "heavy"   # +20% augmented demand
NUM_ITERATIONS         = 52
PLANNING_HORIZON_WEEKS = 3
DECISION_HORIZON_WEEKS = 1
SKIP_RESULTS           = "vessel-utilisation,demand-fulfilment"

SNAPSHOT_DIRS = ["stock", "model-flows"]


def run_one(year: int, week: int) -> tuple[Path, float]:
    inst_label = f"{year}w{week}"
    target_dir = OUTPUT_DIR / inst_label
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
    env["RUNNER_MODEL_FILE"]             = MODEL_FILE
    env["RUNNER_NUM_ITERATIONS"]         = str(NUM_ITERATIONS)
    env["RUNNER_PLANNING_HORIZON_WEEKS"] = str(PLANNING_HORIZON_WEEKS)
    env["RUNNER_DECISION_HORIZON_WEEKS"] = str(DECISION_HORIZON_WEEKS)
    env["RUNNER_START_YEAR"]             = str(year)
    env["RUNNER_START_WEEK"]             = str(week)
    env["RUNNER_SKIP_RESULTS"]           = SKIP_RESULTS

    print(f"\n{'#'*70}")
    print(f"# Full-year rejection sweep | instance {inst_label}")
    print(f"# {NUM_ITERATIONS} iter × planning={PLANNING_HORIZON_WEEKS}w decision={DECISION_HORIZON_WEEKS}w")
    print(f"{'#'*70}")

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(REPO_ROOT),
        env=env,
    )
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(f"runner.py failed for instance={inst_label} (exit {result.returncode})")

    for sub in SNAPSHOT_DIRS:
        src = RESULTS_DIR / sub
        dst = target_dir / sub
        if src.exists():
            shutil.copytree(src, dst)

    (target_dir / "wall_time_seconds.txt").write_text(f"{wall:.3f}\n")
    return target_dir, wall


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for year, week in INSTANCES:
        run_one(year, week)


if __name__ == "__main__":
    main()
