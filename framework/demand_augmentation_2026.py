import os
import sys
import time
import json
from pathlib import Path


if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import pandas as pd

sys.path.insert(0, ".")

from framework import runner

# ── Experiment parameters ────────────────────────────────────────────────────

YEAR             = 2026
START_WEEK       = 1
PLANNING_WEEKS   = 1
DECISION_WEEKS   = 1
NUM_ITERATIONS   = 1          # single iteration (decision week 1 only)

# (display alpha, scenario directory) per scheme. Baseline is shared.
# The +50% level emulates the busier mid-year demand (~50% above January) and
# serves as the baseline; the two growth scenarios add 5% and 10% on top of that
# mid-year baseline, i.e. 1.50*1.05 = +57.5% and 1.50*1.10 = +65% over January.
SCHEMES = {
    "uniform": [
        (0.50,  "aug50_uniform"),
        (0.575, "aug575_uniform"),
        (0.65,  "aug65_uniform"),
        (0.80,  "aug80_uniform"),
    ],
    "proportional": [
        (0.50,  "aug50_scaled"),
        (0.575, "aug575_scaled"),
        (0.65,  "aug65_scaled"),
        (0.80,  "aug80_scaled"),
    ],
}

RESULTS_DIR = Path("results/demand-augmentation")


def run_one(scenario: str, alpha: float) -> dict:
    """Override runner params, invoke runner.main(), return aggregate metrics."""
    runner.START_YEAR              = YEAR
    runner.START_WEEK              = START_WEEK
    runner.PLANNING_HORIZON_WEEKS  = PLANNING_WEEKS
    runner.DECISION_HORIZON_WEEKS  = DECISION_WEEKS
    runner.NUM_ITERATIONS          = NUM_ITERATIONS
    runner.SCENARIO                = scenario
    runner.TRUNCATE_HORIZON_AT_END = True

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: scenario={scenario} (alpha={alpha:+.0%}), "
          f"year={YEAR}, planning={PLANNING_WEEKS}w, decision={DECISION_WEEKS}w")
    print(f"{'#'*70}")

    t0 = time.time()
    iter_summaries = runner.main()
    elapsed = time.time() - t0

    iter_summaries = iter_summaries or []
    tot_dem        = sum(s["dw_demand"]      for s in iter_summaries)
    tot_ful        = sum(s["dw_fulfilled"]   for s in iter_summaries)
    tot_trans      = sum(s["in_transit"]     for s in iter_summaries)
    tot_unmet      = sum(s["dw_unmet"]       for s in iter_summaries)
    tot_unfilled_n = sum(s["dw_unfilled_n"]  for s in iter_summaries)
    tot_rate       = (tot_ful / tot_dem * 100) if tot_dem else 0

    return {
        "scenario":            scenario,
        "alpha":               alpha,
        "iterations":          iter_summaries,
        "dw_total_demand":     tot_dem,
        "dw_total_fulfilled":  tot_ful,
        "dw_total_in_transit": tot_trans,
        "dw_total_unmet":      tot_unmet,
        "dw_unfilled_commodities": tot_unfilled_n,
        "dw_overall_fill_rate": round(tot_rate, 2),
        "wall_time_s":         round(elapsed, 1),
    }


def print_summary_table(scheme: str, results: list[dict]) -> None:
    if not results:
        return
    print(f"\n{'='*100}")
    print(f"DEMAND AUGMENTATION ({scheme}) — YEAR={YEAR}, "
          f"PLANNING={PLANNING_WEEKS}w, DECISION={DECISION_WEEKS}w, "
          f"ITERATIONS={NUM_ITERATIONS}")
    print('='*100)
    header = (f"{'Scenario':>16}  {'alpha':>6}  {'DW Dem':>8}  {'DW Ful':>8}  "
              f"{'InTrans':>8}  {'Unmet':>7}  {'Unfilled':>9}  {'Fill%':>7}  {'Wall(s)':>8}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['scenario']:>16}  {r['alpha']:>+6.0%}  "
              f"{r['dw_total_demand']:>8,}  {r['dw_total_fulfilled']:>8,}  "
              f"{r['dw_total_in_transit']:>8,}  {r['dw_total_unmet']:>7,}  "
              f"{r['dw_unfilled_commodities']:>9,}  "
              f"{r['dw_overall_fill_rate']:>6.2f}%  "
              f"{r['wall_time_s']:>8.1f}")
    print('-' * len(header))


def save_results(all_results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{YEAR}_p{PLANNING_WEEKS}d{DECISION_WEEKS}"

    with open(RESULTS_DIR / f"results_{tag}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for scheme, results in all_results.items():
        for r in results:
            rows.append({"scheme": scheme,
                         **{k: v for k, v in r.items() if k != "iterations"}})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"summary_{tag}.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/ (results_{tag}.json, summary_{tag}.csv)")


def main():
    # Run each distinct scenario once (baseline is shared between schemes).
    cache: dict[str, dict] = {}

    def get(scenario: str, alpha: float) -> dict:
        if scenario not in cache:
            cache[scenario] = run_one(scenario, alpha)
        return cache[scenario]

    all_results = {
        scheme: [get(scn, a) for a, scn in scns]
        for scheme, scns in SCHEMES.items()
    }

    for scheme, results in all_results.items():
        print_summary_table(scheme, results)
    save_results(all_results)


if __name__ == "__main__":
    main()
