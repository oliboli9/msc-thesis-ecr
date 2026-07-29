import sys
import time
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from framework import runner

# ── Experiment parameters ────────────────────────────────────────────────────

HORIZONS         = [4]   # planning-horizon lengths (weeks) to sweep
YEARS            = [2025]  # instance years to sweep (each runs the full HORIZONS list)
START_WEEK       = 1
DECISION_WEEKS   = 1                 # decision horizon (fixed across experiments)
NUM_ITERATIONS   = 10                 # rolling-horizon iterations per experiment
SCENARIO         = "baseline"        # "baseline" | "light" | "moderate" | "heavy"

RESULTS_DIR = Path("results/horizon-sensitivity")


def run_one(h: int, year: int) -> dict:
    """Override runner params, invoke runner.main(), return aggregate metrics."""
    runner.START_YEAR             = year
    runner.START_WEEK             = START_WEEK
    runner.PLANNING_HORIZON_WEEKS = h
    runner.DECISION_HORIZON_WEEKS = DECISION_WEEKS
    runner.NUM_ITERATIONS         = NUM_ITERATIONS
    runner.SCENARIO               = SCENARIO
    runner.TRUNCATE_HORIZON_AT_END = True

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: year={year}, planning horizon={h} weeks")
    print(f"{'#'*70}")

    t0 = time.time()
    iter_summaries = runner.main()
    elapsed = time.time() - t0

    iter_summaries = iter_summaries or []
    tot_dem    = sum(s["dw_demand"]     for s in iter_summaries)
    tot_ful    = sum(s["dw_fulfilled"]  for s in iter_summaries)
    tot_trans  = sum(s["in_transit"]    for s in iter_summaries)
    tot_unmet  = sum(s["dw_unmet"]      for s in iter_summaries)
    tot_unfilled_n = sum(s["dw_unfilled_n"] for s in iter_summaries)
    tot_rate   = (tot_ful / tot_dem * 100) if tot_dem else 0

    return {
        "year":                year,
        "planning_weeks":      h,
        "iterations":          iter_summaries,
        "dw_total_demand":     tot_dem,
        "dw_total_fulfilled":  tot_ful,
        "dw_total_in_transit": tot_trans,
        "dw_total_unmet":      tot_unmet,
        "dw_unfilled_commodities": tot_unfilled_n,
        "dw_overall_fill_rate": round(tot_rate, 2),
        "wall_time_s":         round(elapsed, 1),
    }


def print_summary_table(results: list[dict]) -> None:
    if not results:
        return
    print(f"\n{'='*100}")
    print(f"PLANNING HORIZON SENSITIVITY — SCENARIO={SCENARIO}, "
          f"DECISION={DECISION_WEEKS}w, ITERATIONS={NUM_ITERATIONS}")
    print('='*100)
    header = (f"{'Year':>6}  {'H(weeks)':>9}  {'DW Dem':>8}  {'DW Ful':>8}  "
              f"{'InTrans':>8}  {'Unmet':>7}  {'Unfilled':>9}  {'Fill%':>7}  {'Wall(s)':>8}")
    print(header)
    print('-' * len(header))
    by_year: dict[int, list[dict]] = {}
    for r in results:
        by_year.setdefault(r["year"], []).append(r)
    for rows in by_year.values():
        best = max(rows, key=lambda r: r["dw_overall_fill_rate"])
        for r in rows:
            marker = "  <-- best" if r is best else ""
            print(f"{r['year']:>6}  {r['planning_weeks']:>9}  "
                  f"{r['dw_total_demand']:>8,}  {r['dw_total_fulfilled']:>8,}  "
                  f"{r['dw_total_in_transit']:>8,}  {r['dw_total_unmet']:>7,}  "
                  f"{r['dw_unfilled_commodities']:>9,}  "
                  f"{r['dw_overall_fill_rate']:>6.2f}%  "
                  f"{r['wall_time_s']:>8.1f}{marker}")
        print('-' * len(header))


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    years_tag = "-".join(str(y) for y in sorted({r["year"] for r in results}))
    tag = f"{years_tag}_{SCENARIO}"

    # JSON: full per-iteration detail
    with open(RESULTS_DIR / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV summary: one row per horizon
    summary_rows = [
        {k: v for k, v in r.items() if k != "iterations"}
        for r in results
    ]
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / f"summary_{tag}.csv", index=False
    )

    # CSV per-iteration: one row per (year, horizon, iteration)
    per_iter_rows = []
    for r in results:
        for it in r["iterations"]:
            per_iter_rows.append({"year": r["year"],
                                  "planning_weeks": r["planning_weeks"],
                                  **it})
    if per_iter_rows:
        pd.DataFrame(per_iter_rows).to_csv(
            RESULTS_DIR / f"per_iteration_{tag}.csv", index=False
        )

    print(f"\nResults saved to {RESULTS_DIR}/")


def main():
    results = [run_one(h, y) for y in YEARS for h in HORIZONS]
    print_summary_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
