
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "data")

from framework.planning_horizon_sensitivity import (
    run_experiment, build_config_params, build_port_vessel_dicts,
    build_commodities, advance_by_weeks,
    HORIZONS, START_YEAR, START_WEEK, DECISION_WEEKS, NUM_ITERATIONS,
    SCENARIO,
)

SEEDS = [0, 1, 2, 3, 4]   # Gurobi seeds to sweep; 5 is enough to see scatter

OUT_DIR = Path("results/horizon-sensitivity")


def main():
    print(f"Loading data (scenario={SCENARIO})...")
    voyages_df = pd.read_csv(
        "data/clean/eimskip_voyages.csv",
        parse_dates=["etaDateTime", "etdDateTime"]
    )
    demand_df = pd.read_csv(f"data/augmented/{SCENARIO}/Eimskip_data_final.csv")
    cap_df    = pd.read_csv("data/raw/vessel-capacities.csv", sep=r"\s+")

    # Pre-build shared (widest-horizon) network and reference DW demand,
    # exactly as planning_horizon_sensitivity.main() does.
    max_h = max(HORIZONS)
    shared_network = []
    for i in range(NUM_ITERATIONS):
        year, week = advance_by_weeks(START_YEAR, START_WEEK, i * DECISION_WEEKS)
        wide_cfg = build_config_params(year, week, max_h, DECISION_WEEKS, voyages_df)
        port_dict, vessel_lines = build_port_vessel_dicts(voyages_df, wide_cfg["calendar_weeks"])
        shared_network.append((port_dict, vessel_lines, wide_cfg["calendar_weeks"]))

    ref_dw_demands = []
    for i in range(NUM_ITERATIONS):
        year, week = advance_by_weeks(START_YEAR, START_WEEK, i * DECISION_WEEKS)
        port_dict, vessel_lines, _ = shared_network[i]
        ref_cfg = build_config_params(year, week, max_h, DECISION_WEEKS, voyages_df)
        ref_comms = build_commodities(demand_df, voyages_df, port_dict, vessel_lines, ref_cfg)
        cutoff = DECISION_WEEKS * 168
        ref_dw = ref_comms[
            (ref_comms["DepartureTime"] < cutoff) &
            ~ref_comms["Commodity"].str.startswith(("CARRY_", "TLEG_"))
        ]
        ref_dw_demands.append(int(ref_dw["Count"].sum()))

    print(f"Reference DW demand per iteration: {ref_dw_demands} "
          f"(total: {sum(ref_dw_demands)})\n")

    # ── Sweep ────────────────────────────────────────────────────────────────
    rows = []
    for seed in SEEDS:
        for h in HORIZONS:
            print(f"#### seed={seed}  H={h}w ####")
            r = run_experiment(h, voyages_df, demand_df, cap_df,
                               shared_network, ref_dw_demands, seed=seed)
            if r["status"] != "OK":
                print(f"  FAILED")
                continue
            print(f"  => DW fill: {r['dw_overall_fill_rate']:.2f}%  "
                  f"unfulfilled: {r['dw_total_unfulfilled']}  "
                  f"({r['total_solve_time_s']:.1f}s)")
            rows.append({
                "seed":              seed,
                "horizon":           h,
                "dw_demand":         r["dw_total_demand"],
                "dw_fulfilled":      r["dw_total_fulfilled"],
                "dw_unfulfilled":    r["dw_total_unfulfilled"],
                "dw_fill_rate":      r["dw_overall_fill_rate"],
                "fh_fill_rate":      r["overall_fill_rate"],
                "solve_time_s":      r["total_solve_time_s"],
            })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"seed_sweep_{START_YEAR}_{SCENARIO}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # ── Summary tables ───────────────────────────────────────────────────────
    print("\n=== Per-seed × horizon: dw_fulfilled ===")
    pivot = df.pivot(index="seed", columns="horizon", values="dw_fulfilled")
    print(pivot.to_string())

    print("\n=== Per-horizon stats across seeds ===")
    stats = df.groupby("horizon")["dw_fulfilled"].agg(["min", "mean", "max", "std"])
    print(stats.to_string())

    print("\n=== Best horizon per seed ===")
    best = df.loc[df.groupby("seed")["dw_fulfilled"].idxmax()][["seed", "horizon", "dw_fulfilled"]]
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
