import sys
import importlib.util
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from framework.runner import (
    START_YEAR, START_WEEK, PLANNING_HORIZON_WEEKS,
    DECISION_HORIZON_WEEKS, NUM_ITERATIONS,
    USE_AUGMENTED_DEMAND, VERBOSE_STOCK, VERBOSE_UNFULFILLED,
    advance_by_weeks, build_config_params, build_calendar_weeks,
    build_port_vessel_dicts, build_arcs, build_commodities,
    load_stock_df, filter_stock_to_ports,
    process_stock_to_supply, build_carryover_commodities,
    extract_carryover_demands, extract_truncated_second_legs,
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

integrated_model = _load_module("integrated", "models/max-demand-fulfilment.py")
sequential_model = _load_module("sequential", "models/sequential-demand-then-reposition.py")


def run_rolling_horizon(solve_fn, model_name, voyages_df, demand_df, cap_df):
    """Run the rolling horizon with a given solve function. Returns per-iteration metrics."""

    metrics = []
    prev_carryover = {}
    prev_truncated_legs = {}

    out_dir = f"results/comparison/{model_name}"

    for i in range(NUM_ITERATIONS):
        year, week = advance_by_weeks(START_YEAR, START_WEEK, i * DECISION_HORIZON_WEEKS)
        label = f"{year}w{week}"

        effective_planning = min(PLANNING_HORIZON_WEEKS, NUM_ITERATIONS - i)
        effective_decision = min(DECISION_HORIZON_WEEKS, effective_planning)

        print(f"\n  [{model_name}] Iteration {i+1}/{NUM_ITERATIONS}: {year} W{week:02d}"
              f"  (planning {effective_planning}w, decision {effective_decision}w)")

        cfg = build_config_params(
            year, week, effective_planning, effective_decision, voyages_df
        )

        port_dict, vessel_lines = build_port_vessel_dicts(voyages_df, cfg["calendar_weeks"])
        arcs_df = build_arcs(voyages_df, cap_df, port_dict, cfg)
        commodities_df = build_commodities(demand_df, voyages_df, port_dict, vessel_lines, cfg)

        # Load stock: iteration 0 uses raw file, iteration 1+ uses own previous stock
        if i == 0:
            raw_stock = load_stock_df(0, START_YEAR, START_WEEK, DECISION_HORIZON_WEEKS)
        else:
            prev_year, prev_week = advance_by_weeks(
                START_YEAR, START_WEEK, (i - 1) * DECISION_HORIZON_WEEKS)
            raw_stock = pd.read_csv(f"{out_dir}/stock/{prev_year}w{prev_week}.csv")
        filtered_stk = filter_stock_to_ports(raw_stock, set(port_dict.keys()))

        if i > 0 and "FullEmpty" in filtered_stk.columns:
            stock_for_supply = filtered_stk[
                ~((filtered_stk["Location"] == "VESSEL") &
                  (filtered_stk["FullEmpty"].str.lower() == "full"))
            ].copy()
        else:
            stock_for_supply = filtered_stk

        initial_sup, vessel_late_rows = process_stock_to_supply(
            stock_for_supply, arcs_df, cfg["stock_cutoff_hour"], cfg["epoch_dt"],
            voyages_df, verbose=VERBOSE_STOCK
        )

        if i > 0 and prev_carryover:
            carryover_df, carryover_sup = build_carryover_commodities(
                prev_carryover, arcs_df, year, week)
            if not carryover_df.empty:
                commodities_df = pd.concat([carryover_df, commodities_df], ignore_index=True)
            for key, count in carryover_sup.items():
                initial_sup[key] = initial_sup.get(key, 0) + count

        if i > 0 and prev_truncated_legs:
            leg_df, _ = build_carryover_commodities(prev_truncated_legs, arcs_df, year, week)
            if not leg_df.empty:
                leg_df["Commodity"] = leg_df["Commodity"].str.replace("CARRY_", "TLEG_", n=1)
                commodities_df = pd.concat([leg_df, commodities_df], ignore_index=True)

        results = solve_fn(arcs_df, commodities_df, initial_sup, cfg)

        if results.status != "OPTIMAL":
            print(f"    FAILED: {results.status}")
            break

        prev_carryover = extract_carryover_demands(results, commodities_df, cfg)
        prev_truncated_legs = extract_truncated_second_legs(results, commodities_df)

        # Save results to model-specific directory
        Path(f"{out_dir}/model-flows").mkdir(parents=True, exist_ok=True)
        Path(f"{out_dir}/stock").mkdir(parents=True, exist_ok=True)
        Path(f"{out_dir}/demand-fulfilment").mkdir(parents=True, exist_ok=True)

        # Convert ArrivalTime from epoch hours to datetime
        stock_out = results.stock_df.copy()
        if "ArrivalTime" in stock_out.columns:
            has_at = stock_out["ArrivalTime"].apply(
                lambda h: pd.notna(h) and str(h).strip() not in ("", "nan"))
            stock_out.loc[has_at, "ArrivalTime"] = stock_out.loc[has_at, "ArrivalTime"].apply(
                lambda h: (cfg["epoch_dt"] + timedelta(hours=int(float(h)))).isoformat()
            )
        # Append vessel_late_rows
        if vessel_late_rows:
            late_df = pd.DataFrame(vessel_late_rows)
            if "ArrivalTime" in late_df.columns:
                late_df["ArrivalTime"] = late_df["ArrivalTime"].apply(
                    lambda h: (cfg["epoch_dt"] + timedelta(hours=int(float(h)))).isoformat()
                    if pd.notna(h) and str(h).strip() != "" else ""
                )
            stock_out = pd.concat([stock_out, late_df], ignore_index=True)

        results.flows_df.to_csv(f"{out_dir}/model-flows/{label}.csv", index=False)
        results.fulfillment_df.to_csv(f"{out_dir}/demand-fulfilment/{label}.csv", index=False)
        stock_out.to_csv(f"{out_dir}/stock/{label}.csv", index=False)

        # Collect metrics (original demands only)
        ful = results.fulfillment_df
        orig = ful[~ful["Commodity"].str.startswith(("CARRY_", "TLEG_"))]
        total_demand = int(orig["Demand"].sum())
        total_fulfilled = int(orig["Fulfilled"].sum())

        # Empty repositioning moves (sailing arcs only)
        empty_flows = results.flows_df[
            results.flows_df["Commodity"].str.startswith("EMPTY_") &
            (results.flows_df["DepPort"] != results.flows_df["ArrPort"])
        ]
        empty_moves = int(empty_flows["Flow"].sum()) if not empty_flows.empty else 0

        metrics.append({
            "iteration": label,
            "demand": total_demand,
            "fulfilled": total_fulfilled,
            "unfulfilled": total_demand - total_fulfilled,
            "fill_rate": total_fulfilled / total_demand if total_demand > 0 else 0,
            "empty_moves": empty_moves,
        })

    return metrics


def main():
    print("Loading data...")
    voyages_df = pd.read_csv("data/raw/Eimskip_voyages.csv",
                             parse_dates=["etaDateTime", "etdDateTime"])
    demand_path = ("data/augmented/Eimskip_data_final.csv" if USE_AUGMENTED_DEMAND
                    else "data/clean/Eimskip_data_final.csv")
    demand_df = pd.read_csv(demand_path)
    print(f"Demand source: {demand_path}")
    cap_df = pd.read_csv("data/raw/vessel-capacities.csv", sep=r"\s+")

    # Run integrated model
    print(f"\n{'='*60}")
    print("INTEGRATED MODEL (joint demand + repositioning)")
    print(f"{'='*60}")
    integrated_metrics = run_rolling_horizon(
        integrated_model.solve, "integrated", voyages_df, demand_df, cap_df)

    # Run sequential model
    print(f"\n{'='*60}")
    print("SEQUENTIAL MODEL (demand first, then repositioning)")
    print(f"{'='*60}")
    sequential_metrics = run_rolling_horizon(
        sequential_model.solve, "sequential", voyages_df, demand_df, cap_df)

    # ── Save metrics ──────────────────────────────────────────────────────────

    int_df = pd.DataFrame(integrated_metrics)
    int_df["model"] = "integrated"
    seq_df = pd.DataFrame(sequential_metrics)
    seq_df["model"] = "sequential"

    metrics_df = pd.concat([int_df, seq_df], ignore_index=True)
    out_path = "results/comparison/metrics.csv"
    Path("results/comparison").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    for name, df in [("Integrated", int_df), ("Sequential", seq_df)]:
        print(f"\n--- {name} Model ---")
        print(df.drop(columns="model").to_string(index=False))
        print(f"\nTotal: {df['fulfilled'].sum()}/{df['demand'].sum()} "
              f"({df['fulfilled'].sum()/df['demand'].sum():.1%}), "
              f"empty moves: {df['empty_moves'].sum()}")

    print("\n--- Difference (Integrated - Sequential) ---")
    diff = pd.DataFrame({
        "iteration": int_df["iteration"].values,
        "fulfilled_diff": int_df["fulfilled"].values - seq_df["fulfilled"].values,
        "fill_rate_diff_pp": (int_df["fill_rate"].values - seq_df["fill_rate"].values) * 100,
        "empty_moves_diff": int_df["empty_moves"].values - seq_df["empty_moves"].values,
    })
    print(diff.to_string(index=False))

    total_int = int_df["fulfilled"].sum()
    total_seq = seq_df["fulfilled"].sum()
    total_demand = int_df["demand"].sum()
    print(f"\nOverall improvement: {total_int - total_seq:+d} containers "
          f"({(total_int - total_seq) / total_demand * 100:+.2f} pp)")

    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
