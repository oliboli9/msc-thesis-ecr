import sys
import re
import argparse
import subprocess
import shutil
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS_BASE = ROOT / "results" / "rolling"
INITIAL_SUPPLY_PATH = ROOT / "data" / "processed" / "initial_supply.csv"


def read_config() -> tuple[int, int, int]:
    """Return current (START_YEAR, START_WEEK, NUM_WEEKS) from config.py."""
    text = (ROOT / "pipeline" / "config.py").read_text()
    year  = int(re.search(r"^START_YEAR\s*=\s*(\d+)", text, re.MULTILINE).group(1))
    week  = int(re.search(r"^START_WEEK\s*=\s*(\d+)", text, re.MULTILINE).group(1))
    nwks  = int(re.search(r"^NUM_WEEKS\s*=\s*(\d+)",  text, re.MULTILINE).group(1))
    return year, week, nwks


def set_config_week(year: int, week: int, num_weeks: int = 1):
    """Update START_YEAR / START_WEEK / NUM_WEEKS in pipeline/config.py."""
    path = ROOT / "pipeline" / "config.py"
    text = path.read_text()
    text = re.sub(r"^START_YEAR\s*=\s*\d+", f"START_YEAR = {year}",       text, flags=re.MULTILINE)
    text = re.sub(r"^START_WEEK\s*=\s*\d+", f"START_WEEK = {week}",       text, flags=re.MULTILINE)
    text = re.sub(r"^NUM_WEEKS\s*=\s*\d+",  f"NUM_WEEKS  = {num_weeks}",  text, flags=re.MULTILINE)
    path.write_text(text)



def run_pipeline():
    """Run voyage-based arc and commodity construction for the current config week."""
    subprocess.run(
        [sys.executable, "pipeline/2_construct_arcs_voyages.py"],
        check=True, cwd=ROOT
    )
    subprocess.run(
        [sys.executable, "pipeline/3_construct_commodities_voyages.py"],
        check=True, cwd=ROOT
    )


def run_model():
    """Solve the optimisation model for the current config week."""
    subprocess.run(
        [sys.executable, "models/single-horizon.py"],
        check=True, cwd=ROOT
    )


def save_week_results(year: int, week: int):
    """Copy current results and processed inputs to a week-specific directory."""
    week_dir = RESULTS_BASE / f"{year}_w{week:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "model_flows.csv", "node_inventory.csv",
        "demand_fulfillment.csv", "unfulfilled_demand.csv",
    ]:
        src = ROOT / "results" / fname
        if src.exists():
            shutil.copy(src, week_dir / fname)
    for fname in ["time_space_arcs.csv", "commodities.csv"]:
        src = ROOT / "data" / "processed" / fname
        if src.exists():
            shutil.copy(src, week_dir / fname)
    stock_src_dir = ROOT / "results" / "stock"
    if stock_src_dir.exists():
        for stock_file in stock_src_dir.glob("*.csv"):
            (week_dir / "stock").mkdir(exist_ok=True)
            shutil.copy(stock_file, week_dir / "stock" / stock_file.name)


def _load_transit_data(week_dir: Path, H: int) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    flows_path = week_dir / "model_flows.csv"
    comms_path = week_dir / "commodities.csv"
    if not flows_path.exists():
        return None, pd.DataFrame()
    flows = pd.read_csv(flows_path)
    in_transit = flows[flows["ArrHour"] >= H].copy()
    if in_transit.empty:
        return None, pd.DataFrame()
    comm_df = pd.read_csv(comms_path) if comms_path.exists() else pd.DataFrame()
    return in_transit, comm_df


def _parse_spec(commodity: str) -> tuple[str, float, str] | None:
    parts = commodity.split("_")
    try:
        return parts[1], float(parts[2]), parts[3]
    except (IndexError, ValueError):
        return None


def extract_end_inventory(H: int | None = None) -> pd.DataFrame:
    inv = pd.read_csv(ROOT / "results" / "node_inventory.csv")
    if H is not None:
        inv = inv[inv["Time"] <= H]
    latest = inv.groupby("Port")["Time"].max().reset_index().rename(columns={"Time": "MaxTime"})
    inv = inv.merge(latest, on="Port")
    end = inv[inv["Time"] == inv["MaxTime"]].copy()
    end = (
        end[end["FinalContainers"] > 0]
        [["Port", "ContainerType", "ContainerSize", "Owner", "FinalContainers"]]
        .rename(columns={"FinalContainers": "Count"})
        .reset_index(drop=True)
    )
    return end


def extract_in_transit_supply(week_dir: Path, H: int = 168) -> pd.DataFrame:
    in_transit, comm_df = _load_transit_data(week_dir, H)
    empty = pd.DataFrame(columns=["Port", "ContainerType", "ContainerSize", "Owner", "Count"])
    if in_transit is None:
        return empty

    records = []
    for _, row in in_transit.iterrows():
        comm  = str(row["Commodity"])
        spec  = _parse_spec(comm)
        if spec is None:
            continue
        ct, cs, eo = spec
        port  = row["ArrPort"]
        count = round(float(row["Flow"]))
        if count < 1:
            continue

        if comm.startswith("EMPTY_"):
            records.append((port, ct, cs, eo, count))
        else:
            # Cargo: only add supply if this is the final destination
            idx = int(comm.split("_")[0][1:])
            try:
                orig_dest = comm_df.iloc[idx]["Destination"]
            except (IndexError, KeyError):
                orig_dest = None
            if orig_dest == port and eo == "Carrier":
                records.append((port, ct, cs, "Carrier", count))

    if not records:
        return empty
    df = pd.DataFrame(records, columns=["Port", "ContainerType", "ContainerSize", "Owner", "Count"])
    return df.groupby(["Port", "ContainerType", "ContainerSize", "Owner"])["Count"].sum().reset_index()


def extract_carryover_demands(week_dir: Path, H: int = 168) -> pd.DataFrame:

    in_transit, comm_df = _load_transit_data(week_dir, H)
    cols = ["ArrPort", "OrigDest", "ContainerType", "ContainerSize", "Owner", "Count", "OrigCommodityID"]
    empty = pd.DataFrame(columns=cols)
    if in_transit is None:
        return empty

    records = []
    for _, row in in_transit.iterrows():
        comm = str(row["Commodity"])
        if comm.startswith("EMPTY_"):
            continue
        spec = _parse_spec(comm)
        if spec is None:
            continue
        ct, cs, eo = spec
        arr_port = row["ArrPort"]
        count    = round(float(row["Flow"]))
        if count < 1:
            continue

        idx = int(comm.split("_")[0][1:])
        try:
            orig_dest = comm_df.iloc[idx]["Destination"]
        except (IndexError, KeyError):
            continue

        if pd.isna(orig_dest) or orig_dest == arr_port:
            continue  # delivered on arrival — handled by extract_in_transit_supply

        records.append((arr_port, orig_dest, ct, cs, eo, count, comm))

    if not records:
        return empty
    df = pd.DataFrame(records, columns=cols)
    return (
        df.groupby(["ArrPort", "OrigDest", "ContainerType", "ContainerSize", "Owner", "OrigCommodityID"])
        ["Count"].sum().reset_index()
    )


def write_initial_supply(inv_df: pd.DataFrame):
    inv_df.to_csv(INITIAL_SUPPLY_PATH, index=False)


def clear_initial_supply():
    if INITIAL_SUPPLY_PATH.exists():
        INITIAL_SUPPLY_PATH.unlink()


# ── Week enumeration ───────────────────────────────────────────────────────────

def available_weeks(
    start_year: int, start_week: int,
    end_year: int | None, end_week: int | None,
) -> list[tuple[int, int]]:
    df = pd.read_csv(
        ROOT / "data" / "raw" / "fixedEimskip_data_final.csv",
        usecols=["Year", "Week"],
    )
    pairs = sorted(set(zip(df["Year"], df["Week"])))
    pairs = [(y, w) for y, w in pairs if (y, w) >= (start_year, start_week)]
    if end_year is not None and end_week is not None:
        pairs = [(y, w) for y, w in pairs if (y, w) <= (end_year, end_week)]
    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rolling horizon container optimisation")
    parser.add_argument("--start-year",       type=int, default=2026)
    parser.add_argument("--start-week",       type=int, default=1)
    parser.add_argument("--planning-horizon", type=int, default=4,
                        help="Weeks in each solve window (look-ahead). Default: 4")
    parser.add_argument("--decision-horizon", type=int, default=1,
                        help="Weeks committed per iteration (advance step). Default: 1")
    parser.add_argument("--iterations",       type=int, default=2,
                        help="Number of iterations. Default: 2")
    args = parser.parse_args()

    P   = args.planning_horizon
    D   = args.decision_horizon
    N   = args.iterations
    H_D = D * 7 * 24   # decision horizon in hours (commit cutoff)

    all_weeks = available_weeks(args.start_year, args.start_week, None, None)
    if not all_weeks:
        print("No weeks found in data for the specified range.")
        return

    print(f"Rolling horizon: P={P}w planning, D={D}w decision, {N} iterations  "
          f"(start {args.start_year} W{args.start_week:02d})")
    print()

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    original_config = read_config()
    summary_rows = []

    try:
        prev_inventory: pd.DataFrame | None = None
        start_idx = 0

        for i in range(N):
            if start_idx >= len(all_weeks):
                print("  No more weeks in data — stopping early.")
                break

            year, week = all_weeks[start_idx]
            end_dec_year,  end_dec_week  = all_weeks[min(start_idx + D - 1, len(all_weeks) - 1)]
            end_plan_year, end_plan_week = all_weeks[min(start_idx + P - 1, len(all_weeks) - 1)]

            print(f"{'='*60}")
            print(f"  Iteration {i+1}/{N}:  "
                  f"Decide {year} W{week:02d}–{end_dec_year} W{end_dec_week:02d}  |  "
                  f"Plan look-ahead → {end_plan_year} W{end_plan_week:02d}")
            print(f"{'='*60}")

            # 1. Update pipeline config
            set_config_week(year, week, num_weeks=P)

            # 2. Build arc and commodity data for this week
            print("  Running pipeline...")
            run_pipeline()

            # 3. Write (or clear) initial supply for the model
            if prev_inventory is None or prev_inventory.empty:
                clear_initial_supply()
                print("  Initial supply: uniform (UNIFORM_SUPPLY / IS_REY_SUPPLY in model)")
            else:
                write_initial_supply(prev_inventory)
                total = int(prev_inventory["Count"].sum())
                ports = prev_inventory["Port"].nunique()
                print(f"  Initial supply: {total} containers across {ports} ports "
                      f"(carried from previous week)")

            # 4. Solve
            print("  Solving...")
            run_model()

            # 5. Archive results
            week_dir = RESULTS_BASE / f"{year}_w{week:02d}"
            save_week_results(year, week)
            print(f"  Saved to results/rolling/{year}_w{week:02d}/")

            # 6. Build next iteration's opening stock (commit only the decision horizon)
            end_inv        = extract_end_inventory(H=H_D)
            transit_supply = extract_in_transit_supply(week_dir, H=H_D)
            carryover      = extract_carryover_demands(week_dir, H=H_D)

     
            carryover_path = ROOT / "data" / "processed" / "carryover_demands.csv"
            if not carryover.empty:
                carryover.to_csv(carryover_path, index=False)
            elif carryover_path.exists():
                carryover_path.unlink()

            # Port inventory + containers delivered in transit = available empties for week N+1
            combined = pd.concat([end_inv, transit_supply], ignore_index=True)
            if not combined.empty:
                prev_inventory = (
                    combined
                    .groupby(["Port", "ContainerType", "ContainerSize", "Owner"])["Count"]
                    .sum()
                    .reset_index()
                )
            else:
                prev_inventory = combined

            # 7. Record summary
            unf_path = ROOT / "results" / "unfulfilled_demand.csv"
            if unf_path.exists():
                unf_df = pd.read_csv(unf_path)
                n_comm  = len(unf_df)
                n_ctrs  = int(unf_df["Unfulfilled"].sum()) if "Unfulfilled" in unf_df.columns else 0
            else:
                n_comm = n_ctrs = 0

            at_port      = int(end_inv["Count"].sum())         if not end_inv.empty        else 0
            transit_sup  = int(transit_supply["Count"].sum())  if not transit_supply.empty else 0
            n_carry      = int(carryover["Count"].sum())        if not carryover.empty      else 0
            carry_ports  = prev_inventory["Port"].nunique()     if not prev_inventory.empty else 0

            summary_rows.append({
                "Year":                   year,
                "Week":                   week,
                "UnfulfilledCommodities": n_comm,
                "UnfulfilledContainers":  n_ctrs,
                "AtPortNextWeek":         at_port,
                "DeliveredInTransit":     transit_sup,
                "CarryoverDemands":       n_carry,
                "CarriedForwardPorts":    carry_ports,
            })

            print(f"  Unfulfilled: {n_comm} commodities / {n_ctrs} containers  |  "
                  f"At port: {at_port}  |  Delivered in transit: {transit_sup}  |  "
                  f"Carry-over (en route): {n_carry} containers")
            print()

            start_idx += D

    finally:
  
        set_config_week(*original_config)
        clear_initial_supply()
        for tmp in ["carryover_demands.csv"]:
            p = ROOT / "data" / "processed" / tmp
            if p.exists():
                p.unlink()


    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_BASE / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nRolling horizon complete.")
    print(f"Summary saved to results/rolling/summary.csv\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
