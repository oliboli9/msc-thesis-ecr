"""
port-summary.py  —  summarise model outputs for a given port.

Usage:
    python analysis/port-summary.py [PORT]

    PORT  three-letter LOCODE prefix, e.g. "CA HAL" (default).
          Matched case-insensitively against DepPort / ArrPort.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, "pipeline")
from config import EPOCH_HOUR_OFFSET, HORIZON_MONDAY, CALENDAR_WEEKS, COMMODITIES_PATH

# ── epoch helper ────────────────────────────────────────────────────────────

_v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
_mask = False
for _yr, _wk in CALENDAR_WEEKS:
    _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
_etds = _v[_mask]["etdDateTime"]
EPOCH_DT = (
    _etds.min().normalize().to_pydatetime() if not _etds.empty
    else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day)
)

def to_dt(h):
    return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

# ── config ───────────────────────────────────────────────────────────────────

PORT = (sys.argv[1] if len(sys.argv) > 1 else "CA HAL").upper()

# ── load data ────────────────────────────────────────────────────────────────

flows_df       = pd.read_csv("results/model_flows.csv")
inventory_df   = pd.read_csv("results/node_inventory.csv")
commodities_df = pd.read_csv(COMMODITIES_PATH)

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_empty(commodity: str) -> bool:
    return str(commodity).startswith("EMPTY")

def _container_spec(commodity: str) -> str:
    """Return e.g. 'Dry 40' from a commodity string."""
    parts = str(commodity).split("_")
    if len(parts) >= 3:
        return f"{parts[-2]} {parts[-1].replace('Carrier','').replace('Leased','').strip()}"
    return commodity

# ── 1. departing flows ────────────────────────────────────────────────────────

dep = flows_df[
    (flows_df["DepPort"].str.upper() == PORT) &
    (flows_df["DepPort"] != flows_df["ArrPort"])  # exclude wait arcs
].copy()

dep["IsEmpty"] = dep["Commodity"].apply(_is_empty)
dep["Time"]    = dep["DepHour"].apply(to_dt)

# ── 2. arriving flows ─────────────────────────────────────────────────────────

arr = flows_df[
    (flows_df["ArrPort"].str.upper() == PORT) &
    (flows_df["DepPort"] != flows_df["ArrPort"])
].copy()

arr["IsEmpty"] = arr["Commodity"].apply(_is_empty)
arr["Time"]    = arr["ArrHour"].apply(to_dt)

# ── 3. demand through port (origin or destination) ────────────────────────────

comm_orig = commodities_df[commodities_df["Origin"].str.upper()      == PORT].copy()
comm_dest = commodities_df[commodities_df["Destination"].str.upper() == PORT].copy()

# ── 4. container inventory at port ───────────────────────────────────────────

inv = (
    inventory_df[inventory_df["Port"].str.upper() == PORT]
    .groupby(["ContainerType", "ContainerSize", "Owner"])[["InitialContainers", "FinalContainers"]]
    .sum()
    .reset_index()
)

# ── printing helpers ──────────────────────────────────────────────────────────

SEP = "─" * 70

def print_header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ── report ────────────────────────────────────────────────────────────────────

print(f"\n{'═'*70}")
print(f"  PORT SUMMARY  —  {PORT}")
print(f"{'═'*70}")

# --- Departures ---
print_header("DEPARTURES")
if dep.empty:
    print("  (none)")
else:
    # Full-container departures by vessel / destination
    full_dep = dep[~dep["IsEmpty"]].copy()
    empty_dep = dep[dep["IsEmpty"]].copy()

    if not full_dep.empty:
        print("\n  Full containers:")
        summary = (
            full_dep.groupby(["Time", "Vessel", "ArrPort", "Line"])["Flow"]
            .sum()
            .reset_index()
            .rename(columns={"ArrPort": "To", "Flow": "Containers"})
            .sort_values("Time")
        )
        for _, r in summary.iterrows():
            print(f"    {r['Time']}  {r['Vessel']:6s}  {r['To']}  via {r['Line']}  — {int(r['Containers'])} ctrs")

    if not empty_dep.empty:
        print("\n  Empty repositioning:")
        summary = (
            empty_dep.groupby(["Time", "Vessel", "ArrPort"])["Flow"]
            .sum()
            .reset_index()
            .rename(columns={"ArrPort": "To", "Flow": "Containers"})
            .sort_values("Time")
        )
        for _, r in summary.iterrows():
            print(f"    {r['Time']}  {r['Vessel']:6s}  {r['To']}  — {int(r['Containers'])} empty")

# --- Arrivals ---
print_header("ARRIVALS")
if arr.empty:
    print("  (none)")
else:
    full_arr  = arr[~arr["IsEmpty"]].copy()
    empty_arr = arr[arr["IsEmpty"]].copy()

    if not full_arr.empty:
        print("\n  Full containers:")
        summary = (
            full_arr.groupby(["Time", "Vessel", "DepPort", "Line"])["Flow"]
            .sum()
            .reset_index()
            .rename(columns={"DepPort": "From", "Flow": "Containers"})
            .sort_values("Time")
        )
        for _, r in summary.iterrows():
            print(f"    {r['Time']}  {r['Vessel']:6s}  from {r['From']}  via {r['Line']}  — {int(r['Containers'])} ctrs")

    if not empty_arr.empty:
        print("\n  Empty repositioning:")
        summary = (
            empty_arr.groupby(["Time", "Vessel", "DepPort"])["Flow"]
            .sum()
            .reset_index()
            .rename(columns={"DepPort": "From", "Flow": "Containers"})
            .sort_values("Time")
        )
        for _, r in summary.iterrows():
            print(f"    {r['Time']}  {r['Vessel']:6s}  from {r['From']}  — {int(r['Containers'])} empty")

# --- Demand (origin) ---
print_header(f"DEMAND ORIGINATING AT {PORT}")
if comm_orig.empty:
    print("  (none)")
else:
    summary = (
        comm_orig.groupby(["Destination", "ContainerType", "ContainerSize", "Owner"])["Count"]
        .sum()
        .reset_index()
        .sort_values(["Destination", "ContainerType", "ContainerSize"])
    )
    summary["DepTime"] = comm_orig.groupby(["Destination", "ContainerType", "ContainerSize", "Owner"])["DepartureTime"].first().values
    summary["DepTime"] = summary["DepTime"].apply(to_dt)
    for _, r in summary.iterrows():
        print(f"    {r['DepTime']}  → {r['Destination']:8s}  {r['ContainerType']:6s} {r['ContainerSize']:2}ft  "
              f"{r['Owner']:8s}  {int(r['Count'])} ctrs")

# --- Demand (destination) ---
print_header(f"DEMAND DESTINED FOR {PORT}")
if comm_dest.empty:
    print("  (none)")
else:
    summary = (
        comm_dest.groupby(["Origin", "ContainerType", "ContainerSize", "Owner"])["Count"]
        .sum()
        .reset_index()
        .sort_values(["Origin", "ContainerType", "ContainerSize"])
    )
    summary["ArrTime"] = comm_dest.groupby(["Origin", "ContainerType", "ContainerSize", "Owner"])["ArrivalTime"].first().values
    summary["ArrTime"] = summary["ArrTime"].apply(to_dt)
    for _, r in summary.iterrows():
        print(f"    {r['ArrTime']}  {r['Origin']:8s} →  {r['ContainerType']:6s} {r['ContainerSize']:2}ft  "
              f"{r['Owner']:8s}  {int(r['Count'])} ctrs")

# --- Container inventory ---
print_header(f"CONTAINER INVENTORY AT {PORT}")
if inv.empty:
    print("  (none)")
else:
    inv_nonzero = inv[(inv["InitialContainers"] > 0) | (inv["FinalContainers"] > 0)]
    if inv_nonzero.empty:
        print("  (all zero)")
    else:
        print(f"  {'Type':8s}  {'Size':5s}  {'Owner':8s}  {'Initial':>8s}  {'Final':>8s}  {'Delta':>7s}")
        print("  " + "-"*54)
        for _, r in inv_nonzero.iterrows():
            delta = int(r["FinalContainers"]) - int(r["InitialContainers"])
            sign  = "+" if delta >= 0 else ""
            print(f"  {r['ContainerType']:8s}  {str(r['ContainerSize']):5s}  "
                  f"{r['Owner']:8s}  {int(r['InitialContainers']):8d}  "
                  f"{int(r['FinalContainers']):8d}  {sign}{delta:6d}")

# --- Totals ---
print_header("TOTALS")
total_dep_full  = dep[~dep["IsEmpty"]]["Flow"].sum()
total_dep_empty = dep[dep["IsEmpty"]]["Flow"].sum()
total_arr_full  = arr[~arr["IsEmpty"]]["Flow"].sum()
total_arr_empty = arr[arr["IsEmpty"]]["Flow"].sum()
total_demand = comm_orig["Count"].sum() + comm_dest["Count"].sum()

print(f"  Departures  — full: {int(total_dep_full):5d}  empty: {int(total_dep_empty):5d}")
print(f"  Arrivals    — full: {int(total_arr_full):5d}  empty: {int(total_arr_empty):5d}")
if total_demand > 0:
    print(f"  Demand      — {int(total_demand)} ctrs ({int(comm_orig['Count'].sum())} out, "
          f"{int(comm_dest['Count'].sum())} in)")
print()
