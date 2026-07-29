"""
analysis/print-network-coverage.py
Compare ports and vessels in the raw 2023 voyage data against eimskip_routes.py.
Prints what is covered and what is missing.
"""

import sys
import pandas as pd

sys.path.insert(0, "models")
from eimskip_routes import port_dict, vessel_lines

MAIN_LINES = ["Red Line", "Green Line", "Yellow Line", "Blue Line"]

voyages = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etaDateTime", "etdDateTime"])
v23 = voyages[voyages["etdDateTime"].dt.year == 2023].copy()
v23 = v23[v23["tradeRouteName"].isin(MAIN_LINES)]
v23["vessel"] = v23["voyage"].str[:3]

print("=== VESSELS per line in 2023 ===")
for line in MAIN_LINES:
    grp = v23[v23["tradeRouteName"] == line]
    vessels = sorted(grp["vessel"].dropna().unique())
    print(f"  {line}: {vessels}")

print()
print("=== PORTS per line in 2023 ===")
for line in MAIN_LINES:
    grp = v23[v23["tradeRouteName"] == line]
    ports = sorted(grp["portID"].dropna().unique())
    print(f"  {line}: {ports}")

print()
print("=== In eimskip_routes ===")
print(f"  Vessels: {sorted(vessel_lines.keys())}")
print(f"  Ports:   {sorted(port_dict.keys())}")

print()
print("=== Missing from vessel_lines ===")
all_vessels = set(v23["vessel"].dropna().unique())
missing_vessels = all_vessels - set(vessel_lines.keys())
if missing_vessels:
    for v in sorted(missing_vessels):
        lines = v23[v23["vessel"] == v]["tradeRouteName"].unique()
        print(f"  {v}: {list(lines)}")
else:
    print("  (none)")

print()
print("=== Missing from port_dict ===")
all_ports = set(v23["portID"].dropna().unique())
missing_ports = all_ports - set(port_dict.keys())
if missing_ports:
    for p in sorted(missing_ports):
        lines = v23[v23["portID"] == p]["tradeRouteName"].unique()
        print(f"  {p}: {list(lines)}")
else:
    print("  (none)")
