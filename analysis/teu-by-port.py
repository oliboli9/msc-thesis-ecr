"""
teu-by-port.py  —  total TEUs sent from a port for a given year/week.

Usage:
    python analysis/teu-by-port.py [PORT] [YEAR] [WEEK]

Defaults: NL RTM, 2026, 1
"""

import sys
import pandas as pd

PORT = sys.argv[1].upper() if len(sys.argv) > 1 else "NL RTM"
YEAR = int(sys.argv[2])    if len(sys.argv) > 2 else 2026
WEEK = int(sys.argv[3])    if len(sys.argv) > 3 else 1

TEU_MAP = {20: 1, 40: 2, 45: 2}

df = pd.read_csv("data/clean/Eimskip_data_final.csv")

subset = df[
    (df["Year"] == YEAR) &
    (df["Week"] == WEEK) &
    (df["Load 1"].str.upper() == PORT) &
    (df["Full/Empty"] == "F") &
    (df["ContainerSize"].isin(TEU_MAP.keys()))
].copy()

subset["TEUs"] = subset["ContainerSize"].map(TEU_MAP) * subset["Total"]

print(f"\nTEUs sent from {PORT}  —  Year {YEAR}, Week {WEEK}")
print(f"{'─'*45}")
print(f"  Total containers : {int(subset['Total'].sum())}")
print(f"  Total TEUs       : {int(subset['TEUs'].sum())}")

if not subset.empty:
    print(f"\n  By container size:")
    by_size = (
        subset.groupby("ContainerSize")[["Total", "TEUs"]]
        .sum()
        .reset_index()
        .sort_values("ContainerSize")
    )
    for _, r in by_size.iterrows():
        print(f"    {int(r['ContainerSize']):2}ft  {int(r['Total']):4} ctrs  =  {int(r['TEUs']):4} TEUs")

    print(f"\n  By destination:")
    by_dest = (
        subset.groupby("Discharge 1")[["Total", "TEUs"]]
        .sum()
        .reset_index()
        .sort_values("TEUs", ascending=False)
        .rename(columns={"Discharge 1": "Destination"})
    )
    for _, r in by_dest.iterrows():
        print(f"    {r['Destination']:10s}  {int(r['Total']):4} ctrs  =  {int(r['TEUs']):4} TEUs")

    print(f"\n  By vessel (Voyage 1):")
    subset["Vessel"] = subset["Voyage 1"].apply(
        lambda v: str(v).strip()[:3].upper() if pd.notna(v) else "?"
    )
    by_vessel = (
        subset.groupby(["Vessel", "Voyage 1"])[["Total", "TEUs"]]
        .sum()
        .reset_index()
        .sort_values("TEUs", ascending=False)
    )
    for _, r in by_vessel.iterrows():
        print(f"    {r['Vessel']}  ({r['Voyage 1']})  {int(r['Total']):4} ctrs  =  {int(r['TEUs']):4} TEUs")
print()
