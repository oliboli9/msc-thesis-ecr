"""
analysis/stock-summary.py  —  Carrier container summary from an end-of-horizon stock file.

Prints, per port, how many Carrier containers are at or arriving at each port in the
end-of-horizon stock file, alongside the counts from Stock_01012023.csv for the same ports.

Usage:
    python analysis/stock-summary.py                          # default: results/stock/2023w1.csv
    python analysis/stock-summary.py results/stock/2026w1.csv
"""

import sys
from pathlib import Path
import pandas as pd

stock_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/stock/2023w1.csv")
baseline_path = Path("data/raw/Stock_01012023.csv")


def carrier_by_port(df: pd.DataFrame, empty_only: bool = False) -> tuple[pd.Series, pd.Series]:
    """Return (at_port, arriving) carrier counts keyed by port name.
    If empty_only=True, restrict to FullEmpty == Empty rows."""
    df = df.copy()
    df["Owner"] = df["Owner"].str.upper()
    c = df[df["Owner"] == "CARRIER"]
    if empty_only and "FullEmpty" in c.columns:
        c = c[c["FullEmpty"].str.lower() == "empty"]
    at_port = c[c["Location"] != "VESSEL"].groupby("Location")["count"].sum()
    arriving = c[c["Location"] == "VESSEL"].groupby("Next location")["count"].sum()
    return at_port, arriving


end_at, end_arr = carrier_by_port(pd.read_csv(stock_path))
bas_at, bas_arr = carrier_by_port(pd.read_csv(baseline_path))

# Only show ports that appear in the end-of-horizon stock
ports = sorted(set(end_at.index) | set(end_arr.index))

print(f"\nCarrier containers (empty + full) — {stock_path.name}  vs  {baseline_path.name} (same ports only)")
print(f"{'Port':<12}  {'End at':>9}  {'End arr':>9}  {'End tot':>9}  {'Base at':>9}  {'Base arr':>9}  {'Base tot':>9}")
print("-" * 78)

end_grand = base_grand = 0
for port in ports:
    ea = int(end_at.get(port, 0))
    er = int(end_arr.get(port, 0))
    et = ea + er
    ba = int(bas_at.get(port, 0))
    br = int(bas_arr.get(port, 0))
    bt = ba + br
    end_grand  += et
    base_grand += bt
    print(f"{port:<12}  {ea:>9,}  {er:>9,}  {et:>9,}  {ba:>9,}  {br:>9,}  {bt:>9,}")

print("-" * 78)
print(f"{'TOTAL':<12}  {int(end_at.sum()):>9,}  {int(end_arr.sum()):>9,}  {end_grand:>9,}  "
      f"{int(bas_at.reindex(ports).sum()):>9,}  {int(bas_arr.reindex(ports).sum()):>9,}  {base_grand:>9,}")
print()
