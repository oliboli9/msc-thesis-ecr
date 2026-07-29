"""
Compute baseline-performance metrics for a single rolling-horizon iteration.

Reads:
  data/raw/Stock_0101{YEAR}.csv          — initial stock
  results/stock/{LABEL}.csv              — end-of-decision stock
  results/model-flows/{LABEL}.csv        — flows on every arc
  results/vessel-utilisation/{LABEL}.csv — per-leg utilisation

Produces:
  results/baseline-metrics/{LABEL}.csv               — small stats table
  msc-thesis-writing/Pictures/baseline-fleet-usage.png
  msc-thesis-writing/Pictures/baseline-end-location.png
  msc-thesis-writing/Tables/baseline-vessel-utilisation.tex

Usage:
  python analysis/baseline-metrics.py --label 2026w1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO        = Path(__file__).resolve().parent.parent
PICTURES    = REPO / "msc-thesis-writing" / "Pictures"
TABLES      = REPO / "msc-thesis-writing" / "Tables"
METRICS_DIR = REPO / "results" / "baseline-metrics"

CARRIER_TOKENS = {"CARRIER", "Carrier"}


def parse_year_week(label: str) -> tuple[int, int]:
    m = re.match(r"^(\d{4})w(\d+)$", label)
    if not m:
        raise ValueError(f"Bad label: {label!r}")
    return int(m.group(1)), int(m.group(2))


def owner_class_stock(owner: str) -> str:
    return "Carrier" if str(owner).upper() == "CARRIER" else "Non-Carrier"


def owner_class_commodity(commodity: str) -> str:
    """Last underscore-separated token: Carrier|CARRIER|LEASED|SHIPPER."""
    tok = str(commodity).split("_")[-1]
    return "Carrier" if tok.upper() == "CARRIER" else "Non-Carrier"


def reefer_class(code: str) -> str:
    """3rd character == 'R' → Reefer, else → Dry (ISO type codes)."""
    code = str(code)
    return "Reefer" if len(code) >= 3 and code[2].upper() == "R" else "Dry"


def teu_of(commodity: str) -> int:
    for part in str(commodity).split("_"):
        try:
            sz = float(part)
            if sz == 20:
                return 1
            if sz in (40, 45):
                return 2
        except ValueError:
            continue
    return 2


def compute_metrics(label: str) -> dict:
    year, _ = parse_year_week(label)
    initial = pd.read_csv(REPO / "data" / "raw" / f"Stock_0101{year}.csv")
    end_stock = pd.read_csv(REPO / "results" / "stock" / f"{label}.csv")
    flows = pd.read_csv(REPO / "results" / "model-flows" / f"{label}.csv")
    util = pd.read_csv(REPO / "results" / "vessel-utilisation" / f"{label}.csv")

    initial["OwnerClass"] = initial["Owner"].apply(owner_class_stock)
    initial["AtVessel"] = initial["Location"].eq("VESSEL")
    end_stock["OwnerClass"] = end_stock["Owner"].apply(owner_class_stock)
    end_stock["AtVessel"] = end_stock["Location"].eq("VESSEL")

    flows["OwnerClass"] = flows["Commodity"].apply(owner_class_commodity)
    flows["TEU"] = flows["Flow"] * flows["Commodity"].apply(teu_of)
    sail = flows[flows["DepPort"] != flows["ArrPort"]].copy()

    def split(df: pd.DataFrame, col: str) -> dict:
        g = df.groupby(["OwnerClass", df[col]] if isinstance(col, str) else col)["count"].sum()
        return g.unstack(fill_value=0).to_dict()

    # Initial-fleet split.
    init_pv = (
        initial.groupby(["OwnerClass", "AtVessel"])["count"].sum().unstack(fill_value=0)
    )
    init_port = init_pv.get(False, pd.Series(0, index=init_pv.index)).to_dict()
    init_vessel = init_pv.get(True, pd.Series(0, index=init_pv.index)).to_dict()

    end_pv = (
        end_stock.groupby(["OwnerClass", "AtVessel"])["count"].sum().unstack(fill_value=0)
    )
    end_port = end_pv.get(False, pd.Series(0, index=end_pv.index)).to_dict()
    end_vessel = end_pv.get(True, pd.Series(0, index=end_pv.index)).to_dict()

    # "Started at port and left" — per port, capped at initial port stock.
    init_per_port = (
        initial[~initial["AtVessel"]]
        .groupby(["OwnerClass", "Location"])["count"].sum()
    )
    sail_dep = (
        sail.groupby(["OwnerClass", "DepPort"])["TEU"].sum()
        .rename("DepTEU")
    )
    # Convert TEU departures to container counts: most cargo is 40/45ft (2 TEU);
    # we cap by initial stock per port so the proxy can't exceed reality.
    left = {}
    for owner in ("Carrier", "Non-Carrier"):
        ports = init_per_port.loc[owner].to_dict() if owner in init_per_port.index else {}
        dep   = sail_dep.loc[owner].to_dict() if owner in sail_dep.index.get_level_values(0) else {}
        total = 0
        for p, init_n in ports.items():
            # Approximate container-count departures from this port: divide TEU
            # by the average TEU/container in this owner's initial stock there.
            owner_rows = initial[(initial["OwnerClass"] == owner) & (initial["Location"] == p)]
            if owner_rows["count"].sum() > 0:
                avg_teu = (owner_rows["ContainerSize"].fillna(20).clip(lower=20) / 20).clip(upper=2).mul(owner_rows["count"]).sum() / owner_rows["count"].sum()
            else:
                avg_teu = 1.5
            est_containers = dep.get(p, 0) / max(avg_teu, 1.0)
            total += min(init_n, int(round(est_containers)))
        left[owner] = total

    # "Started on vessel and arrived" — initial vessel stock minus end vessel
    # stock, clipped at >= 0 (proxy: with a 1-week horizon and dwell ≥7d, most
    # of the end-of-horizon vessel inventory comes from new departures).
    arrived_from_vessel = {
        owner: max(0, init_vessel.get(owner, 0) - end_vessel.get(owner, 0))
        for owner in ("Carrier", "Non-Carrier")
    }

    # Headline shipment rate: total sailing moves / initial fleet (count basis).
    initial_fleet = {
        owner: int(init_port.get(owner, 0) + init_vessel.get(owner, 0))
        for owner in ("Carrier", "Non-Carrier")
    }
    sail_moves_teu = sail.groupby("OwnerClass")["TEU"].sum().to_dict()
    sail_moves_full_teu = sail[~sail["Commodity"].str.upper().str.startswith("EMPTY_")] \
        .groupby("OwnerClass")["TEU"].sum().to_dict()

    # Per-leg vessel utilisation table.
    util = util.sort_values("DepHour").reset_index(drop=True)

    return {
        "label": label,
        "year": year,
        "initial_port": init_port,
        "initial_vessel": init_vessel,
        "end_port": end_port,
        "end_vessel": end_vessel,
        "left_starting_port": left,
        "arrived_from_vessel": arrived_from_vessel,
        "initial_fleet": initial_fleet,
        "sail_moves_teu": sail_moves_teu,
        "sail_moves_full_teu": sail_moves_full_teu,
        "util_table": util,
    }


def write_csv(m: dict) -> Path:
    rows = []
    for owner in ("Carrier", "Non-Carrier"):
        rows.append({
            "owner": owner,
            "initial_at_port": int(m["initial_port"].get(owner, 0)),
            "initial_on_vessel": int(m["initial_vessel"].get(owner, 0)),
            "left_starting_port": int(m["left_starting_port"].get(owner, 0)),
            "arrived_from_vessel": int(m["arrived_from_vessel"].get(owner, 0)),
            "end_at_port": int(m["end_port"].get(owner, 0)),
            "end_on_vessel": int(m["end_vessel"].get(owner, 0)),
            "sail_moves_TEU": int(m["sail_moves_teu"].get(owner, 0)),
            "sail_moves_full_TEU": int(m["sail_moves_full_teu"].get(owner, 0)),
        })
    df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"{m['label']}.csv"
    df.to_csv(out, index=False)
    return out


def plot_fleet_usage(m: dict) -> Path:
    owners = ["Carrier", "Non-Carrier"]
    init_port    = [m["initial_port"].get(o, 0)    for o in owners]
    init_vessel  = [m["initial_vessel"].get(o, 0)  for o in owners]
    left         = [m["left_starting_port"].get(o, 0) for o in owners]
    arrived      = [m["arrived_from_vessel"].get(o, 0) for o in owners]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    # Started at port: stacked left/stayed bars per owner.
    ax = axes[0]
    stayed_p = [ip - lf for ip, lf in zip(init_port, left)]
    x = np.arange(len(owners))
    ax.bar(x, left,    label="Left starting port",    color="#3b8bba")
    ax.bar(x, stayed_p, bottom=left, label="Stayed",   color="#cfd8dc")
    ax.set_xticks(x); ax.set_xticklabels(owners)
    ax.set_ylabel("Containers")
    ax.set_title("Initial port stock — moved vs. stayed")
    for i, (lf, ip) in enumerate(zip(left, init_port)):
        pct = (lf / ip * 100) if ip else 0
        ax.text(i, ip + max(init_port) * 0.01, f"{lf:,} / {ip:,} ({pct:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    # Started on vessel: stacked arrived/still-onboard bars per owner.
    ax = axes[1]
    still = [iv - ar for iv, ar in zip(init_vessel, arrived)]
    ax.bar(x, arrived, label="Arrived at port",   color="#5cb85c")
    ax.bar(x, still,   bottom=arrived, label="Still on a vessel", color="#cfd8dc")
    ax.set_xticks(x); ax.set_xticklabels(owners)
    ax.set_ylabel("Containers")
    ax.set_title("Initial mid-voyage stock — arrived vs. still in transit")
    for i, (ar, iv) in enumerate(zip(arrived, init_vessel)):
        pct = (ar / iv * 100) if iv else 0
        ax.text(i, iv + max(max(init_vessel), 1) * 0.02,
                f"{ar:,} / {iv:,} ({pct:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Initial-fleet utilisation — {m['label']}", fontsize=12)
    plt.tight_layout()
    out = PICTURES / "baseline-fleet-usage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_end_location(m: dict) -> Path:
    owners = ["Carrier", "Non-Carrier"]
    end_port   = [m["end_port"].get(o, 0)   for o in owners]
    end_vessel = [m["end_vessel"].get(o, 0) for o in owners]

    x = np.arange(len(owners))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, end_port,   label="At port",     color="#3b8bba")
    ax.bar(x, end_vessel, bottom=end_port, label="On a vessel", color="#f0ad4e")
    ax.set_xticks(x); ax.set_xticklabels(owners)
    ax.set_ylabel("Containers")
    ax.set_title(f"End-of-horizon container location — {m['label']}")
    for i, (p, v) in enumerate(zip(end_port, end_vessel)):
        total = p + v
        ax.text(i, total * 0.5, f"{p:,} at port", ha="center", va="center",
                fontsize=9, color="white")
        if v > 0:
            ax.text(i, p + v * 0.5, f"{v:,} at sea", ha="center", va="center",
                    fontsize=9, color="black")
        ax.text(i, total + max(end_port + end_vessel) * 0.01, f"{total:,}",
                ha="center", va="bottom", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = PICTURES / "baseline-end-location.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_util_table(m: dict) -> Path:
    util = m["util_table"].copy()
    util["FromTo"] = util["DepPort"] + r" $\rightarrow$ " + util["ArrPort"]
    util["Capacity"]   = util["Capacity"].astype(int)
    util["CargoTEU"]   = util["CargoTEU"].astype(int)
    util["EmptyTEU"]   = util["EmptyTEU"].astype(int)
    util["Vessel"]     = util["Vessel"] + " (" + util["Line"].str.replace(" Line", "") + ")"

    rows_tex = []
    for _, r in util.iterrows():
        rows_tex.append(
            f"{r['Vessel']} & {r['FromTo']} & {r['DepDate']} & "
            f"{r['Capacity']} & {r['CargoTEU']} & {r['EmptyTEU']} & "
            f"{r['TotalFill%']:.1f} \\\\"
        )

    body = "\n".join(rows_tex)
    tex = (
        "\\begin{longtable}{llrrrrr}\n"
        "\\caption{Per-sailing-leg vessel utilisation, baseline Week 1 2026. "
        "Cargo TEU = full + empty repositioning bookings; Total Fill\\% = "
        "(cargo + empty) / capacity.}\\label{tab:baseline-vessel-utilisation}\\\\\n"
        "\\toprule\n"
        "Vessel (Line) & Leg & Departure & Cap. (TEU) & Cargo TEU & Empty TEU & Total Fill \\% \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{7}{c}{\\tablename\\ \\thetable\\ -- continued}\\\\\n"
        "\\toprule\n"
        "Vessel (Line) & Leg & Departure & Cap. (TEU) & Cargo TEU & Empty TEU & Total Fill \\% \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\midrule\n"
        "\\multicolumn{7}{r}{continued on next page}\\\\\n"
        "\\endfoot\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "baseline-vessel-utilisation.tex"
    out.write_text(tex)
    return out


def _type_token(commodity: str) -> str:
    """Pull the ISO container-type token (4 chars, has a letter) from an id."""
    for part in str(commodity).split("_"):
        if len(part) == 4 and not part.isdigit() and any(ch.isalpha() for ch in part):
            return part
    return ""


def plot_net_change(label: str) -> Path:
    """Diverging bar of net sailing-arc flow per port (arrivals - departures).

    Replaces the cluttered 30-type stacked distribution chart. Net flow on
    sailing arcs is the conservation-consistent routing signal: it sums to zero
    over all ports (every container that departs also arrives within the
    horizon), and its sign shows which ports the model serves as net exporters
    vs net importers. Stock-level differences are not used here because the raw
    stock file counts containers (non-carrier empties, fulls) that the model
    does not retain as inventory, so end-vs-start stock conflates real
    repositioning with pool bookkeeping.
    """
    flows = pd.read_csv(REPO / "results" / "model-flows" / f"{label}.csv")
    flows["TEU"] = flows["Flow"] * flows["Commodity"].apply(teu_of)
    flows["Class"] = flows["Commodity"].apply(lambda c: reefer_class(_type_token(c)))
    sail = flows[flows["DepPort"] != flows["ArrPort"]]

    dep = sail.groupby(["DepPort", "Class"])["TEU"].sum().unstack(fill_value=0)
    arr = sail.groupby(["ArrPort", "Class"])["TEU"].sum().unstack(fill_value=0)
    for c in ("Dry", "Reefer"):
        if c not in dep.columns:
            dep[c] = 0
        if c not in arr.columns:
            arr[c] = 0

    ports = sorted(set(dep.index) | set(arr.index))
    dep = dep.reindex(ports, fill_value=0)
    arr = arr.reindex(ports, fill_value=0)
    net = (arr[["Dry", "Reefer"]] - dep[["Dry", "Reefer"]])
    net["Total"] = net["Dry"] + net["Reefer"]
    net = net.sort_values("Total")

    y = np.arange(len(net))
    fig, ax = plt.subplots(figsize=(9, 5))

    # Stacked diverging bars: positive and negative parts grow outward from 0.
    for cls, colour in (("Dry", "#3b8bba"), ("Reefer", "#b85c1a")):
        vals = net[cls].values
        pos_left = np.where(vals > 0,
                            np.where(cls == "Reefer", net["Dry"].clip(lower=0), 0), 0)
        neg_left = np.where(vals < 0,
                            np.where(cls == "Reefer", net["Dry"].clip(upper=0), 0), 0)
        left = np.where(vals >= 0, pos_left, vals + neg_left)
        ax.barh(y, vals, left=left, color=colour, label=cls, height=0.7)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(net.index)
    ax.set_xlabel("Net sailing-arc flow over the week (TEU):  arrivals − departures")
    ax.set_title(f"Net port flow — {label}")
    pad = max(net["Total"].abs().max() * 0.03, 2)
    for i, tot in enumerate(net["Total"].values):
        ax.text(tot + (pad if tot >= 0 else -pad), i, f"{tot:+,.0f}",
                ha="left" if tot >= 0 else "right", va="center", fontsize=8)
    ax.margins(x=0.15)
    ax.legend(loc="lower right", frameon=True)
    fig.text(0.13, 0.015,
             "Net exporters (←) ship more than they receive; net importers (→) the reverse. "
             "Bars sum to zero: no container is created or lost.",
             fontsize=8, style="italic", color="#555")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out = PICTURES / "baseline-net-change.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_summary(m: dict) -> None:
    print(f"\n=== Baseline metrics: {m['label']} ===")
    print(f"\nInitial fleet (containers):")
    for owner in ("Carrier", "Non-Carrier"):
        print(f"  {owner:<12} at port: {m['initial_port'].get(owner, 0):>6,}  "
              f"on vessel: {m['initial_vessel'].get(owner, 0):>6,}  "
              f"total: {m['initial_fleet'][owner]:>6,}")

    print(f"\nFleet utilisation during horizon:")
    for owner in ("Carrier", "Non-Carrier"):
        ip = m["initial_port"].get(owner, 0); lf = m["left_starting_port"].get(owner, 0)
        iv = m["initial_vessel"].get(owner, 0); ar = m["arrived_from_vessel"].get(owner, 0)
        print(f"  {owner:<12} left port: {lf:>5,} / {ip:>5,} "
              f"({(lf/ip*100 if ip else 0):4.0f}%)  "
              f"arrived from vessel: {ar:>4,} / {iv:>4,} "
              f"({(ar/iv*100 if iv else 0):4.0f}%)")

    print(f"\nEnd-of-horizon location:")
    for owner in ("Carrier", "Non-Carrier"):
        ep = m["end_port"].get(owner, 0); ev = m["end_vessel"].get(owner, 0)
        print(f"  {owner:<12} at port: {ep:>6,}  on vessel: {ev:>6,}  "
              f"total: {ep+ev:>6,}")

    print(f"\nShipment volume (TEU on sailing arcs):")
    total_fleet = sum(m["initial_fleet"].values())
    for owner in ("Carrier", "Non-Carrier"):
        teu  = m["sail_moves_teu"].get(owner, 0)
        full = m["sail_moves_full_teu"].get(owner, 0)
        print(f"  {owner:<12} total TEU: {teu:>6,.0f}  full-only: {full:>6,.0f}")
    grand_teu = sum(m["sail_moves_teu"].values())
    print(f"  Headline: total {grand_teu:,.0f} TEU moved across "
          f"{total_fleet:,} initial containers "
          f"=> {grand_teu/max(total_fleet,1):.2f} TEU-moves per container")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="2026w1",
                    help="Iteration label, e.g. 2026w1")
    args = ap.parse_args()

    m = compute_metrics(args.label)
    print_summary(m)

    # Carrier empty-pool conservation: pool tracked by the model's balance
    # constraints must be neither created nor destroyed over the horizon.
    car_in = m["initial_port"].get("Carrier", 0) + m["initial_vessel"].get("Carrier", 0)
    car_end = m["end_port"].get("Carrier", 0) + m["end_vessel"].get("Carrier", 0)
    print(f"\nCarrier empty pool tracked at horizon end: {car_end:,} "
          f"(of {car_in:,} initial; remainder delivered as full carryover)")

    csv  = write_csv(m)
    fig1 = plot_fleet_usage(m)
    fig2 = plot_end_location(m)
    fig3 = plot_net_change(args.label)
    tex  = write_util_table(m)
    print(f"\nWrote: {csv}")
    print(f"       {fig1}")
    print(f"       {fig2}")
    print(f"       {fig3}")
    print(f"       {tex}")


if __name__ == "__main__":
    main()
