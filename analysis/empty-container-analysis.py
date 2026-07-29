"""
§5.7 empty container analysis.

Aggregates carrier-empty repositioning decisions from the full-year rolling-horizon
rejection-model sweep at results/full-year-rejection/{2023,2024,2025}w1/.
Per-iteration model-flows files contain a 3-week planning window each; we filter
to the first 168 hours so each calendar week appears exactly once across the sweep.

Outputs:
  results/empty-analysis/per-port-summary.csv
  results/empty-analysis/per-port-monthly.csv
  msc-thesis-writing/Pictures/empty-net-flow-by-month.png
  msc-thesis-writing/Pictures/empty-dwell-by-port.png
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_fig, DIVERGING_CMAP, LINE_COLORS

apply_style()

REPO_ROOT   = Path(__file__).resolve().parent.parent
SWEEP_ROOT  = REPO_ROOT / "results" / "full-year-rejection"
OUTPUT_DIR  = REPO_ROOT / "results" / "empty-analysis"
PICS_DIR    = REPO_ROOT / "msc-thesis-writing" / "Pictures"
INSTANCES   = ["2023w1", "2024w1", "2025w1"]
DECISION_HOURS = 168                 # 1-week decision window per iteration

HUB_PORTS      = {"IS REY", "FO THO", "DK AAR", "NL RTM"}
ICELAND_PORTS  = {"IS REY", "IS GRT", "IS RFJ", "IS VES"}

FILE_RE = re.compile(r"(\d{4})w(\d{1,2})\.csv$")


def iso_week_month(year: int, week: int) -> int:
    """Calendar month of the Monday of ISO (year, week)."""
    return date.fromisocalendar(year, week, 1).month


def parse_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add ContainerSize/Owner/TEU columns to a model-flows DataFrame, return carrier empties only."""
    e = df[df["Commodity"].str.startswith("EMPTY_")].copy()
    if e.empty:
        return e
    parts = e["Commodity"].str.split("_", expand=True)
    e["ContainerSize"] = parts[2].astype(float).astype(int)
    e["Owner"]         = parts[3]
    e = e[e["Owner"] == "Carrier"].copy()
    e["TEU"]      = e["ContainerSize"].map(lambda s: 1 if s == 20 else 2)
    e["TEU_flow"] = e["TEU"] * e["Flow"].astype(float)
    return e


def load_all_decision_windows() -> dict[tuple[int, int], pd.DataFrame]:
    """Load each (year, week) decision window's empty flows, keyed by (year, iso_week)."""
    out: dict[tuple[int, int], pd.DataFrame] = {}
    for inst in INSTANCES:
        flow_dir = SWEEP_ROOT / inst / "model-flows"
        if not flow_dir.exists():
            print(f"[warn] missing {flow_dir}")
            continue
        for csv in sorted(flow_dir.glob("*.csv")):
            m = FILE_RE.search(csv.name)
            if not m:
                continue
            year, week = int(m.group(1)), int(m.group(2))
            df = pd.read_csv(csv)
            df = df[df["DepHour"] < DECISION_HOURS]      # decision window only
            e  = parse_empty_rows(df)
            out[(year, week)] = e
    return out


def per_port_net_flow_sailing(e: pd.DataFrame) -> pd.Series:
    """Inflow − outflow of carrier-empty TEUs on sailing arcs (excludes WAIT and TRANS)."""
    sail = e[(e["Vessel"] != "WAIT") & (e["Line"] != "TRANS")]
    if sail.empty:
        return pd.Series(dtype=float)
    outflow = sail.groupby("DepPort")["TEU_flow"].sum()
    inflow  = sail.groupby("ArrPort")["TEU_flow"].sum()
    return inflow.subtract(outflow, fill_value=0.0)


def idle_empty_inventory(e: pd.DataFrame) -> pd.Series:
    """Time-averaged idle empty TEU at each port over the 168-hour decision window.

    A wait arc with TEU flow f and duration d covers f·d TEU-hours; the portion
    of that duration falling within [0, 168] divided by 168 gives the mean TEU
    sitting on that wait segment during the decision window.
    """
    wait = e[e["Vessel"] == "WAIT"].copy()
    if wait.empty:
        return pd.Series(dtype=float)
    eff_dep = wait["DepHour"].clip(lower=0)
    eff_arr = wait["ArrHour"].clip(upper=DECISION_HOURS)
    eff_dur = (eff_arr - eff_dep).clip(lower=0)
    wait["mean_idle_TEU"] = wait["TEU_flow"] * eff_dur / DECISION_HOURS
    return wait.groupby("DepPort")["mean_idle_TEU"].sum()


def per_port_arrivals_sailing(e: pd.DataFrame) -> pd.Series:
    sail = e[(e["Vessel"] != "WAIT") & (e["Line"] != "TRANS")]
    if sail.empty:
        return pd.Series(dtype=float)
    return sail.groupby("ArrPort")["TEU_flow"].sum()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICS_DIR.mkdir(parents=True, exist_ok=True)

    windows = load_all_decision_windows()
    if not windows:
        raise SystemExit("No decision windows loaded.")
    print(f"Loaded {len(windows)} decision windows.")

    # ── Per-port net flow per week, then per month and overall ───────────────
    weekly_rows = []
    monthly_acc: dict[tuple[str, int], list[float]] = {}
    dwell_acc: dict[str, dict[str, float]] = {}

    for (year, week), e in windows.items():
        net = per_port_net_flow_sailing(e)
        month = iso_week_month(year, week)
        for port, net_flow in net.items():
            weekly_rows.append({"year": year, "week": week, "month": month,
                                "port": port, "net_flow_teu": float(net_flow)})
            monthly_acc.setdefault((port, month), []).append(float(net_flow))

        idle = idle_empty_inventory(e)
        arrivals = per_port_arrivals_sailing(e)
        for port in set(idle.index) | set(arrivals.index):
            acc = dwell_acc.setdefault(port, {"idle_TEU_sum": 0.0,
                                              "arriving_TEU": 0.0,
                                              "n_weeks": 0})
            acc["idle_TEU_sum"]   += float(idle.get(port, 0.0))
            acc["arriving_TEU"]   += float(arrivals.get(port, 0.0))
            acc["n_weeks"]        += 1

    weekly_df = pd.DataFrame(weekly_rows)

    # Per-port summary (chronic picture + dwell).
    summary = (weekly_df.groupby("port")
                       .agg(mean_net_flow_teu=("net_flow_teu", "mean"),
                            n_weeks=("net_flow_teu", "count"),
                            n_weeks_in_deficit=("net_flow_teu", lambda s: int((s < 0).sum())))
                       .reset_index())
    dwell_df = pd.DataFrame([{"port": p,
                              "mean_idle_TEU": (v["idle_TEU_sum"] / v["n_weeks"]
                                                  if v["n_weeks"] > 0 else 0.0),
                              "arriving_TEU": v["arriving_TEU"]}
                             for p, v in dwell_acc.items()])
    summary = summary.merge(dwell_df, on="port", how="left")
    summary["mean_idle_TEU"] = summary["mean_idle_TEU"].fillna(0.0)
    summary["arriving_TEU"]  = summary["arriving_TEU"].fillna(0.0)
    summary = summary.sort_values("mean_net_flow_teu").reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "per-port-summary.csv", index=False)
    print(f"\nPer-port summary (sorted by mean net flow, deficit-first):")
    print(summary.to_string(index=False))

    # Monthly per port.
    monthly_rows = [{"port": p, "month": m, "mean_net_flow_teu": float(np.mean(v))}
                    for (p, m), v in monthly_acc.items()]
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(OUTPUT_DIR / "per-port-monthly.csv", index=False)

    # ── Figure 1: monthly heatmap, ports × month ──────────────────────────────
    # Curated selection: main deficit ports and surplus hubs discussed in the text.
    SELECTED_PORTS = ["IS REY", "GB IMM", "US PWM", "DE BRV",
                      "GB TEE", "NL RTM", "IS GRT", "DK AAR"]
    busy_ports = (summary[summary["port"].isin(SELECTED_PORTS)]
                  .sort_values("mean_net_flow_teu")["port"].tolist())
    if not busy_ports:
        busy_ports = summary.sort_values("arriving_TEU", ascending=False).head(10)["port"].tolist()

    pivot = (monthly_df[monthly_df["port"].isin(busy_ports)]
             .pivot(index="port", columns="month", values="mean_net_flow_teu")
             .reindex(busy_ports)
             .reindex(columns=range(1, 13)))

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(busy_ports) + 1.5))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap=DIVERGING_CMAP,
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(busy_ports)))
    ax.set_yticklabels(busy_ports)
    ax.set_xlabel("Calendar month")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Empty TEU flow per week")
    fig.tight_layout()
    save_fig(fig, "empty-net-flow-by-month")
    plt.close(fig)

    # ── Figure 2: mean dwell by port ──────────────────────────────────────────
    dwell_busy = (summary.set_index("port")
                  .reindex(busy_ports)
                  .reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(dwell_busy["port"], dwell_busy["mean_idle_TEU"], color=LINE_COLORS["Blue"])
    ax.set_xticklabels(dwell_busy["port"], rotation=45, ha="right")
    ax.set_ylabel("Mean idle empty inventory (TEU)")
    fig.tight_layout()
    save_fig(fig, "empty-dwell-by-port")
    plt.close(fig)


if __name__ == "__main__":
    main()
