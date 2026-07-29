"""
analysis/thesis-comparison-figures.py

Generate Section 5.4 figures:
  1. fill-rate-boxplot.png         - box+whisker fill rate across instances, single block
  2. fill-rate-per-iteration.png   - line chart fill rate per decision week, rolling

Run from repo root:
    python analysis/thesis-comparison-figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

from plot_style import apply_style, save_fig, MODEL, STATUS

OUT = Path("results/comparison-figures")
OUT.mkdir(parents=True, exist_ok=True)

SINGLE  = Path("results/model-comparison-single-block")
ROLLING = Path("results/model-comparison-rolling")

INSTANCES = ["2023w1", "2024w1", "2025w1", "2026w1"]
MODELS    = ["rejection", "delay", "sequential"]
LABELS    = {"rejection": "Rejection", "delay": "Delay", "sequential": "Sequential"}
COLORS    = {"rejection": MODEL["rejection"], "delay": MODEL["delay"],
             "sequential": STATUS["delivered"]}

apply_style()


# ── Figure 1: unfulfilled-demand boxplot across instances ─────────────────────

def _unfulfilled_boxplot(df, name, ymax=4000, yticks=(0, 1, 10, 100, 1000),
                         start_at_zero=True, ymin=None):
    """Box-and-whisker of unfulfilled containers per model.

    With ``start_at_zero`` (single block, where some instances reach 100% fill)
    the axis is symlog: linear through 0 so those instances sit exactly at the
    origin, logarithmic above. Otherwise (rolling horizon, where every instance
    leaves hundreds unserved) a plain log axis is used over ``[ymin, ymax]`` so
    the populated range fills the panel instead of an empty zero-band.
    Start-of-year instances are drawn as circles, mid-year (week 26) as triangles.
    """
    df = df.copy()
    df["unfilled"] = df["demand"] - df["fulfilled"]
    df["midyear"] = ~df["instance"].str.endswith("w1")

    data = {m: df[df["model"] == m]["unfilled"].values for m in MODELS}
    positions = [1, 2, 3]
    rng = np.random.default_rng(0)
    jitter = {m: rng.uniform(-0.10, 0.10, size=len(df[df["model"] == m]))
              for m in MODELS}

    fig, ax = plt.subplots(figsize=(6.2, 5))

    bp = ax.boxplot(
        [data[m] for m in MODELS], positions=positions, widths=0.5,
        patch_artist=True, medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
        showfliers=False, zorder=3,
    )
    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.55)

    for pos, model in zip(positions, MODELS):
        sub = df[df["model"] == model]
        for marker, mask in [("o", ~sub["midyear"]), ("^", sub["midyear"])]:
            sel = mask.values
            ax.scatter(pos + jitter[model][sel], sub["unfilled"].values[sel],
                       marker=marker, color=COLORS[model],
                       edgecolors="white", linewidths=0.6, s=55, zorder=4)

    if start_at_zero:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, ymax)                   # axis starts at 0
        ax.set_yticks(list(yticks))            # explicit ticks incl. the 0 tick
    else:
        ax.set_yscale("log")                   # normal log axis: default decade ticks
        ax.set_ylim(ymin if ymin is not None else 100, ymax)
        if yticks is not None:
            ax.set_yticks(list(yticks))
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel("Unfulfilled containers")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)

    soy = mlines.Line2D([], [], marker="o", linestyle="none", color="0.35",
                        markeredgecolor="white", markersize=8, label="Start of year")
    mid = mlines.Line2D([], [], marker="^", linestyle="none", color="0.35",
                        markeredgecolor="white", markersize=8, label="Mid year (week 26)")
    ax.legend(handles=[soy, mid], frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    save_fig(fig, name)
    return fig


def fig_boxplot():
    df = pd.read_csv(SINGLE / "summary.csv")
    return _unfulfilled_boxplot(df, "single-block-unfulfilled-boxplot")


def fig_rolling_boxplot():
    df = pd.read_csv(ROLLING / "summary.csv")
    # Normal log axis starting at 100 (every rolling instance leaves hundreds
    # unserved, so there are no zeros to accommodate).
    return _unfulfilled_boxplot(
        df, "rolling-unfulfilled-boxplot",
        start_at_zero=False, ymin=100, ymax=5000, yticks=None)


# ── Figure 2: fill rate per decision week, rolling horizon ───────────────────

def fig_line():
    records = []
    for inst in INSTANCES:
        year = inst[:4]
        for model in MODELS:
            base = ROLLING / f"{inst}_{model}" / "demand-fulfilment"
            for wk in range(1, 6):
                csv_path = base / f"{year}w{wk}.csv"
                if not csv_path.exists():
                    continue
                d = pd.read_csv(csv_path)
                dem = d["Demand"].sum()
                ful = d["Fulfilled"].sum()
                records.append({"instance": inst, "model": model,
                                "week": wk, "fill_rate": ful / dem * 100 if dem else 0})

    df = pd.DataFrame(records)
    avg = df.groupby(["model", "week"])["fill_rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"rejection": "o", "delay": "s", "sequential": "^"}

    for model in MODELS:
        sub = avg[avg["model"] == model].sort_values("week")
        ax.plot(sub["week"], sub["fill_rate"],
                marker=markers[model], color=COLORS[model],
                linewidth=2, markersize=6, label=LABELS[model])
        # annotate each point
        for _, row in sub.iterrows():
            ax.annotate(f"{row['fill_rate']:.1f}",
                        xy=(row["week"], row["fill_rate"]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=COLORS[model])

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"Week {i}" for i in range(1, 6)])
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("Fill rate per decision week — rolling horizon (mean over 4 instances)", pad=10)
    ax.set_ylim(88, 104)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    path = OUT / "fill-rate-per-iteration.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_boxplot()
    fig_rolling_boxplot()
    fig_line()
    plt.show()
    print(f"\nFigures saved to {OUT}/")
