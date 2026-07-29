"""
analysis/plot_style.py — shared figure styling for all thesis figures.

Single source of truth for colours, fonts, and export settings so every figure
included in the thesis reads as one visual system. Import this at the top of any
plotting script:

    from plot_style import apply_style, save_fig, LINE_COLORS, line_color, STATUS, MODEL

    apply_style()
    ...
    save_fig(fig, "my-figure")          # -> msc-thesis-writing/Pictures/my-figure.png

Design rules
------------
* The four Eimskip service lines are named after colours, so any figure that
  colours data "by line" MUST use that line's namesake colour. Use ``line_color``.
* Non-line encodings (delivery status, container type, model variant) use the
  secondary palettes below, chosen not to clash with the reserved line colours.
* Sequential fill-rate heatmaps keep ``SEQ_CMAP`` (RdYlGn).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PICTURES = REPO / "msc-thesis-writing" / "Pictures"

# ── Reserved service-line palette ─────────────────────────────────────────────
# Keyed on the bare line name ("Green") as used in the route/port tables; the
# "Green Line" form is accepted by line_color() too. Yellow is a gold so it stays
# legible on a white background.
LINE_COLORS = {
    "Red":    "#E53935",
    "Green":  "#2E8B57",
    "Yellow": "#E1A400",
    "Blue":   "#2C5F99",
}


def line_color(name: str) -> str:
    """Return the hex colour for a service line.

    Accepts "Green", "Green Line", "green", etc. Raises KeyError on an unknown
    line so a typo fails loudly rather than silently mis-colouring a figure.
    """
    key = str(name).strip().removesuffix(" Line").removesuffix(" line").strip().title()
    try:
        return LINE_COLORS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown service line {name!r}; expected one of {list(LINE_COLORS)}"
        ) from exc


# ── Secondary palettes (non-line encodings) ──────────────────────────────────
# Delivery status (stacked fulfilment bars).
STATUS = {
    "delivered":   "#2E8B57",   # green  — arrived at destination
    "in_transit":  "#94C794",   # light green — fulfilled, still moving
    "unfulfilled": "#E53935",   # red    — unmet demand
}

# Model-variant contrast (integrated vs sequential, rejection/delay comparisons).
MODEL = {
    "integrated": "#2C5F99",
    "sequential": "#E07B39",
    "baseline":   "#7F7F7F",
    "rejection":  "#2C5F99",
    "delay":      "#E07B39",
}

# Container-type contrast (dark/light pairs) for stock distribution bars.
TYPE_COLORS = {
    "Dry":     ("#2C5F99", "#A8C4E0"),
    "Reefer":  ("#B85C1A", "#F0C4A0"),
    "Unknown": ("#7F7F7F", "#CCCCCC"),
}

# Sequential / diverging colormaps.
SEQ_CMAP = "RdYlGn"      # fill-rate heatmaps
DIVERGING_CMAP = "RdBu"  # net empty flow


def apply_style() -> None:
    """Set global matplotlib rcParams shared by every thesis figure."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "legend.frameon":     False,
        "figure.dpi":         200,
        "savefig.dpi":        200,
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
    })


def save_fig(fig, name: str) -> Path:
    """Save ``fig`` as a PNG into the thesis Pictures directory.

    ``name`` may include or omit the ``.png`` extension. Returns the path written.
    """
    PICTURES.mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.endswith(".png") else name
    out = PICTURES / f"{stem}.png"
    fig.savefig(out)
    print(f"Saved {out}")
    return out
