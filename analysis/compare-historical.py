import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Load historical data (weeks 6-8, 2026, full containers only)
# ----------------------------

hist_raw = pd.read_csv("data/clean/Eimskip_data_final.csv")
hist = (
    hist_raw[
        (hist_raw["Year"] == 2026)
        & (hist_raw["Week"].isin([6, 7, 8]))
        & (hist_raw["Full/Empty"] == "F")
    ]
    .copy()
)

# Match commodity construction: final destination = Discharge 2 if present, else Discharge 1
hist["Destination"] = hist["Discharge 2"].combine_first(hist["Discharge 1"])
hist["Origin"] = hist["Load 1"]
hist["Owner"] = hist["Owner"].str.capitalize()  # CARRIER -> Carrier, etc.

hist_agg = (
    hist.groupby(["Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])["Total"]
    .sum()
    .reset_index()
    .rename(columns={"Total": "Historical"})
)

# Container size 26 exists in historical data but is not modelled — flag it
sizes_not_modelled = hist_agg[~hist_agg["ContainerSize"].isin([20, 40, 45])]
if len(sizes_not_modelled) > 0:
    print(f"NOTE: {len(sizes_not_modelled)} historical rows with container sizes not in model "
          f"({sizes_not_modelled['ContainerSize'].unique().tolist()}) — excluded from comparison.")
    hist_agg = hist_agg[hist_agg["ContainerSize"].isin([20, 40, 45])]

# ----------------------------
# Load model fulfillment
# ----------------------------

model = pd.read_csv("results/demand_fulfillment.csv")

# Extract ContainerType and ContainerSize from commodity ID (e.g. k0_Dry_40_Carrier)
model[["ContainerType", "ContainerSize"]] = (
    model["Commodity"]
    .str.extract(r'_(\w+?)_(\d+)_')
    .rename(columns={0: "ContainerType", 1: "ContainerSize"})
)
model["ContainerSize"] = model["ContainerSize"].astype(int)

model["Owner"] = model["Owner"].str.capitalize()

model_agg = (
    model.groupby(["Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])[["Demand", "Fulfilled"]]
    .sum()
    .reset_index()
)

# ----------------------------
# Merge
# ----------------------------

comparison = pd.merge(
    hist_agg,
    model_agg,
    on=["Origin", "Destination", "ContainerType", "ContainerSize", "Owner"],
    how="outer"
).fillna(0)

comparison["ContainerSize"] = comparison["ContainerSize"].astype(int)
comparison["Spec"] = comparison["ContainerType"] + " " + comparison["ContainerSize"].astype(str) + "ft"
comparison["OD"] = comparison["Origin"] + " → " + comparison["Destination"]

# Rows only in historical (model never considered them as demand)
only_hist = comparison[comparison["Demand"] == 0]
# Rows only in model (no historical equivalent — shouldn't happen since model uses the same data)
only_model = comparison[comparison["Historical"] == 0]

print("\n=== OD/Spec pairs in historical but absent from model demand ===")
print(only_hist[["OD", "Spec", "Owner", "Historical"]].to_string(index=False) if len(only_hist) else "  None")

print("\n=== OD/Spec pairs in model demand but absent from historical ===")
print(only_model[["OD", "Spec", "Owner", "Demand", "Fulfilled"]].to_string(index=False) if len(only_model) else "  None")

# Focus comparison on rows present in both
both = comparison[(comparison["Historical"] > 0) & (comparison["Demand"] > 0)].copy()
both["FillRate"] = both["Fulfilled"] / both["Demand"]
both["HistVsDemand"] = both["Historical"] / both["Demand"]  # how close historical is to model demand

print(f"\n=== Summary ===")
print(f"Total historical containers (matched specs): {comparison['Historical'].sum():.0f}")
print(f"Total model demand:                          {comparison['Demand'].sum():.0f}")
print(f"Total model fulfilled:                       {comparison['Fulfilled'].sum():.0f}")
print(f"Overall model fill rate:                     {comparison['Fulfilled'].sum() / comparison['Demand'].sum():.1%}")

# ----------------------------
# Plot 1: Historical vs Model Demand vs Model Fulfilled — aggregated by OD
# ----------------------------

od_summary = (
    comparison.groupby("OD")[["Historical", "Demand", "Fulfilled"]]
    .sum()
    .reset_index()
    .sort_values("Historical", ascending=False)
)

x = np.arange(len(od_summary))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, od_summary["Historical"], width, label="Historical (actual)", color="#4C72B0")
ax.bar(x,         od_summary["Demand"],     width, label="Model demand",        color="#aec6e8", edgecolor="#4C72B0")
ax.bar(x + width, od_summary["Fulfilled"],  width, label="Model fulfilled",      color="#55A868")

ax.set_title("Historical Actuals vs Model Demand vs Model Fulfilled — by OD Pair")
ax.set_xticks(x)
ax.set_xticklabels(od_summary["OD"], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Containers")
ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: Fill rate heatmap — model vs historical, by Owner
# ----------------------------

owners = sorted(comparison["Owner"].unique())
specs  = sorted(comparison["Spec"].unique())
od_list = sorted(comparison["OD"].unique())

fig, axes = plt.subplots(1, len(owners), figsize=(9 * len(owners), max(6, len(od_list) * 0.35)))
if len(owners) == 1:
    axes = [axes]

for ax, owner in zip(axes, owners):
    sub = comparison[comparison["Owner"] == owner]

    hist_pivot  = sub.pivot_table(index="OD", columns="Spec", values="Historical",  aggfunc="sum").reindex(index=od_list, columns=specs)
    model_pivot = sub.pivot_table(index="OD", columns="Spec", values="Fulfilled",   aggfunc="sum").reindex(index=od_list, columns=specs)
    demand_pivot= sub.pivot_table(index="OD", columns="Spec", values="Demand",      aggfunc="sum").reindex(index=od_list, columns=specs)

    # Fill rate = fulfilled / demand; show NaN where no demand
    rate = (model_pivot / demand_pivot).astype(float)

    # Annotation: "fulfilled / hist" so you can see model vs actual side-by-side
    annot = (
        model_pivot.fillna(0).astype(int).astype(str)
        + "\nvs\n"
        + hist_pivot.fillna(0).astype(int).astype(str)
    )
    annot[demand_pivot.isna()] = ""

    # Drop OD rows with no data for this owner
    has_data = ~(rate.isna().all(axis=1))
    rate  = rate[has_data]
    annot = annot[has_data]
    od_labels = rate.index.tolist()

    cmap = plt.cm.RdYlGn
    cmap.set_bad("#eeeeee")
    im = ax.imshow(rate.values.astype(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for r in range(rate.shape[0]):
        for c in range(rate.shape[1]):
            text = annot.iloc[r, c]
            if text:
                val = rate.values[r, c]
                fc = "black" if 0.25 < val < 0.85 else "white"
                ax.text(c, r, text, ha="center", va="center", fontsize=6, color=fc)

    ax.set_yticks(range(len(od_labels)))
    ax.set_yticklabels(od_labels, fontsize=7)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(specs, fontsize=8)
    ax.set_title(f"{owner}\n(cell: model fulfilled vs historical actual)", fontsize=10)
    plt.colorbar(im, ax=ax, label="Model fill rate", format="{x:.0%}", shrink=0.5)

fig.suptitle("Model Fill Rate Heatmap — Model Fulfilled vs Historical Actual", fontsize=13)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 3: Scatter — model fulfilled vs historical, one point per OD/Spec/Owner
# ----------------------------

fig, ax = plt.subplots(figsize=(7, 7))
for owner, color in zip(["Carrier", "Leased", "Shipper"], ["#4C72B0", "#DD8452", "#55A868"]):
    sub = comparison[comparison["Owner"] == owner]
    ax.scatter(sub["Historical"], sub["Fulfilled"], label=owner, alpha=0.7, color=color, edgecolors="white", s=60)

max_val = comparison[["Historical", "Fulfilled"]].max().max()
ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Perfect match")
ax.set_xlabel("Historical containers moved")
ax.set_ylabel("Model containers fulfilled")
ax.set_title("Model Fulfilled vs Historical — per OD/Spec/Owner")
ax.legend()
plt.tight_layout()
plt.show()
