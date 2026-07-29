import sys
import pandas as pd

sys.path.insert(0, "pipeline")
from config import COMMODITIES_PATH

# ----------------------------
# Historical vessel assignments — per week
# ----------------------------

hist_raw = pd.read_csv("data/clean/Eimskip_data_final.csv")
hist = (
    hist_raw[
        (hist_raw["Year"] == 2026)
        & (hist_raw["Week"].isin([6, 7, 8]))
        & (hist_raw["Full/Empty"] == "F")
        & (hist_raw["ContainerSize"].isin([20, 40, 45]))
    ]
    .copy()
)

hist["Destination"] = hist["Discharge 2"].combine_first(hist["Discharge 1"])
hist["Origin"]      = hist["Load 1"]
hist["Owner"]       = hist["Owner"].str.capitalize()
hist["HistVessel"]  = hist["Voyage 1"].str[:3]

hist_vessels = (
    hist.groupby(["Week", "Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])
    .agg(HistVessel=("HistVessel", "first"),   # one vessel per week per voyage
         HistCount=("Total", "sum"))
    .reset_index()
)

# ----------------------------
# Model vessel assignments — per week
# (map commodity ID back to its week using the commodities CSV)
# ----------------------------

commod_df = pd.read_csv(COMMODITIES_PATH)
commod_df["Commodity"] = [
    f"k{i}_{row.ContainerType}_{row.ContainerSize}_{row.Owner}"
    for i, row in commod_df.iterrows()
]
week_of = dict(zip(commod_df["Commodity"], commod_df["Week"]))

model = pd.read_csv("results/demand_fulfillment.csv")
model["Owner"]  = model["Owner"].str.capitalize()
model["Week"]   = model["Commodity"].map(week_of)
model[["ContainerType", "ContainerSize"]] = (
    model["Commodity"]
    .str.extract(r'_(\w+?)_(\d+)_')
    .rename(columns={0: "ContainerType", 1: "ContainerSize"})
)
model["ContainerSize"] = model["ContainerSize"].astype(int)

model_vessels = (
    model[model["Fulfilled"] > 0]
    .groupby(["Week", "Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])
    .agg(ModelVessel=("Vessel", "first"),
         ModelFulfilled=("Fulfilled", "sum"))
    .reset_index()
)

# ----------------------------
# Merge and compare per week
# ----------------------------

comparison = pd.merge(
    hist_vessels,
    model_vessels,
    on=["Week", "Origin", "Destination", "ContainerType", "ContainerSize", "Owner"],
    how="outer"
)

comparison["Match"] = comparison["HistVessel"] == comparison["ModelVessel"]
comparison = comparison.sort_values(["Match", "Week", "Origin", "Destination"])

mismatches = comparison[comparison["Match"] == False]
matches    = comparison[comparison["Match"] == True]

print(f"Total commodity-weeks compared: {len(comparison)}")
print(f"  Match:    {len(matches)}")
print(f"  Mismatch: {len(mismatches[mismatches['HistVessel'].notna() & mismatches['ModelVessel'].notna()])}")
print(f"  Hist only (unfulfilled or missing): {len(mismatches[mismatches['ModelVessel'].isna()])}")
print(f"  Model only (no historical):          {len(mismatches[mismatches['HistVessel'].isna()])}")

print("\n" + "="*90)
print("VESSEL MISMATCHES  (Hist vessel ≠ Model vessel, both present)")
print("="*90)

true_mismatches = mismatches[mismatches["HistVessel"].notna() & mismatches["ModelVessel"].notna()]
if len(true_mismatches) > 0:
    for _, row in true_mismatches.iterrows():
        print(
            f"  W{int(row['Week'])}  {row['Origin']:8s} → {row['Destination']:8s} | "
            f"{row['ContainerType']:6s} {int(row['ContainerSize']):2d}ft | "
            f"{row['Owner']:8s} | "
            f"Hist: {str(row['HistVessel']):6s}  Model: {str(row['ModelVessel'])}"
        )
else:
    print("  None.")

print("\n" + "="*90)
print("UNFULFILLED IN MODEL  (historical count > 0, model fulfilled = 0 or missing)")
print("="*90)

hist_only = mismatches[mismatches["ModelVessel"].isna()]
if len(hist_only) > 0:
    for _, row in hist_only.iterrows():
        print(
            f"  W{int(row['Week'])}  {row['Origin']:8s} → {row['Destination']:8s} | "
            f"{row['ContainerType']:6s} {int(row['ContainerSize']):2d}ft | "
            f"{row['Owner']:8s} | "
            f"Hist: {str(row['HistVessel']):6s}  HistCount: {int(row['HistCount'])}"
        )
else:
    print("  None.")

comparison.to_csv("results/vessel_comparison.csv", index=False)
print("\nSaved to vessel_comparison.csv")
