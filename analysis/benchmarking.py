import pandas as pd

# ----------------------------
# Load data
# ----------------------------

orig_df = pd.read_csv("data/clean/Eimskip_data_final.csv")
orig_df = orig_df[(orig_df["Year"]==2026) & (orig_df["Week"].isin([6,7,8]))].copy()
model_df = pd.read_csv("results/model_flows.csv")

# ----------------------------
# Remove waiting flows (same port)
# ----------------------------
IGNORE_WAIT_FLOWS = True

if IGNORE_WAIT_FLOWS:
    model_df = model_df[model_df["DepPort"] != model_df["ArrPort"]]

# ----------------------------
# Clean vessel names (remove trailing digits)
# ----------------------------

# Original data
orig_df["Voyage 1"] = orig_df["Voyage 1"].str.replace(r"\d+$", "", regex=True)
orig_df["Voyage 2"] = orig_df["Voyage 2"].str.replace(r"\d+$", "", regex=True)

# ----------------------------
# Step 1: Convert original data to arc format
# ----------------------------

rows = []

for _, row in orig_df.iterrows():

    flow = row["Total"]

    # --- Leg 1 ---
    if pd.notna(row["Load 1"]) and pd.notna(row["Discharge 1"]):
        rows.append({
            "Vessel": row["Voyage 1"],
            "Line": row["Line 1"],
            "Origin": row["Load 1"],
            "Destination": row["Discharge 1"],
            "Flow": flow,
            "Size": row["ContainerSize"]
        })

    # --- Leg 2 ---
    if pd.notna(row["Load 2"]) and pd.notna(row["Discharge 2"]):
        rows.append({
            "Vessel": row["Voyage 2"],
            "Line": row["Line 2"],
            "Origin": row["Load 2"],
            "Destination": row["Discharge 2"],
            "Flow": flow,
            "Size": row["ContainerSize"]
        })

orig_arcs_df = pd.DataFrame(rows)

# ----------------------------
# Step 2: Convert to TEU
# ----------------------------

def size_to_teu(size):
    if size == 20:
        return 1
    elif size in [40, 45]:
        return 2
    else:
        return 0

orig_arcs_df["TEU"] = orig_arcs_df["Flow"] * orig_arcs_df["Size"].apply(size_to_teu)

# ----------------------------
# Step 3: Aggregate original data
# ----------------------------

orig_agg = (
    orig_arcs_df
    .groupby(["Vessel", "Origin", "Destination"])["TEU"]
    .sum()
    .reset_index()
    .rename(columns={"TEU": "TEU_data"})
)

# ----------------------------
# Step 4: Prepare model data
# ----------------------------

# Remove empty container flows
model_df = model_df[model_df["Commodity"] != "EMPTY"]

# NOTE: If Flow is already in containers, convert to TEU if needed
# (optional depending on your model output)

# Example: assume Flow already matches TEU → otherwise adjust
model_df.rename(columns={
    "DepPort": "Origin",
    "ArrPort": "Destination",
    "Flow": "TEU_model"
}, inplace=True)

model_agg = (
    model_df
    .groupby(["Vessel", "Origin", "Destination"])["TEU_model"]
    .sum()
    .reset_index()
)

# ----------------------------
# Step 5: Merge and compare
# ----------------------------

comparison = pd.merge(
    orig_agg,
    model_agg,
    on=["Vessel", "Origin", "Destination"],
    how="outer"
).fillna(0)

comparison["diff"] = comparison["TEU_model"] - comparison["TEU_data"]
comparison["abs_error"] = comparison["diff"].abs()
comparison["rel_error"] = comparison["diff"] / (comparison["TEU_data"] + 1e-6)

# ----------------------------
# Step 6: Summary metrics
# ----------------------------

total_data = comparison["TEU_data"].sum()
total_model = comparison["TEU_model"].sum()

mape = (comparison["abs_error"] / (comparison["TEU_data"] + 1e-6)).mean()

print("\n=== SUMMARY ===")
print(f"Total TEU (Data):  {total_data:.2f}")
print(f"Total TEU (Model): {total_model:.2f}")
print(f"MAPE:              {mape:.4f}")

# ----------------------------
# Step 7: Vessel-level comparison
# ----------------------------

vessel_summary = (
    comparison
    .groupby("Vessel")[["TEU_data", "TEU_model"]]
    .sum()
    .reset_index()
)

vessel_summary["diff"] = vessel_summary["TEU_model"] - vessel_summary["TEU_data"]

print("\n=== VESSEL SUMMARY ===")
print(vessel_summary.sort_values("diff", ascending=False).to_string(index=False))

# ----------------------------
# Save results
# ----------------------------

comparison.to_csv("results/comparison_arc_level.csv", index=False)
vessel_summary.to_csv("results/comparison_vessel_level.csv", index=False)

print("\nSaved:")
print("- comparison_arc_level.csv")
print("- comparison_vessel_level.csv")