import argparse
import os

import numpy as np
import pandas as pd

INPUT_PATH = "data/clean/Eimskip_data_final.csv"
OUTPUT_DIR = "data/augmented"


SCENARIOS = [
    ("baseline", 0.00),
    ("light",    0.05),
    ("moderate", 0.10),
    ("heavy",    0.20),
]


def _od_row_weights(df: pd.DataFrame) -> np.ndarray:

    od_total = df.groupby(["Load 1", "Discharge 1"])["Total"].transform("sum")
    grand_total = int(df["Total"].sum())
    w = od_total.to_numpy(dtype=float) / max(grand_total, 1)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def augment_demand(df: pd.DataFrame, target: float, outlier_rate: float,
                   bump_mean: float, bump_std: float,
                   selection: str,
                   rng: np.random.Generator) -> pd.DataFrame:

    out = df.copy()
    totals = out["Total"].values.astype(int)
    original_total = int(totals.sum())
    target_total   = int(round(original_total * (1.0 + target)))

    if target == 0.0:
        return out

    n_rows = len(totals)

    # Compute OD weights
    od_weights = _od_row_weights(out) if selection == "od-weighted" else None

    if selection == "uniform":
        is_outlier = rng.random(n_rows) < outlier_rate
    elif selection == "od-weighted":
        n_pick = int(round(outlier_rate * n_rows))
        # Sample without replacement; rows on busy OD pairs are more likely.
        nz = od_weights > 0
        picks = rng.choice(np.where(nz)[0], size=n_pick, replace=False, p=od_weights[nz] / od_weights[nz].sum())
        is_outlier = np.zeros(n_rows, dtype=bool)
        is_outlier[picks] = True
    else:
        raise ValueError(f"Unknown selection method: {selection!r}")

    bump = rng.normal(loc=totals * bump_mean, scale=totals * bump_std)
    bump = np.maximum(bump, 0).round().astype(int)
    bump[~is_outlier] = 0
    new_totals = totals + bump

    achieved = int(new_totals.sum())
    delta    = target_total - achieved

    if delta > 0:
        if selection == "od-weighted":
            eligible = np.where(od_weights > 0)[0]
            w = od_weights[eligible]
            w = w / w.sum()
        else:
            eligible = np.where(totals > 0)[0]
            w = totals[eligible].astype(float)
            w = w / w.sum()
        picks = rng.choice(eligible, size=delta, replace=True, p=w)
        for idx in picks:
            new_totals[idx] += 1
    elif delta < 0:
        for _ in range(-delta):
            eligible = np.where(new_totals > totals)[0]
            if len(eligible) == 0:
                eligible = np.where(new_totals > 1)[0]
            if selection == "od-weighted":
                w = od_weights[eligible]
                if w.sum() > 0:
                    w = w / w.sum()
                    idx = int(rng.choice(eligible, p=w))
                else:
                    idx = int(rng.choice(eligible))
            else:
                idx = int(rng.choice(eligible))
            new_totals[idx] -= 1

    out["Total"] = new_totals.astype(int)
    return out


def generate_new_commodities(df: pd.DataFrame, target: float,
                             rng: np.random.Generator,
                             top_lanes: int = 0) -> pd.DataFrame:
 
    if target == 0.0:
        return df.copy()

    full = df[df["Full/Empty"].astype(str).str.upper().str.startswith("F")].copy()
    if full.empty:
        return df.copy()

    original_total = int(df["Total"].values.sum())
    need_add = int(round(original_total * target))

    if top_lanes and top_lanes > 0:
        lane_vol = full.groupby(["Load 1", "Discharge 1"])["Total"].sum()
        keep = set(lane_vol.sort_values(ascending=False).head(top_lanes).index)
        full = full[full.set_index(["Load 1", "Discharge 1"]).index.isin(keep)].copy()
        print(f"    [generate] restricted to top {top_lanes} lanes "
              f"({len(full):,} booking rows, "
              f"{int(full['Total'].sum()):,} containers in pool)")

    records = full.to_dict("records")
    mean_total = max(float(full["Total"].mean()), 1.0)
    new_rows = []
    added = 0
    while added < need_add:
        n = max(int((need_add - added) / mean_total) + 1, 1)
        idx = rng.choice(len(records), size=n)     # uniform over bookings
        for i in idx:
            r = dict(records[i])                   # template + attrs + own Total
            qty = int(r["Total"])
            if qty <= 0:
                continue
            if added + qty > need_add:             # trim final row to hit target
                qty = need_add - added
            r["Total"] = qty
            new_rows.append(r)
            added += qty
            if added >= need_add:
                break

    generated = pd.DataFrame(new_rows, columns=df.columns)
    return pd.concat([df, generated], ignore_index=True)


def scale_demand(df: pd.DataFrame, target: float,
                 rng: np.random.Generator) -> pd.DataFrame:

    out = df.copy()
    if target == 0.0:
        return out

    is_full = out["Full/Empty"].astype(str).str.upper().str.startswith("F")
    full = out[is_full]
    if full.empty:
        return out

    original_total = int(full["Total"].values.sum())
    need_add = int(round(original_total * target))

    lane_vol = full.groupby(["Load 1", "Discharge 1"])["Total"].transform("sum")
    total_vol = float(full["Total"].sum())
    pop = (lane_vol / total_vol).to_numpy(float)         
    totals = full["Total"].to_numpy(float)


    denom = float((totals * pop).sum())
    c = need_add / denom if denom > 0 else 0.0
    extra = np.round(totals * c * pop).astype(int)
    new_full = full["Total"].to_numpy(int) + extra


    delta = need_add - int(extra.sum())
    if delta != 0:
        w = (totals * pop)
        w = w / w.sum() if w.sum() > 0 else None
        step = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            i = int(rng.choice(len(new_full), p=w))
            if step < 0 and new_full[i] <= 1:
                continue
            new_full[i] += step

    out.loc[is_full, "Total"] = new_full
    return out


def uniform_scale_demand(df: pd.DataFrame, target: float,
                         rng: np.random.Generator) -> pd.DataFrame:

    out = df.copy()
    if target == 0.0:
        return out

    is_full = out["Full/Empty"].astype(str).str.upper().str.startswith("F")
    full = out[is_full]
    if full.empty:
        return out

    original_total = int(full["Total"].values.sum())
    need_add = int(round(original_total * target))

    totals = full["Total"].to_numpy(float)
    extra = np.round(totals * target).astype(int)     
    new_full = full["Total"].to_numpy(int) + extra

    delta = need_add - int(extra.sum())
    if delta != 0:
        w = totals / totals.sum() if totals.sum() > 0 else None
        step = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            i = int(rng.choice(len(new_full), p=w))
            if step < 0 and new_full[i] <= 1:
                continue
            new_full[i] += step

    out.loc[is_full, "Total"] = new_full
    return out


def write_scenario(df: pd.DataFrame, scenario: str, target: float,
                   outlier_rate: float, bump_mean: float, bump_std: float,
                   selection: str, seed: int, method: str = "bump",
                   top_lanes: int = 0) -> None:
    rng = np.random.default_rng(seed)
    if method == "uscale":
        augmented = uniform_scale_demand(df, target, rng)
    elif method == "scale":
        augmented = scale_demand(df, target, rng)
    elif method == "generate":
        augmented = generate_new_commodities(df, target, rng, top_lanes)
    else:
        augmented = augment_demand(df, target, outlier_rate, bump_mean, bump_std,
                                   selection, rng)

 
    if method in ("scale", "uscale"):
        full_mask = df["Full/Empty"].astype(str).str.upper().str.startswith("F")
        original_total = int(df.loc[full_mask, "Total"].sum())
        aug_full_mask  = augmented["Full/Empty"].astype(str).str.upper().str.startswith("F")
        achieved_total = int(augmented.loc[aug_full_mask, "Total"].sum())
    else:
        original_total = int(df["Total"].sum())
        achieved_total = int(augmented["Total"].sum())
    target_total   = int(round(original_total * (1.0 + target)))


    if scenario == "baseline":
        out_dir = os.path.join(OUTPUT_DIR, scenario)
    elif method == "uscale":
        out_dir = os.path.join(OUTPUT_DIR, f"{scenario}_uniform")
    elif method == "scale":
        out_dir = os.path.join(OUTPUT_DIR, f"{scenario}_scaled")
    elif method == "generate":
        suffix = f"_gen_top{top_lanes}" if top_lanes and top_lanes > 0 else "_gen"
        out_dir = os.path.join(OUTPUT_DIR, f"{scenario}{suffix}")
    elif selection == "uniform":
        out_dir = os.path.join(OUTPUT_DIR, scenario)
    else:
        out_dir = os.path.join(OUTPUT_DIR, f"{scenario}_od")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Eimskip_data_final.csv")
    augmented.to_csv(out_path, index=False)

    pct_actual = (achieved_total / original_total - 1.0) * 100
    delta = achieved_total - target_total
    method_tag = "uscale" if method == "uscale" else \
                 ("scale" if method == "scale" else
                  ("generate" if method == "generate" else
                   ("uniform" if selection == "uniform" else "od-weighted")))
    print(f"  [{scenario:<8} | {method_tag:<11}] target {target*100:5.1f}% | "
          f"orig {original_total:,} -> {achieved_total:,} "
          f"(actual {pct_actual:+.2f}%, delta vs target = {delta:+d}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment demand to an exact target percentage")
    parser.add_argument("--input",        default=INPUT_PATH,  help="Input demand CSV")
    parser.add_argument("--target",       type=float, default=0.10,
                        help="Target fractional increase in total containers (default: 0.10)")
    parser.add_argument("--scenario",     default="moderate",
                        help="Scenario name; output goes to data/augmented/{scenario}[_od]/")
    parser.add_argument("--selection",    choices=["uniform", "od-weighted"],
                        default="uniform",
                        help="Row-selection mechanism for bumping (default: uniform)")
    parser.add_argument("--method",        choices=["bump", "generate", "scale", "uscale"],
                        default="bump",
                        help="bump random rows, generate new OD-weighted "
                             "bookings, scale every row by route popularity, or "
                             "uscale every row by the same percentage "
                             "(default: bump)")
    parser.add_argument("--top-lanes",     type=int, default=0,
                        help="generate only: concentrate new bookings on the N "
                             "busiest OD lanes (0 = all lanes; default: 0)")
    parser.add_argument("--outlier-rate", type=float, default=0.1,
                        help="Fraction of rows that become outliers (default: 0.1)")
    parser.add_argument("--bump-mean",    type=float, default=0.5,
                        help="Mean bump on outlier rows as fraction of row Total (default: 0.5)")
    parser.add_argument("--bump-std",     type=float, default=0.15,
                        help="Std of bump on outlier rows as fraction of row Total (default: 0.15)")
    parser.add_argument("--seed",         type=int,   default=42, help="Random seed")
    parser.add_argument("--all", action="store_true",
                        help="Generate all scenarios for the chosen --selection method")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Original total containers: {int(df['Total'].sum()):,}  "
          f"(selection: {args.selection})")

    if args.all:
        for name, tgt in SCENARIOS:

            if name == "baseline" and (args.selection == "od-weighted"
                                       or args.method in ("generate", "scale", "uscale")):
                continue
            write_scenario(df, name, tgt, args.outlier_rate, args.bump_mean,
                           args.bump_std, args.selection, args.seed, args.method,
                           args.top_lanes)
    else:
        write_scenario(df, args.scenario, args.target, args.outlier_rate,
                       args.bump_mean, args.bump_std, args.selection, args.seed,
                       args.method, args.top_lanes)


if __name__ == "__main__":
    main()
