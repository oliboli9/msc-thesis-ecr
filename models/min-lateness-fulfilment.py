import math
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from datetime import timedelta


@dataclass
class ModelResults:
    flows_df:       pd.DataFrame   # DepPort, DepHour, ArrPort, ArrHour, Line, Vessel, Commodity, Flow, Cost
    inventory_df:   pd.DataFrame   # Port, Time, ContainerType, ContainerSize, Owner, InitialContainers, FinalContainers
    fulfillment_df: pd.DataFrame   # Commodity, Origin, Destination, Owner, Demand, Fulfilled, FillRate, Vessel, Status, LatenessDays, MeanLatenessDays
    stock_df:       pd.DataFrame   # Stock_01012023.csv format
    status:         str            # "OPTIMAL" | "INFEASIBLE" | "OTHER_{code}"


def solve(arcs_df: pd.DataFrame,
          commodities_df: pd.DataFrame,
          initial_supply: dict,
          cfg: dict) -> ModelResults:
    EPOCH_DT        = cfg["epoch_dt"]
    STOCK_CUTOFF    = cfg["stock_cutoff_hour"]
    HORIZON_END     = cfg["horizon_end"]

    def to_dt(h):
        return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

    arcs_df = arcs_df.copy()
    arcs_df["arc_id"] = arcs_df.index

    A = arcs_df["arc_id"].tolist()

    capacity = arcs_df.set_index("arc_id")["Capacity"].to_dict()
    arc_costs = arcs_df.set_index("arc_id")["Cost"].to_dict()

    arcs_df["dep_node"] = list(zip(arcs_df["DepPort"], arcs_df["DepHour"]))
    arcs_df["arr_node"] = list(zip(arcs_df["ArrPort"], arcs_df["ArrHour"]))

    nodes = set(arcs_df["dep_node"]).union(set(arcs_df["arr_node"]))
    N = list(nodes)

    out_arcs = {i: [] for i in N}
    in_arcs  = {i: [] for i in N}

    for _, row in arcs_df.iterrows():
        a = row["arc_id"]
        out_arcs[row["dep_node"]].append(a)
        in_arcs[row["arr_node"]].append(a)


    port_node_times_sorted = defaultdict(list)
    for port, h in nodes:
        port_node_times_sorted[port].append(h)
    for port in port_node_times_sorted:
        port_node_times_sorted[port].sort()


    commodities_df = commodities_df.copy()
    commodities_df["DepartureTime"] = commodities_df["DepartureTime"].astype(int)
    commodities_df["ArrivalTime"]   = commodities_df["ArrivalTime"].astype(int)
    commodities_df["Owner"]         = commodities_df["Owner"].str.title()

    if "Truncated" not in commodities_df.columns:
        commodities_df["Truncated"] = False
    commodities_df["Truncated"] = commodities_df["Truncated"].fillna(False).astype(bool)

    if "Commodity" not in commodities_df.columns:
        commodities_df["Commodity"] = [
            f"k{i}_{row.ContainerType}_{int(row.ContainerSize)}_{row.Owner}"
            for i, row in commodities_df.iterrows()
        ]

    teu          = {}
    commodity_spec = {}
    owner_of       = {}
    truncated_of   = {}

    for _, row in commodities_df.iterrows():
        k = row["Commodity"]
        size  = row["ContainerSize"]
        ctype = row["ContainerType"]

        if size == 20:
            teu[k] = 1
        elif size in [40, 45]:
            teu[k] = 2
        else:
            raise ValueError(f"Unknown container size: {size}")

        commodity_spec[k] = (ctype, size)
        owner_of[k]        = row["Owner"]
        truncated_of[k]    = bool(row["Truncated"])

    container_types = set(commodities_df["ContainerType"].unique())
    container_sizes = set(commodities_df["ContainerSize"].unique())
    for (_, ct, cs, eo) in initial_supply:
        container_types.add(ct)
        container_sizes.add(cs)
    container_types = sorted(container_types)
    container_sizes = sorted(container_sizes)
    container_specs = [(ct, cs) for ct in container_types for cs in container_sizes]
    teu_spec        = {(ct, cs): 1 if cs == 20 else 2 for ct, cs in container_specs}
    empty_owners    = ["Carrier"]

    K = commodities_df["Commodity"].tolist()

    origin = {
        row["Commodity"]: (row["Origin"], row["DepartureTime"])
        for _, row in commodities_df.iterrows()
    }
    dest_port_of = {
        row["Commodity"]: row["Destination"]
        for _, row in commodities_df.iterrows()
    }
    due_time = {
        row["Commodity"]: int(row["ArrivalTime"])
        for _, row in commodities_df.iterrows()
    }
    demand = {
        row["Commodity"]: int(row["Count"])
        for _, row in commodities_df.iterrows()
    }


    nodes_set = set(N)
    over_horizon = {
        k: (due_time[k] > HORIZON_END
            or (dest_port_of[k], due_time[k]) not in nodes_set)
        for k in K
    }


    max_delay_hours = cfg.get("max_delay_days", 0) * 24

    dest_times = {}
    for k in K:
        if over_horizon[k]:
            dest_times[k] = []
            continue
        p = dest_port_of[k]
        dep_h = origin[k][1]
        slots = [t for t in port_node_times_sorted.get(p, []) if t >= dep_h]
        if max_delay_hours > 0:
            slots = [t for t in slots if t <= due_time[k] + max_delay_hours]
        dest_times[k] = slots

    if cfg.get("fractional_lateness", False):
        late_days = {
            (k, t): max(0.0, (t - due_time[k]) / 24)
            for k in K for t in dest_times[k]
        }
    else:
        late_days = {
            (k, t): max(0, math.ceil((t - due_time[k]) / 24))
            for k in K for t in dest_times[k]
        }

  

    supply = dict(initial_supply)

    # ------------Gurobi ----------------------------------------------

    m = gp.Model("container_lateness")
    m.Params.MIPGap = cfg.get("mip_gap", 1e-4)

    m.Params.Seed = cfg.get("seed", 0)
    m.Params.Threads = 1

    x = m.addVars(A, K, lb=0, vtype=GRB.INTEGER, name="x")
    e = m.addVars(A, container_types, container_sizes, empty_owners,
                  lb=0, vtype=GRB.INTEGER, name="e")
    f_kt = m.addVars(
        [(k, t) for k in K for t in dest_times[k]],
        lb=0, vtype=GRB.INTEGER, name="f"
    )
    K_over = [k for k in K if over_horizon[k]]
    f_over = m.addVars(K_over, lb=0, vtype=GRB.INTEGER, name="f_over")
    I = m.addVars(N, container_types, container_sizes, empty_owners,
                  lb=0, vtype=GRB.INTEGER, name="inventory")

    def f_total(k):
        if over_horizon[k]:
            return f_over[k]
        return gp.quicksum(f_kt[k, t] for t in dest_times[k])

    lambda_cost = float(cfg.get("lambda_cost", 0.01))
    mu_lateness = float(cfg.get("mu_lateness", 0.1))

    mu_reefer = float(cfg.get("mu_reefer", 0.0))
    if mu_reefer > 0:
        def _is_reefer(k):
            ct = commodity_spec[k][0]
            return len(ct) >= 3 and ct[2].upper() == "R"
        mu_k = {k: (mu_reefer if _is_reefer(k) else mu_lateness) for k in K}
    else:
        mu_k = {k: mu_lateness for k in K}

    m.setObjective(
        gp.quicksum(f_total(k) for k in K)
        - gp.quicksum(
            mu_k[k] * late_days[k, t] * f_kt[k, t]
            for k in K for t in dest_times[k]
        )
        - lambda_cost * gp.quicksum(
            (gp.quicksum(x[a, k] for k in K)
             + gp.quicksum(e[a, ct, cs, eo]
                           for ct, cs in container_specs for eo in empty_owners)
            ) * arc_costs[a]
            for a in A
        ),
        GRB.MAXIMIZE
    )

    # Capacity constraints
    for a in A:
        m.addConstr(
            gp.quicksum(teu[k] * x[a, k] for k in K)
            + gp.quicksum(
                teu_spec[ct, cs] * e[a, ct, cs, eo]
                for ct, cs in container_specs for eo in empty_owners
            )
            <= capacity[a],
            name=f"cap_{a}"
        )

    # Cargo flow conservation 
    dest_time_set = {k: set(dest_times[k]) for k in K}

    for k in K:
        p_dest = dest_port_of[k]
        o_node = origin[k]
        is_over = over_horizon[k]

        for i in N:
            lhs = (gp.quicksum(x[a, k] for a in out_arcs[i])
                   - gp.quicksum(x[a, k] for a in in_arcs[i]))

            is_origin = (i == o_node)

            if is_over:
                if is_origin:
                    m.addConstr(lhs == f_total(k), name=f"flow_origin_oh_{k}_{i}")
                else:
                    m.addConstr(lhs <= 0, name=f"flow_inflight_oh_{k}_{i}")
                continue

            is_dest_slot = (i[0] == p_dest and i[1] in dest_time_set[k])

            if is_origin and is_dest_slot:
                m.addConstr(
                    lhs == f_total(k) - f_kt[k, i[1]],
                    name=f"flow_origin_dest_{k}_{i}"
                )
            elif is_origin:
                m.addConstr(lhs == f_total(k), name=f"flow_origin_{k}_{i}")
            elif is_dest_slot:
                m.addConstr(lhs == -f_kt[k, i[1]], name=f"flow_dest_{k}_{i}")
            else:
                m.addConstr(lhs == 0, name=f"flow_trans_{k}_{i}")

    # Demand caps
    for k in K:
        m.addConstr(f_total(k) <= demand[k], name=f"demand_{k}")

    # Delayed empty return (dwell time) 

    DWELL_HOURS = 168

    def delayed_return_node(port, delivery_hour):
        target = delivery_hour + DWELL_HOURS
        for h in port_node_times_sorted[port]:
            if h >= target:
                return (port, h)
        return None


    delayed_at = defaultdict(list)
    for k in K:
        if owner_of[k] != "Carrier":
            continue
        ct, cs = commodity_spec[k]
        p = dest_port_of[k]
        for t in dest_times[k]:
            ret_node = delayed_return_node(p, t)
            if ret_node is not None:
                delayed_at[(ret_node, ct, cs, "Carrier")].append((k, t))

    # Empty container balance
    for i in N:
        for ct, cs in container_specs:
            for eo in empty_owners:
                inbound_empty  = gp.quicksum(e[a, ct, cs, eo] for a in in_arcs[i])
                outbound_empty = gp.quicksum(e[a, ct, cs, eo] for a in out_arcs[i])

                loaded = gp.quicksum(
                    f_total(k) for k in K
                    if origin[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
                )

                if eo == "Carrier":
                    pairs = delayed_at.get((i, ct, cs, eo), [])
                    delivered = gp.quicksum(f_kt[k, t] for (k, t) in pairs)
                else:
                    delivered = 0  # Leased: no return to pool

                m.addConstr(
                    supply.get((i, ct, cs, eo), 0)
                    + inbound_empty
                    + delivered
                    ==
                    outbound_empty
                    + loaded
                    + I[i[0], i[1], ct, cs, eo],
                    name=f"empty_balance_{i}_{ct}_{cs}_{eo}"
                )

    # Solve 

    m.optimize()
    print(f"  GUROBI_RUNTIME_S={m.Runtime:.3f}")

    #  Results 

    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        m.computeIIS()
        m.write("results/model.ilp")
        print("IIS written to results/model.ilp")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, "INFEASIBLE")

    if m.Status != GRB.OPTIMAL:
        print(f"Optimization ended with status: {m.Status}")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, f"OTHER_{m.Status}")

    print("Optimal solution found.")


    fulfilled_total = {
        k: (f_over[k].X if over_horizon[k]
            else sum(f_kt[k, t].X for t in dest_times[k]))
        for k in K
    }
    lateness_total = {
        k: sum(late_days[k, t] * f_kt[k, t].X for t in dest_times[k]) for k in K
    }

    rows = []
    for a in A:
        arc      = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
        dep      = arc["DepPort"]
        arr      = arc["ArrPort"]
        dep_hour = arc["DepHour"]
        arr_hour = arc["ArrHour"]
        line     = arc["Line"]
        vessel   = arc["Vessel"]
        arc_cost = arc["Cost"]

        for k in K:
            if x[a, k].X > 1e-6:
                rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                             k, x[a, k].X, arc_cost])

        for ct, cs in container_specs:
            for eo in empty_owners:
                if e[a, ct, cs, eo].X > 1e-6:
                    rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                                 f"EMPTY_{ct}_{cs}_{eo}", e[a, ct, cs, eo].X, arc_cost])

    flows_df = pd.DataFrame(rows, columns=[
        "DepPort", "DepHour", "ArrPort", "ArrHour",
        "Line", "Vessel", "Commodity", "Flow", "Cost"
    ]).sort_values("DepHour").reset_index(drop=True)

    inventory_rows = []
    for i in N:
        port, time = i
        for ct, cs in container_specs:
            for eo in empty_owners:
                inventory_rows.append([
                    port, time, ct, cs, eo,
                    supply.get((i, ct, cs, eo), 0),
                    I[i[0], i[1], ct, cs, eo].X
                ])
    inventory_df = pd.DataFrame(inventory_rows, columns=[
        "Port", "Time", "ContainerType", "ContainerSize", "Owner",
        "InitialContainers", "FinalContainers"
    ])

    stock_totals: dict[tuple, float] = defaultdict(float)
    for port, h in nodes:
        if h > STOCK_CUTOFF:
            continue
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = I[port, h, ct, cs, eo].X
                if val > 1e-6:
                    stock_totals[(port, ct, cs, eo, "Empty")] += val

    wait_crossing_arcs = arcs_df[
        (arcs_df["DepPort"] == arcs_df["ArrPort"]) &
        (arcs_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (arcs_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]
    for _, arc in wait_crossing_arcs.iterrows():
        a = arc["arc_id"]
        port = arc["DepPort"]
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = e[a, ct, cs, eo].X
                if val > 1e-6:
                    stock_totals[(port, ct, cs, eo, "Empty")] += val

    vessel_totals: dict[tuple, float] = defaultdict(float)
    crossing_arcs = arcs_df[
        (arcs_df["DepPort"] != arcs_df["ArrPort"]) &
        (arcs_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (arcs_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]
    for _, arc in crossing_arcs.iterrows():
        a = arc["arc_id"]
        dep, arr = arc["DepPort"], arc["ArrPort"]
        arr_hour = int(arc["ArrHour"])
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = e[a, ct, cs, eo].X
                if val > 1e-6:
                    vessel_totals[(dep, arr, arr_hour, ct, cs, eo, "Empty")] += val

    cargo_on_vessel = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (flows_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]

    vessel_full_per_k: dict = (
        cargo_on_vessel.groupby("Commodity")["Flow"].sum().to_dict()
        if not cargo_on_vessel.empty else {}
    )
    for _, row in cargo_on_vessel.iterrows():
        k = row["Commodity"]
        ct, cs = commodity_spec[k]
        eo = owner_of[k]
        arr_hour = int(row["ArrHour"])
        vessel_totals[(row["DepPort"], row["ArrPort"], arr_hour, ct, cs, eo, "Full")] += row["Flow"]

    cargo_within_horizon = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["ArrHour"].astype(int) <= STOCK_CUTOFF)
    ].copy()

    port_full_totals: dict[tuple, float] = defaultdict(float)
    with_customer_totals: dict[tuple, float] = defaultdict(float)
    for k in K:
        if owner_of[k] != "Carrier":
            continue
        if origin[k][1] > STOCK_CUTOFF:
            continue  
        ct, cs = commodity_spec[k]
        p = dest_port_of[k]

        delivered_before_cutoff = 0.0
        for t in dest_times[k]:
            v = f_kt[k, t].X
            if v < 1e-6:
                continue
            if t <= STOCK_CUTOFF:
                ret_node = delayed_return_node(p, t)
                if ret_node is None or ret_node[1] > STOCK_CUTOFF:
                    return_hour = t + DWELL_HOURS
                    with_customer_totals[(p, ct, cs, return_hour)] += round(v)
   
                delivered_before_cutoff += v

        loaded = fulfilled_total[k]
        on_vessel = vessel_full_per_k.get(k, 0)
        in_transit = loaded - delivered_before_cutoff - on_vessel
        if in_transit < 1e-6:
            continue
        k_rows = cargo_within_horizon[cargo_within_horizon["Commodity"] == k]
        if k_rows.empty:
            current_port = origin[k][0]
        else:
            current_port = k_rows.loc[k_rows["ArrHour"].idxmax(), "ArrPort"]
        port_full_totals[(current_port, ct, cs)] += round(in_transit)

    stock_rows = []
    for (port, ct, cs, eo, fe), total in sorted(stock_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "FullEmpty": fe, "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": eo.upper(), "count": int(round(total)),
        })
    for (port, ct, cs), total in sorted(port_full_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "FullEmpty": "Full", "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": "CARRIER", "count": int(round(total)),
        })
    for (dep, arr, arr_hour, ct, cs, eo, fe), total in sorted(vessel_totals.items()):
        stock_rows.append({
            "Location": "VESSEL", "Last location": dep, "Next location": arr,
            "ArrivalTime": arr_hour,
            "FullEmpty": fe, "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": eo.upper(), "count": int(round(total)),
        })
    for (port, ct, cs, return_hour), total in sorted(with_customer_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "ArrivalTime": return_hour,
            "FullEmpty": "Empty", "With customer": "True",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": "CARRIER", "count": int(round(total)),
        })
    stock_df = pd.DataFrame(stock_rows)


    fulfillment_rows = []
    for k in K:
        origin_port, origin_time = origin[k]
        p_dest = dest_port_of[k]

        vessels_used = [arcs_df.loc[arcs_df["arc_id"] == a, "Vessel"].values[0]
                        for a in A if x[a, k].X > 1e-6]
        vessel = vessels_used[0] if vessels_used else ""

        fulfilled = fulfilled_total[k]
        lateness  = lateness_total[k]
        mean_late = (lateness / fulfilled) if fulfilled > 1e-6 else 0.0

        delivered_in_horizon = sum(
            f_kt[k, t].X for t in dest_times[k] if t <= STOCK_CUTOFF
        )

        if fulfilled < 1e-6:
            status = "Unfulfilled"
        elif truncated_of[k]:
            status = "WaitingTransship"
        elif delivered_in_horizon < 1e-6:
            status = "InTransit"
        else:
            status = "Delivered"

        fulfillment_rows.append([
            k, origin_port, p_dest, owner_of[k],
            demand[k], fulfilled, fulfilled / demand[k] if demand[k] else 0.0,
            vessel, status, lateness, mean_late
        ])

    fulfillment_df = pd.DataFrame(fulfillment_rows, columns=[
        "Commodity", "Origin", "Destination", "Owner",
        "Demand", "Fulfilled", "FillRate", "Vessel", "Status",
        "LatenessDays", "MeanLatenessDays"
    ])

    n_unfulfilled = (fulfillment_df["Fulfilled"] < fulfillment_df["Demand"]).sum()
    n_containers  = int((fulfillment_df["Demand"] - fulfillment_df["Fulfilled"]).sum())
    total_late_cd = float(fulfillment_df["LatenessDays"].sum())
    print(f"Unfulfilled demand: {n_unfulfilled} commodities, {n_containers} containers")
    print(f"Total lateness: {total_late_cd:.0f} container-days")


    DIAGNOSE = cfg.get("verbose_unfulfilled", True)

    if DIAGNOSE and n_unfulfilled > 0:
        TEU_SIZE = {20: 1, 40: 2, 45: 2}

        arc_teu = {}
        for a in A:
            arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
            if arc["DepPort"] == arc["ArrPort"]:
                continue
            teus = (sum(x[a, k].X * teu[k] for k in K if x[a, k].X > 1e-6)
                    + sum(e[a, ct, cs, eo].X * TEU_SIZE.get(int(cs), 1)
                          for ct in container_types for cs in container_sizes
                          for eo in empty_owners if e[a, ct, cs, eo].X > 1e-6))
            arc_key = (arc["DepPort"], arc["DepHour"], arc["ArrPort"], arc["ArrHour"])
            arc_teu[arc_key] = arc_teu.get(arc_key, 0) + teus

        arc_cap = {
            (row["DepPort"], row["DepHour"], row["ArrPort"], row["ArrHour"]): row["Capacity"]
            for _, row in arcs_df[arcs_df["DepPort"] != arcs_df["ArrPort"]].iterrows()
        }

        print("\n=== Unfulfilled Demand Diagnosis ===")
        for k in K:
            if fulfilled_total[k] >= demand[k] - 1e-6:
                continue

            shortfall = demand[k] - fulfilled_total[k]
            ct, cs = commodity_spec[k]
            origin_port, origin_time = origin[k]
            p_dest = dest_port_of[k]

            print(f"\n  {k}  ({origin_port} -> {p_dest})  "
                  f"Unfulfilled: {shortfall:.0f} x {ct} {cs}ft "
                  f"(due h{due_time[k]})")

            origin_arcs = arcs_df[
                (arcs_df["DepPort"] == origin_port) &
                (arcs_df["DepHour"] >= origin_time) &
                (arcs_df["DepPort"] != arcs_df["ArrPort"])
            ]
            if origin_arcs.empty:
                print(f"    -> No sailing from {origin_port} after h{origin_time}")
                continue

            for _, arc_row in origin_arcs.iterrows():
                akey = (arc_row["DepPort"], arc_row["DepHour"],
                        arc_row["ArrPort"], arc_row["ArrHour"])
                used = arc_teu.get(akey, 0)
                cap  = arc_cap.get(akey, arc_row["Capacity"])
                pct  = used / cap * 100 if cap > 0 else 0
                flag = " ** AT CAPACITY **" if used >= cap - 1 else f" ({pct:.0f}% full)"
                print(f"    Arc {arc_row['DepPort']} -> {arc_row['ArrPort']} "
                      f"via {arc_row['Vessel']}: {used:.0f}/{cap:.0f} TEUs{flag}")

            if owner_of[k] == "Carrier":
                sup = sum(v for (node, nct, ncs, neo), v in supply.items()
                          if node[0] == origin_port and nct == ct and ncs == cs and neo == "Carrier")
                total_demand_at_origin = sum(
                    demand[kk] for kk in K
                    if origin[kk] == origin[k]
                    and commodity_spec[kk] == (ct, cs)
                    and owner_of[kk] == "Carrier"
                )
                emp_flag = " ** INSUFFICIENT **" if sup < total_demand_at_origin else ""
                print(f"    Empty supply at port: {sup:.0f}  "
                      f"(total {ct} {cs}ft Carrier demand at node: "
                      f"{total_demand_at_origin:.0f}){emp_flag}")

    return ModelResults(flows_df, inventory_df, fulfillment_df, stock_df, "OPTIMAL")



if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, "pipeline")
    from config import EPOCH_HOUR_OFFSET, H, CALENDAR_WEEKS, HORIZON_MONDAY, START_YEAR, START_WEEK, ARCS_PATH, COMMODITIES_PATH

    arcs_df = pd.read_csv(ARCS_PATH)
    commodities_df = pd.read_csv(COMMODITIES_PATH)

    _v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
    _mask = False
    for _yr, _wk in CALENDAR_WEEKS:
        _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
    _etds = _v[_mask]["etdDateTime"]
    _epoch_dt = (_etds.min().normalize().to_pydatetime() if not _etds.empty
                 else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day))

    cfg = {
        "epoch_hour_offset": EPOCH_HOUR_OFFSET,
        "H": H,
        "horizon_end": EPOCH_HOUR_OFFSET + H,
        "stock_cutoff_hour": EPOCH_HOUR_OFFSET + H,
        "epoch_dt": _epoch_dt,
        "calendar_weeks": CALENDAR_WEEKS,
    }

    results = solve(arcs_df, commodities_df, initial_supply={}, cfg=cfg)

    if results.status == "OPTIMAL":
        Path("results/stock").mkdir(parents=True, exist_ok=True)
        results.flows_df.to_csv("results/model_flows.csv", index=False)
        results.inventory_df.to_csv("results/node_inventory.csv", index=False)
        results.fulfillment_df.to_csv("results/demand_fulfillment.csv", index=False)
        results.stock_df.to_csv(f"results/stock/{START_YEAR}w{START_WEEK}.csv", index=False)
        print("Results written to results/")
