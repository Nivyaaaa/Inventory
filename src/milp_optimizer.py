# =============================================================================
# MILP INVENTORY OPTIMIZER (GLOBAL CAPACITY ONLY)
# =============================================================================
# - 90-day demand + safety stock coverage
# - MOQ enforcement
# - Optional SOFT or HARD GLOBAL capacity
# - NO category-level capacity constraints
# =============================================================================

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpInteger, LpBinary, LpStatus, value, PULP_CBC_CMD
)

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    MILP-based inventory optimizer with GLOBAL capacity constraint only.
    """

    # ------------------------------------------------------------------
    def __init__(self, config=None):
        """
        config keys:
        - service_level_z
        - solver_time_limit
        - capacity_penalty
        - use_soft_capacity
        """
        self.config = config or {}
        self.results = None
        self.metrics = None

    # ------------------------------------------------------------------
    def load_data(self, demand_df, inventory_df, supplier_df):
        self.demand_df = demand_df.copy()
        self.inventory_df = inventory_df.copy()
        self.supplier_df = supplier_df.copy()
        logger.info("[OK] Data loaded into MILP optimizer")

    # ------------------------------------------------------------------
    def prepare_optimization_data(self):

        # ----------------------------
        # Aggregate demand (90 days)
        # ----------------------------
        demand_stats = (
            self.demand_df
            .groupby("SKU")
            .agg(
                Daily_Avg_Demand=("Predicted_Demand", "mean"),
                Daily_Std_Demand=("Predicted_Demand", "std"),
                Total_90D_Demand=("Predicted_Demand", "sum")
            )
            .reset_index()
        )

        demand_stats["Daily_Std_Demand"] = demand_stats["Daily_Std_Demand"].fillna(
            np.maximum(0.15 * demand_stats["Daily_Avg_Demand"], 1.0)
        )

        demand_stats["Total_90D_Demand"] = (
            demand_stats["Total_90D_Demand"].round().astype(int)
        )

        # ----------------------------
        # Inventory normalization
        # ----------------------------
        inv = self.inventory_df.copy()
        inv.columns = inv.columns.str.strip()

        inv.rename(columns={
            "Product ID": "SKU",
            "Stock Levels": "Current_Stock",
            "MOQ": "SKU_MOQ",
            "Purchase_Cost": "Supplier_Cost"
        }, inplace=True)

        inv = inv.merge(demand_stats, on="SKU", how="left")
        inv = inv.merge(self.supplier_df, on="Category", how="left")

        # ----------------------------
        # Safe defaults
        # ----------------------------
        defaults = {
            "Current_Stock": 0,
            "Unit_Volume": 1.0,
            "Supplier_Cost": 0.0,
            "Holding_Cost": 0.1,
            "Ordering_Cost": 500.0,
            "Supplier_MOQ": 1,
            "SKU_MOQ": 1,
            "Supplier_Lead_Time": 30
        }

        for col, default in defaults.items():
            if col not in inv.columns:
                inv[col] = default
            inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(default)

        # Final MOQ per SKU
        inv["Final_MOQ"] = inv[["SKU_MOQ", "Supplier_MOQ"]].max(axis=1).astype(int)

        logger.info(f"[OK] Prepared optimization data for {len(inv)} SKUs")
        return inv

    # ------------------------------------------------------------------
    def optimize(self, inv):

        # ----------------------------
        # Config
        # ----------------------------
        Z = self.config.get("service_level_z", 1.65)
        TIME_LIMIT = self.config.get("solver_time_limit", 300)
        USE_SOFT_CAPACITY = self.config.get("use_soft_capacity", True)
        CAPACITY_PENALTY = self.config.get("capacity_penalty", 1000.0)

        inv["SKU"] = inv["SKU"].astype(str)
        SKUs = inv["SKU"].tolist()

        # ----------------------------
        # Dictionaries
        # ----------------------------
        stock = inv.set_index("SKU")["Current_Stock"].to_dict()
        demand = inv.set_index("SKU")["Total_90D_Demand"].to_dict()
        std = inv.set_index("SKU")["Daily_Std_Demand"].to_dict()
        lead = inv.set_index("SKU")["Supplier_Lead_Time"].to_dict()
        cost = inv.set_index("SKU")["Supplier_Cost"].to_dict()
        moq = inv.set_index("SKU")["Final_MOQ"].to_dict()
        holding = inv.set_index("SKU")["Holding_Cost"].to_dict()
        ordering = inv.set_index("SKU")["Ordering_Cost"].to_dict()
        volume = inv.set_index("SKU")["Unit_Volume"].to_dict()

        # Global capacity = total warehouse capacity
        GLOBAL_CAP = sum(inv["Unit_Volume"] * inv["Final_MOQ"]) * 10

        # Safety stock
        safety = {
            s: Z * std[s] * np.sqrt(max(lead[s], 1))
            for s in SKUs
        }

        # ----------------------------
        # MODEL
        # ----------------------------
        model = LpProblem("Inventory_Optimization_90D", LpMinimize)

        Q = {s: LpVariable(f"Q_{s}", lowBound=0, cat=LpInteger) for s in SKUs}
        K = {s: LpVariable(f"K_{s}", lowBound=0, cat=LpInteger) for s in SKUs}
        Y = {s: LpVariable(f"Y_{s}", 0, 1, LpBinary) for s in SKUs}

        # MOQ + demand coverage
        for s in SKUs:
            model += Q[s] == K[s] * moq[s]
            M = max(demand[s] * 2, 1000)
            model += Q[s] >= moq[s] * Y[s]
            model += Q[s] <= M * Y[s]
            model += stock[s] + Q[s] >= demand[s] + safety[s]

        # ----------------------------
        # GLOBAL CAPACITY
        # ----------------------------
        if USE_SOFT_CAPACITY:
            slack = LpVariable("Slack_Global", lowBound=0)
            model += (
                lpSum((stock[s] + Q[s]) * volume[s] for s in SKUs)
                <= GLOBAL_CAP + slack
            )
        else:
            slack = None
            model += (
                lpSum((stock[s] + Q[s]) * volume[s] for s in SKUs)
                <= GLOBAL_CAP
            )

        # ----------------------------
        # OBJECTIVE
        # ----------------------------
        objective = (
            lpSum(cost[s] * Q[s] for s in SKUs)
            + lpSum(ordering[s] * Y[s] for s in SKUs)
            + lpSum(holding[s] * ((stock[s] + Q[s]) / 2) for s in SKUs)
        )

        if USE_SOFT_CAPACITY:
            objective += CAPACITY_PENALTY * slack

        model += objective

        solver = PULP_CBC_CMD(msg=1, timeLimit=TIME_LIMIT)
        model.solve(solver)

        # ----------------------------
        # RESULTS
        # ----------------------------
        rows = []
        for s in SKUs:
            rows.append({
                "Product ID": s,
                "Current Stock": stock[s],
                "90D_Demand": demand[s],
                "Safety Stock": round(safety[s], 2),
                "Required Target": round(demand[s] + safety[s], 2),
                "OrderQty": int(value(Q[s]) or 0),
                "OrderFlag": int(value(Y[s]) or 0),
                "MOQ": moq[s],
                "Batches": int(value(K[s]) or 0),
                "Added Volume": round((value(Q[s]) or 0) * volume[s], 2)
            })

        self.results = {
            "status": LpStatus[model.status],
            "plan": rows,
            "summary": {
                "total_units_optimized": sum(r["OrderQty"] for r in rows),
                "total_cost_90days": round(value(model.objective), 2),
                "global_capacity_violation": round(value(slack), 2) if USE_SOFT_CAPACITY else 0,
                "timestamp": datetime.now().isoformat()
            }
        }

        return self.results
