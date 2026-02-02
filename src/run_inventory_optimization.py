#!/usr/bin/env python
"""
Inventory Optimization Workflow
================================

STEP 2 → Demand Forecasting (using REAL future data)
STEP 3 → MILP Inventory Optimization
STEP 4 → Save JSON & CSV Outputs

⚠️ Model training is intentionally excluded.
This workflow assumes a PRE-TRAINED demand model is already loaded.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np   # ✅ FIX 1: Proper NumPy import
import csv

# ======================================================
# CONFIG IMPORTS
# ======================================================
from .config import (
    INVENTORY_DATA_FILE,
    SUPPLIER_DATA_FILE,
    DEMAND_PREDICTIONS_FILE,
    INVENTORY_OPTIMIZATION_INPUT_FILE,
    INVENTORY_OPTIMIZATION_OUTPUT_FILE,
    DEMAND_FORECAST_PERIOD,
    RAW_DATA_DIR
)

# ======================================================
# INTERNAL MODULE IMPORTS
# ======================================================
from .milp_optimizer import InventoryOptimizer
from .utils import (
    save_json,
    load_json,
    add_temporal_features,
    create_optimization_input,
    create_optimization_output,
    load_csv_quoted,
    export_demand_predictions_csv,
    export_inventory_optimization_csv,
    export_inventory_optimization_detailed_csv
)

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryOptimizationWorkflow:
    """
    Orchestrates demand forecasting → inventory optimization → output export.
    """

    def __init__(self):
        self.demand_model = None
        self.optimizer = None
        self.demand_predictions = None
        self.optimization_results = None

    # ======================================================
    # STEP 2 — DEMAND FORECASTING (REAL FUTURE DATA)
    # ======================================================
    def step2_forecast_demand(self) -> Dict[str, Any]:

        logger.info("=" * 70)
        logger.info("STEP 2: GENERATING DEMAND FORECASTS (REAL FUTURE DATA)")
        logger.info("=" * 70)

        try:
            FUTURE_FILE = RAW_DATA_DIR / "future_values_90.xls"

            # --------------------------------------------------
            # SAFE LOAD: EXCEL FIRST, CSV FALLBACK
            # --------------------------------------------------
            try:
                future_df = pd.read_excel(FUTURE_FILE)
                logger.info("[OK] Loaded future data using read_excel()")
            except Exception:
                logger.warning("[WARN] Excel load failed, attempting CSV fallback")

                rows = []
                with open(FUTURE_FILE, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        rows.append(row)

                future_df = pd.DataFrame(rows[1:], columns=rows[0])

            # --------------------------------------------------
            # FIX BROKEN SINGLE-COLUMN CSV
            # --------------------------------------------------
            if len(future_df.columns) == 1 and "," in future_df.columns[0]:
                logger.warning("[FIX] Detected single-column future file. Splitting...")
                future_df = future_df.iloc[:, 0].str.split(",", expand=True)
                future_df.columns = [
                    "Date",
                    "SKU",
                    "Category",
                    "Our_Price",
                    "Comp_Price",
                    "Promotion_Flag",
                    "Discount_Depth"
                ]

            future_df.columns = future_df.columns.str.strip()

            # --------------------------------------------------
            # HARD VALIDATION
            # --------------------------------------------------
            required_cols = {
                "Date", "SKU", "Category",
                "Our_Price", "Comp_Price",
                "Promotion_Flag", "Discount_Depth"
            }
            missing = required_cols - set(future_df.columns)
            if missing:
                raise ValueError(f"Future data missing columns: {missing}")

            # --------------------------------------------------
            # TYPE CASTING
            # --------------------------------------------------
            future_df["Date"] = pd.to_datetime(future_df["Date"], errors="coerce")

            for col in ["Our_Price", "Comp_Price", "Discount_Depth", "Promotion_Flag"]:
                future_df[col] = pd.to_numeric(future_df[col], errors="coerce")

            # --------------------------------------------------
            # FEATURE ENGINEERING (MATCH TRAINING)
            # --------------------------------------------------
            future_df = add_temporal_features(future_df)

            future_df["Price_Ratio"] = future_df["Our_Price"] / future_df["Comp_Price"]
            future_df["Effective_Price"] = (
                future_df["Our_Price"] * (1 - future_df["Discount_Depth"])
            )
            future_df["Price_Discount_Interaction"] = (
                future_df["Our_Price"] * future_df["Discount_Depth"]
            )
            future_df["Price_Advantage"] = (
                future_df["Comp_Price"] - future_df["Our_Price"]
            ).clip(lower=0)

            future_df["Promo_Discount_Interaction"] = (
                future_df["Promotion_Flag"] * future_df["Discount_Depth"]
            )

            # ✅ FIX 2: Replace deprecated pd.np with NumPy
            future_df["Log_Our_Price"] = np.log1p(
                future_df["Our_Price"].clip(lower=0)
            )
            future_df["Log_Comp_Price"] = np.log1p(
                future_df["Comp_Price"].clip(lower=0)
            )

            # Aggregated features from training
            future_df = self.demand_model.add_aggregated_features(future_df)

            # Encode categoricals
            future_df["SKU_Encoded"] = self.demand_model.le_sku.transform(future_df["SKU"])
            future_df["Category_Encoded"] = self.demand_model.le_category.transform(
                future_df["Category"]
            )

            # --------------------------------------------------
            # PREDICT
            # --------------------------------------------------
            self.demand_predictions = self.demand_model.predict(future_df)

            save_json(
                {
                    "timestamp": datetime.now().isoformat(),
                    "forecast_period_days": DEMAND_FORECAST_PERIOD,
                    "predictions": self.demand_predictions.to_dict("records"),
                },
                DEMAND_PREDICTIONS_FILE
            )

            logger.info(f"[OK] Generated {len(self.demand_predictions)} demand forecasts")
            return {"status": "success"}

        except Exception as e:
            logger.error("[ERROR] Step 2 failed", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # ======================================================
    # STEP 3 — MILP INVENTORY OPTIMIZATION
    # ======================================================
    def step3_optimize_inventory(self) -> Dict[str, Any]:

        try:
            inventory_df = load_csv_quoted(INVENTORY_DATA_FILE)
            supplier_df = load_csv_quoted(SUPPLIER_DATA_FILE)

            optimizer_config = {
                "service_level_z": 1.65,
                "capacity_penalty": 1000.0,
                "solver_time_limit": 300,
                "use_soft_capacity": True
            }

            self.optimizer = InventoryOptimizer(config=optimizer_config)
            self.optimizer.load_data(
                self.demand_predictions,
                inventory_df,
                supplier_df
            )

            opt_data = self.optimizer.prepare_optimization_data()
            self.optimization_results = self.optimizer.optimize(opt_data)
            self.optimizer.metrics = self.optimization_results["summary"]

            logger.info("[OK] Inventory optimization completed")
            return {"status": "success"}

        except Exception as e:
            logger.error("[ERROR] Step 3 failed", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # ======================================================
    # STEP 4 — SAVE OUTPUTS
    # ======================================================
    def step4_save_outputs(self) -> Dict[str, Any]:

        try:
            inventory_df = load_csv_quoted(INVENTORY_DATA_FILE)
            supplier_df = load_csv_quoted(SUPPLIER_DATA_FILE)

            save_json(
                create_optimization_input(
                    self.demand_predictions,
                    inventory_df,
                    supplier_df,
                    DEMAND_FORECAST_PERIOD
                ),
                INVENTORY_OPTIMIZATION_INPUT_FILE
            )

            save_json(
                create_optimization_output(
                    self.optimization_results,
                    self.optimizer.metrics,
                    "success"
                ),
                INVENTORY_OPTIMIZATION_OUTPUT_FILE
            )

            export_demand_predictions_csv(
                load_json(DEMAND_PREDICTIONS_FILE),
                Path("data/outputs/demand_predictions_90d.csv")
            )

            export_inventory_optimization_csv(
                load_json(INVENTORY_OPTIMIZATION_OUTPUT_FILE),
                Path("data/outputs/inventory_optimization_output.csv")
            )

            export_inventory_optimization_detailed_csv(
                load_json(INVENTORY_OPTIMIZATION_OUTPUT_FILE),
                self.demand_predictions,
                inventory_df,
                supplier_df,
                Path("data/outputs/inventory_optimization_detailed_report.csv")
            )

            logger.info("[OK] All outputs saved successfully")
            return {"status": "success"}

        except Exception as e:
            logger.error("[ERROR] Step 4 failed", exc_info=True)
            return {"status": "failed", "error": str(e)}
