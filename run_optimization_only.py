#!/usr/bin/env python
"""
Run Inventory Optimization WITHOUT retraining the demand model.

This script:
- Loads a pre-trained demand forecasting model
- Generates 90-day demand forecasts
- Runs MILP-based inventory optimization
- Saves JSON and CSV outputs

USAGE:
------
python run_optimization_only.py
"""

import sys
import logging
from pathlib import Path

# ======================================================
# FIX: Ensure project root is on PYTHONPATH
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================
# CONFIG IMPORTS
# ======================================================
from src.config import DEMAND_ARTIFACTS_PATH

# ======================================================
# INTERNAL IMPORTS
# ======================================================
from src.run_inventory_optimization import InventoryOptimizationWorkflow
from src.demand_forecasting import DemandForecaster

# ======================================================
# LOGGING SETUP
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("INVENTORY OPTIMIZATION (USING PRE-TRAINED MODEL)")
    print("=" * 80)

    # --------------------------------------------------
    # Step 0: Verify pre-trained model exists
    # --------------------------------------------------
    if not DEMAND_ARTIFACTS_PATH.exists():
        logger.error("Pre-trained demand model not found!")
        logger.error(f"Expected path: {DEMAND_ARTIFACTS_PATH}")
        logger.error("Run training pipeline first.")
        sys.exit(1)

    logger.info(f"[OK] Found pre-trained model: {DEMAND_ARTIFACTS_PATH}")

    # --------------------------------------------------
    # Initialize workflow
    # --------------------------------------------------
    workflow = InventoryOptimizationWorkflow()

    # --------------------------------------------------
    # Load pre-trained demand model
    # --------------------------------------------------
    logger.info("Loading demand forecasting model...")
    workflow.demand_model = DemandForecaster()
    workflow.demand_model.load_model()
    logger.info("[OK] Demand model loaded")

    print("\n" + "=" * 80)
    print("RUNNING STEPS: Forecast → Optimize → Save")
    print("=" * 80)

    # --------------------------------------------------
    # STEP 2: Demand Forecasting
    # --------------------------------------------------
    logger.info("STEP 2: Generating 90-day demand forecasts...")
    step2 = workflow.step2_forecast_demand()

    if step2["status"] != "success":
        logger.error(f"Step 2 failed: {step2.get('error')}")
        sys.exit(1)

    logger.info("[OK] Demand forecasting completed")

    # --------------------------------------------------
    # STEP 3: Inventory Optimization
    # --------------------------------------------------
    logger.info("STEP 3: Running inventory optimization...")
    step3 = workflow.step3_optimize_inventory()

    if step3["status"] != "success":
        logger.error(f"Step 3 failed: {step3.get('error')}")
        sys.exit(1)

    logger.info("[OK] Inventory optimization completed")

    # --------------------------------------------------
    # STEP 4: Save Outputs
    # --------------------------------------------------
    logger.info("STEP 4: Saving outputs...")
    step4 = workflow.step4_save_outputs()

    if step4["status"] != "success":
        logger.error(f"Step 4 failed: {step4.get('error')}")
        sys.exit(1)

    logger.info("[OK] Outputs saved successfully")

    # --------------------------------------------------
    # DONE
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("SUCCESS ✅ INVENTORY OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nOutputs available in:")
    print("  data/outputs/")
    print("    - demand_predictions_90d.json")
    print("    - inventory_optimization_input.json")
    print("    - inventory_optimization_output.json")
    print("    - inventory_optimization_output.csv")
    print("    - inventory_optimization_detailed_report.csv")
    print("    - optimization_summary.json")
    print()
