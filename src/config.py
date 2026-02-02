import os
from pathlib import Path

# ======================================================
# PROJECT PATHS
# ======================================================
# Root of the project (…/inventory_optimization)
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Ensure required directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ======================================================
# DATA FILES (RAW INPUTS)
# ======================================================
SALES_DATA_FILE = RAW_DATA_DIR / "sales_data_2024_expanded.csv"
INVENTORY_DATA_FILE = RAW_DATA_DIR / "inventory_cleaned.csv"
SUPPLIER_DATA_FILE = RAW_DATA_DIR / "supplier_master.csv"

# 🔥 REAL FUTURE DATA (USED IN STEP 2)
# NOTE:
# - File may be .xls but contain CSV-style text
# - Loader handles broken single-column rows
FUTURE_DATA_FILE = RAW_DATA_DIR / "future_values_90.xls"

# ======================================================
# MODEL ARTIFACT PATHS
# ======================================================
DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.json"
DEMAND_ARTIFACTS_PATH = MODELS_DIR / "demand_model_artifacts.pkl"

# ======================================================
# OUTPUT FILE PATHS
# ======================================================
DEMAND_PREDICTIONS_FILE = OUTPUT_DIR / "demand_predictions_90d.json"
INVENTORY_OPTIMIZATION_INPUT_FILE = OUTPUT_DIR / "inventory_optimization_input.json"
INVENTORY_OPTIMIZATION_OUTPUT_FILE = OUTPUT_DIR / "inventory_optimization_output.json"
OPTIMIZATION_SUMMARY_FILE = OUTPUT_DIR / "optimization_summary.json"

# ======================================================
# DEMAND FORECASTING CONFIGURATION
# ======================================================
DEMAND_FORECAST_PERIOD = 90  # days
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# XGBoost model parameters (training only)
DEMAND_MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

# Features expected by the trained model
DEMAND_FEATURES = [
    "SKU_Encoded", "Category_Encoded",
    "Our_Price", "Comp_Price", "Promotion_Flag",
    "Discount_Depth", "Month", "DayOfWeek", "Quarter",
    "IsWeekend", "IsHolidaySeason", "IsSummer", "IsBackToSchool",
    "Price_Ratio", "Effective_Price",
    "Price_Discount_Interaction", "Price_Advantage",
    "Promo_Discount_Interaction",
    "Log_Our_Price", "Log_Comp_Price",
    "Category_Avg_Price", "Category_Std_Price",
    "Category_Avg_Sales", "Category_Std_Sales",
    "SKU_Avg_Price", "SKU_Std_Price",
    "SKU_Avg_Sales", "SKU_Std_Sales",
    "Price_vs_Category_Avg", "Price_vs_SKU_Avg",
    "Discount_Holiday_Interaction", "Price_Holiday_Interaction",
]

# ======================================================
# INVENTORY OPTIMIZATION (MILP) CONFIGURATION
# ======================================================
OPTIMIZATION_HORIZON = 90  # must match forecast period

SERVICE_LEVEL_TARGET = 0.95  # informational (Z ≈ 1.65)

# ---- COSTS
HOLDING_COST_PER_UNIT_PER_DAY = 0.5
STOCKOUT_COST_PER_UNIT = 50.0
ORDERING_COST_DEFAULT = 500.0

# ======================================================
# 🚨 CAPACITY CONSTRAINT MODE (GLOBAL SWITCH)
# ======================================================
# True  → Soft capacity (penalty allowed)
# False → Hard capacity (strict)
USE_SOFT_CAPACITY = True

CAPACITY_PENALTY = 1000.0

# ======================================================
# SOLVER SETTINGS
# ======================================================
SOLVER_TIME_LIMIT = 300
SOLVER_THREAD_COUNT = 4
SOLVER_MIP_GAP = 0.01

# ======================================================
# LOG CONFIRMATION
# ======================================================
print(f"[OK] Configuration loaded from {__file__}")
print(f"  - Base Directory: {BASE_DIR}")
print(f"  - Data Directory: {DATA_DIR}")
print(f"  - Output Directory: {OUTPUT_DIR}")
print(f"  - Future Data File: {FUTURE_DATA_FILE}")
print(f"  - Soft Capacity Enabled: {USE_SOFT_CAPACITY}")
