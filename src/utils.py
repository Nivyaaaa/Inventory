# src/utils.py

import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# ======================================================
# BASIC IO
# ======================================================
def save_json(data: Dict[str, Any], filepath: Path, pretty: bool = True):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2 if pretty else None, default=str)


def load_json(filepath: Path):
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(obj, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ======================================================
# CSV / XLS LOADERS (ROBUST)
# ======================================================
def load_csv_quoted(filepath: Path) -> pd.DataFrame:
    """
    Load CSV safely, including:
    - UTF-8 BOM
    - quoted headers
    - broken single-column CSVs
    """
    import csv

    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"File is empty: {filepath}")

    # Handle single-column CSV with commas inside
    if len(rows[0]) == 1 and "," in rows[0][0]:
        header = [c.strip() for c in rows[0][0].split(",")]
        data = [r[0].split(",") for r in rows[1:]]
        return pd.DataFrame(data, columns=header)

    return pd.DataFrame(rows[1:], columns=rows[0])


def load_future_data_flexible(filepath: Path) -> pd.DataFrame:
    """
    Load future demand input data:
    - Supports .csv or .xls extension
    - Handles CSV content saved as .xls
    - Fixes single-cell-per-row corruption
    """

    logger.info(f"Loading future data from: {filepath}")

    # --------------------------------------------------
    # 1. Try Excel first (if truly an Excel file)
    # --------------------------------------------------
    if filepath.suffix.lower() in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(filepath)
            df.columns = df.columns.str.strip()
            if df.shape[1] > 1:
                logger.info("[OK] Loaded real Excel future file")
                return df
        except Exception:
            logger.warning("[WARN] Not a real Excel file, falling back to CSV logic")

    # --------------------------------------------------
    # 2. Fallback: treat as CSV (even if extension is .xls)
    # --------------------------------------------------
    df = load_csv_quoted(filepath)
    df.columns = df.columns.str.strip()

    # --------------------------------------------------
    # 3. Final validation
    # --------------------------------------------------
    required = {
        "Date", "SKU", "Category",
        "Our_Price", "Comp_Price",
        "Promotion_Flag", "Discount_Depth"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Future data missing columns {missing}. Found: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in [
        "Our_Price", "Comp_Price",
        "Promotion_Flag", "Discount_Depth"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"[OK] Loaded {len(df)} future rows for forecasting")
    return df


# ======================================================
# VALIDATION
# ======================================================
def validate_dataframe(df: pd.DataFrame, required_cols: List[str], name="df"):
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def get_dataframe_stats(df: pd.DataFrame, name="df"):
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "nulls": df.isnull().sum().to_dict()
    }


# ======================================================
# DEMAND MODEL PLACEHOLDER
# ======================================================
def create_demand_model_input(*args, **kwargs):
    return {"note": "demand model metadata placeholder"}


# ======================================================
# TEMPORAL FEATURES
# ======================================================
def add_temporal_features(df: pd.DataFrame, date_col: str = "Date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["Month"] = df[date_col].dt.month
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    df["Quarter"] = df[date_col].dt.quarter
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsHolidaySeason"] = df["Month"].isin([11, 12]).astype(int)
    df["IsSummer"] = df["Month"].isin([7, 8]).astype(int)
    df["IsBackToSchool"] = df["Month"].isin([9, 10]).astype(int)

    return df


# ======================================================
# OPTIMIZATION JSON BUILDERS
# ======================================================
def create_optimization_input(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
    horizon: int
):
    return {
        "timestamp": datetime.now().isoformat(),
        "horizon_days": horizon,
        "demand_rows": len(demand_df),
        "inventory_rows": len(inventory_df),
        "supplier_rows": len(supplier_df)
    }


def create_optimization_output(results: Dict[str, Any], metrics: Dict[str, Any], status: str):
    return {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "metrics": metrics,
        "results": results
    }


# ======================================================
# CSV EXPORTS
# ======================================================
def export_demand_predictions_csv(json_data: Dict[str, Any], filepath: Path):
    df = pd.DataFrame(json_data["predictions"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def export_inventory_optimization_csv(json_data: Dict[str, Any], filepath: Path):
    rows = []
    for r in json_data["results"]["plan"]:
        rows.append({
            "Product ID": r["Product ID"],
            "OrderQty": r["OrderQty"],
            "OrderFlag": r["OrderFlag"]
        })

    df = pd.DataFrame(rows)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def export_inventory_optimization_detailed_csv(
    json_data: Dict[str, Any],
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
    filepath: Path
):
    df = pd.DataFrame(json_data["results"]["plan"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


print("[OK] Utils module loaded")
