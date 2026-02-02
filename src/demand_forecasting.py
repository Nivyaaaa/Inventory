# =============================================================================
# DEMAND FORECASTING MODULE
# -----------------------------------------------------------------------------
# Purpose:
# - Train an XGBoost demand forecasting model
# - Generate SKU-level daily demand predictions
# - Persist model + encoders + statistics
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG IMPORTS
# ======================================================
from .config import (
    DEMAND_MODEL_PARAMS,
    DEMAND_FEATURES,
    RANDOM_STATE,
    DEMAND_MODEL_PATH,
    DEMAND_ARTIFACTS_PATH
)

from .utils import (
    save_pickle,
    load_pickle,
    save_json,
    load_csv_quoted,
    validate_dataframe,
    create_demand_model_input
)

# ======================================================
# DEMAND FORECASTER CLASS
# ======================================================
class DemandForecaster:
    """
    XGBoost-based demand forecasting model.
    """

    def __init__(self):
        self.model = None
        self.features = DEMAND_FEATURES
        self.le_sku = None
        self.le_category = None
        self.category_stats = None
        self.sku_stats = None

    # --------------------------------------------------
    # LOAD & PREPROCESS SALES DATA
    # --------------------------------------------------
    def load_and_preprocess_data(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading sales data from {filepath}")

        df = load_csv_quoted(filepath)
        df.columns = df.columns.str.strip()

        validate_dataframe(
            df,
            required_cols=["Date", "SKU", "Category", "Our_Price", "Units_Sold"],
            name="Sales Data"
        )

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        numeric_cols = [
            "Our_Price",
            "Comp_Price",
            "Discount_Depth",
            "Units_Sold",
            "Promotion_Flag",
            "Month"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Temporal features
        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Quarter"] = df["Date"].dt.quarter
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["IsHolidaySeason"] = df["Month"].isin([11, 12]).astype(int)
        df["IsSummer"] = df["Month"].isin([7, 8]).astype(int)
        df["IsBackToSchool"] = df["Month"].isin([9, 10]).astype(int)

        # Price features
        df["Price_Ratio"] = df["Our_Price"] / df["Comp_Price"]
        df["Effective_Price"] = df["Our_Price"] * (1 - df["Discount_Depth"])
        df["Price_Discount_Interaction"] = df["Our_Price"] * df["Discount_Depth"]
        df["Price_Advantage"] = (df["Comp_Price"] - df["Our_Price"]).clip(lower=0)
        df["Promo_Discount_Interaction"] = df["Promotion_Flag"] * df["Discount_Depth"]
        df["Log_Our_Price"] = np.log1p(df["Our_Price"])
        df["Log_Comp_Price"] = np.log1p(df["Comp_Price"])

        logger.info(f"[OK] Loaded {len(df)} rows with {len(df.columns)} features")
        return df

    # --------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8):
        np.random.seed(RANDOM_STATE)
        train_idx, test_idx = [], []

        for m in df["Month"].unique():
            idx = df[df["Month"] == m].index.tolist()
            np.random.shuffle(idx)
            split = int(len(idx) * train_ratio)
            train_idx.extend(idx[:split])
            test_idx.extend(idx[split:])

        return df.loc[train_idx], df.loc[test_idx]

    # --------------------------------------------------
    # AGGREGATED STATS
    # --------------------------------------------------
    def compute_category_sku_stats(self, train_df: pd.DataFrame):
        self.category_stats = train_df.groupby("Category").agg(
            Category_Avg_Price=("Our_Price", "mean"),
            Category_Std_Price=("Our_Price", "std"),
            Category_Avg_Sales=("Units_Sold", "mean"),
            Category_Std_Sales=("Units_Sold", "std")
        ).reset_index()

        self.sku_stats = train_df.groupby("SKU").agg(
            SKU_Avg_Price=("Our_Price", "mean"),
            SKU_Std_Price=("Our_Price", "std"),
            SKU_Avg_Sales=("Units_Sold", "mean"),
            SKU_Std_Sales=("Units_Sold", "std")
        ).reset_index()

        logger.info("[OK] Category & SKU statistics computed")

    # --------------------------------------------------
    # ADD AGG FEATURES
    # --------------------------------------------------
    def add_aggregated_features(self, df: pd.DataFrame):
        df = df.merge(self.category_stats, on="Category", how="left")
        df = df.merge(self.sku_stats, on="SKU", how="left")

        df["Price_vs_Category_Avg"] = df["Our_Price"] / df["Category_Avg_Price"]
        df["Price_vs_SKU_Avg"] = df["Our_Price"] / df["SKU_Avg_Price"]
        df["Discount_Holiday_Interaction"] = df["Discount_Depth"] * df["IsHolidaySeason"]
        df["Price_Holiday_Interaction"] = df["Effective_Price"] * df["IsHolidaySeason"]

        return df

    # --------------------------------------------------
    # ENCODING
    # --------------------------------------------------
    def encode_categorical(self, train_df, test_df):
        self.le_sku = LabelEncoder()
        self.le_category = LabelEncoder()

        train_df["SKU_Encoded"] = self.le_sku.fit_transform(train_df["SKU"])
        test_df["SKU_Encoded"] = self.le_sku.transform(test_df["SKU"])

        train_df["Category_Encoded"] = self.le_category.fit_transform(train_df["Category"])
        test_df["Category_Encoded"] = self.le_category.transform(test_df["Category"])

        return train_df, test_df

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self, train_df, test_df):
        validate_dataframe(train_df, self.features + ["Units_Sold"], "Training Data")

        X_train = train_df[self.features]
        y_train = train_df["Units_Sold"]
        X_test = test_df[self.features]
        y_test = test_df["Units_Sold"]

        self.model = xgb.XGBRegressor(**DEMAND_MODEL_PARAMS)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        return {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        }

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------
    def save_model(self, raw_data=None):
        artifacts = {
            "model": self.model,
            "features": self.features,
            "le_sku": self.le_sku,
            "le_category": self.le_category,
            "category_stats": self.category_stats,
            "sku_stats": self.sku_stats
        }

        save_pickle(artifacts, DEMAND_ARTIFACTS_PATH)

        if raw_data is not None:
            save_json(
                create_demand_model_input(raw_data, model_config=DEMAND_MODEL_PARAMS),
                Path("data/outputs/demand_model_input.json")
            )

        logger.info("[OK] Demand model saved")

    def load_model(self):
        artifacts = load_pickle(DEMAND_ARTIFACTS_PATH)
        self.model = artifacts["model"]
        self.features = artifacts["features"]
        self.le_sku = artifacts["le_sku"]
        self.le_category = artifacts["le_category"]
        self.category_stats = artifacts["category_stats"]
        self.sku_stats = artifacts["sku_stats"]

        logger.info("[OK] Demand model loaded")

    # --------------------------------------------------
    # PREDICT (✅ FIXED)
    # --------------------------------------------------
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate demand predictions for future data.
        Ensures no negative demand.
        """

        for f in self.features:
            if f not in future_df.columns:
                logger.warning(f"Missing feature: {f} — filling with 0")
                future_df[f] = 0

        X = future_df[self.features]

        predictions = self.model.predict(X)

        # ✅ FIX: Pandas clip, not NumPy keyword
        future_df["Predicted_Demand"] = (
            pd.Series(predictions)
            .clip(lower=0)
            .values
        )

        logger.info(f"[OK] Generated {len(future_df)} predictions")
        return future_df
