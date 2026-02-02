# 📦 Inventory Optimization System — End-to-End Documentation

This document explains the complete inventory optimization workflow implemented in this project.  
It covers **what each step does, what each file is responsible for, what each field means, and how to train and run the system end-to-end** in a simple and practical manner.

---

## 🧠 High-Level Workflow

The system is divided into **four logical stages**:

1. Demand Model Training (one-time / optional)
2. Future Demand Forecasting (90 days)
3. Inventory Optimization using MILP
4. Output Generation (JSON and CSV reports)

> Demand model training is done **once**.  
> Daily or weekly runs reuse the **pre-trained model**.

---

## 📁 Project Structure (Key Files)

```
inventory_optimization/
│
├── data/
│   ├── raw/
│   │   ├── sales_data_2024_expanded.csv
│   │   ├── inventory_cleaned.csv
│   │   ├── supplier_master.csv
│   │   └── future_values_90.xls
│   │
│   └── outputs/
│       ├── demand_predictions_90d.json
│       ├── inventory_optimization_output.json
│       ├── inventory_optimization_output.csv
│       └── inventory_optimization_detailed_report.csv
│
├── models/
│   └── demand_model_artifacts.pkl
│
├── src/
│   ├── demand_forecasting.py
│   ├── milp_optimizer.py
│   ├── run_inventory_optimization.py
│   ├── utils.py
│   └── config.py
│
└── run_optimization_only.py
```

---

## ⚙️ Configuration (`src/config.py`)

This file controls **paths, model settings, and optimization switches**.

### Important Parameters

| Parameter | Description |
|--------|-------------|
| RAW_DATA_DIR | Location of raw input data |
| DEMAND_FORECAST_PERIOD | Forecast horizon (90 days) |
| USE_SOFT_CAPACITY | Allow warehouse overflow |
| CAPACITY_PENALTY | Penalty per unit overflow |
| SOLVER_TIME_LIMIT | MILP solver time limit |

This file acts as the **control panel** of the system.

---

## 🔹 STEP 1 — Demand Model Training (Optional)

**File:** `src/demand_forecasting.py`

### Purpose
Train an **XGBoost regression model** to predict daily demand per SKU.

### Input File
```
data/raw/sales_data_2024_expanded.csv
```

### Required Columns

| Column | Meaning |
|------|--------|
| Date | Transaction date |
| SKU | Product ID |
| Category | Product category |
| Our_Price | Selling price |
| Comp_Price | Competitor price |
| Discount_Depth | Discount percentage |
| Promotion_Flag | Promotion indicator |
| Units_Sold | Target variable |

### Output
```
models/demand_model_artifacts.pkl
```

Contains:
- Trained model
- Label encoders
- SKU & category statistics

---

## 🔹 STEP 2 — Demand Forecasting (Real Future Data)

**File:** `src/run_inventory_optimization.py`

### Purpose
Predict **daily demand for the next 90 days** using **real future assumptions**.

### Input File
```
data/raw/future_values_90.xls
```

> File may look like Excel but contain CSV data — loader handles both.

### Required Columns

| Column | Meaning |
|------|--------|
| Date | Future date |
| SKU | Product ID |
| Category | Category |
| Our_Price | Planned price |
| Comp_Price | Expected competitor price |
| Promotion_Flag | Promotion plan |
| Discount_Depth | Discount depth |

### Output
```
data/outputs/demand_predictions_90d.json
```

---

## 🔹 STEP 3 — Inventory Optimization (MILP)

**File:** `src/milp_optimizer.py`

### Purpose
Decide **how much to order for each SKU** to:
- Cover 90-day demand
- Maintain safety stock
- Respect MOQ constraints
- Minimize total cost

---

### Decision Variables

| Variable | Meaning |
|--------|--------|
| Q | Order quantity |
| K | Number of MOQ batches |
| Y | Order flag (0 or 1) |

---

### Constraints

**Demand Coverage**
```
Current Stock + Order Quantity ≥ 90D Demand + Safety Stock
```

**MOQ Enforcement**
```
Order Quantity = MOQ × Batches
```

**Capacity**
- Hard or soft depending on config

---

### Objective Function

Minimize:
```
Purchase Cost
+ Ordering Cost
+ Holding Cost
+ Capacity Penalty (if soft)
```

---

## 🔹 STEP 4 — Output Generation

**File:** `src/utils.py`

### Output Files

```
inventory_optimization_output.json
inventory_optimization_output.csv
inventory_optimization_detailed_report.csv
```

Each SKU record includes:
- Current Stock
- 90D Demand
- Safety Stock
- Required Target
- Order Quantity
- MOQ Batches
- Added Volume

---

## ▶️ How to Run the System

### Activate environment
```bash
source venv/bin/activate
```

### Run optimization
```bash
python run_optimization_only.py
```

### Success message
```
SUCCESS ✅ INVENTORY OPTIMIZATION COMPLETE
```

---

## ✅ How to Validate Output

For every SKU:
```
Current Stock + OrderQty ≥ Required Target
OrderQty % MOQ == 0
```

If both hold → solution is correct.

---

## 📌 Key Design Principles

- MILP guarantees global optimality
- Safety stock ensures service level
- MOQ avoids unrealistic orders
- Soft capacity prevents infeasible solutions
- Real future data increases business realism

---




