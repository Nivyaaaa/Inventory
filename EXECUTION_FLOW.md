# Execution Flow Guide

## Step-by-Step Execution Flow

### **How the Workflow Runs**

When you execute `python main.py`, here's exactly what happens:

```
1. main.py
   └─> Imports InventoryOptimizationWorkflow from src.run_inventory_optimization
   └─> Creates workflow instance
   └─> Calls workflow.run()
       │
       ├─ STEP 1: step1_train_demand_model()
       │   │
       │   ├─> Initialize DemandForecaster
       │   │
       │   ├─> Load Data (sales_data_2024_expanded.csv)
       │   │   └─> Read CSV → 365+ days × 1000s of SKUs
       │   │
       │   ├─> Preprocess & Feature Engineering
       │   │   ├─> Parse dates
       │   │   ├─> Extract temporal features (month, day, dayofweek, etc.)
       │   │   ├─> Calculate price-based features
       │   │   └─> Create interaction features
       │   │
       │   ├─> Split Data (80/20 stratified by month)
       │   │   ├─> Training set: ~80% of data
       │   │   └─> Test set: ~20% of data
       │   │
       │   ├─> Compute Statistics
       │   │   ├─> Category-level aggregates
       │   │   └─> SKU-level aggregates
       │   │
       │   ├─> Add Aggregated Features
       │   │   ├─> Price vs category average
       │   │   ├─> Price vs SKU average
       │   │   └─> Discount×Holiday interactions
       │   │
       │   ├─> Encode Categorical Variables
       │   │   ├─> LabelEncode SKU
       │   │   └─> LabelEncode Category
       │   │
       │   ├─> Train XGBoost Model
       │   │   ├─> n_estimators: 500
       │   │   ├─> max_depth: 8
       │   │   ├─> learning_rate: 0.05
       │   │   └─> Process all features
       │   │
       │   ├─> Evaluate Model
       │   │   ├─> Predict on test set
       │   │   ├─> Calculate R², RMSE, MAE, MAPE
       │   │   └─> Log metrics
       │   │
       │   └─> Save Model & Artifacts
       │       ├─> Save demand_model.json
       │       ├─> Save demand_model_artifacts.pkl (with encoders)
       │       └─> ✓ Step 1 Complete
       │
       ├─ STEP 2: step2_forecast_demand()
       │   │
       │   ├─> Load Original Data (for statistics)
       │   │   └─> sales_data_2024_expanded.csv
       │   │
       │   ├─> Prepare Future Data
       │   │   ├─> Last date from original data = D
       │   │   ├─> Generate dates D+1 to D+90
       │   │   ├─> For each date, create record for every SKU
       │   │   ├─> Fill features with historical averages
       │   │   └─> Result: 90 days × num_skus records
       │   │
       │   ├─> Add Temporal Features
       │   │   ├─> Parse dates
       │   │   ├─> Extract month, day, dayofweek, etc.
       │   │   └─> Create seasonal flags
       │   │
       │   ├─> Add Aggregated Features
       │   │   ├─> Merge category statistics
       │   │   ├─> Merge SKU statistics
       │   │   └─> Create price ratios
       │   │
       │   ├─> Encode Categorical Variables
       │   │   ├─> Transform SKU using saved encoder
       │   │   └─> Transform Category using saved encoder
       │   │
       │   ├─> Generate Predictions
       │   │   ├─> Load trained model
       │   │   ├─> Select 32 features per record
       │   │   ├─> Predict Units_Sold for each record
       │   │   ├─> Rename to Predicted_Demand
       │   │   └─> Clip to non-negative values
       │   │
       │   ├─> Calculate Summary Statistics
       │   │   ├─> Total predicted units
       │   │   ├─> Average daily demand
       │   │   ├─> Max/min daily demand
       │   │   └─> Standard deviation
       │   │
       │   └─> Save Predictions
       │       ├─> Create JSON structure
       │       ├─> Include all predictions + metadata
       │       ├─> Save to demand_predictions_90d.json
       │       └─> ✓ Step 2 Complete
       │
       ├─ STEP 3: step3_optimize_inventory()
       │   │
       │   ├─> Load Additional Data
       │   │   ├─> inventory_cleaned.csv
       │   │   └─> supplier_master.csv
       │   │
       │   ├─> Validate Data
       │   │   ├─> Check required columns exist
       │   │   ├─> Check for missing values
       │   │   └─> Log validation results
       │   │
       │   ├─> Initialize InventoryOptimizer
       │   │   ├─> Set config parameters
       │   │   │   ├─> holding_cost: 0.5 per unit/day
       │   │   │   ├─> service_level: 0.95 (95%)
       │   │   │   └─> safety_factor: 1.5
       │   │   └─> Create optimizer instance
       │   │
       │   ├─> Prepare Optimization Data
       │   │   ├─> Aggregate demand by SKU & Date
       │   │   ├─> Load current inventory levels
       │   │   ├─> Load supplier info (lead time, cost)
       │   │   └─> Create optimization data structure
       │   │
       │   ├─> Run Optimization For Each SKU
       │   │   │
       │   │   └─> For each SKU in unique_skus:
       │   │       ├─> Filter demand data for SKU
       │   │       ├─> Calculate average demand
       │   │       ├─> Calculate demand std deviation
       │   │       ├─> Calculate Safety Stock
       │   │       │   └─> safety_stock = 1.5 × std_demand
       │   │       ├─> Calculate Reorder Point
       │   │       │   └─> ROP = (avg_demand × lead_time) + safety_stock
       │   │       ├─> Calculate EOQ
       │   │       │   └─> EOQ = sqrt((2×annual_demand×order_cost) / holding_cost)
       │   │       ├─> Calculate Optimal Inventory
       │   │       │   └─> optimal = ROP + (EOQ/2)
       │   │       ├─> Calculate Daily Holding Cost
       │   │       │   └─> daily_cost = optimal × 0.5
       │   │       ├─> Store Results
       │   │       │   ├─> optimized_inventory[SKU] = optimal
       │   │       │   ├─> reorder_points[SKU] = ROP
       │   │       │   ├─> order_quantities[SKU] = EOQ
       │   │       │   └─> cost_analysis[SKU] = costs
       │   │       └─> Repeat for next SKU
       │   │
       │   ├─> Generate Summary Metrics
       │   │   ├─> Total units across all SKUs
       │   │   ├─> Total 90-day holding cost
       │   │   ├─> Average cost per unit
       │   │   └─> Number of SKUs optimized
       │   │
       │   ├─> Generate Recommendations
       │   │   └─> For each SKU:
       │   │       ├─> Compare optimized vs current
       │   │       ├─> If >20% increase: RECOMMEND INCREASE
       │   │       ├─> If <-20% decrease: RECOMMEND DECREASE
       │   │       └─> Else: NO ACTION NEEDED
       │   │
       │   └─> ✓ Step 3 Complete
       │
       └─ STEP 4: step4_save_outputs()
           │
           ├─> Create Optimization Input JSON
           │   ├─> Add demand summary & data
           │   ├─> Add inventory summary & data
           │   ├─> Add supplier summary & data
           │   └─> Save to inventory_optimization_input.json
           │
           ├─> Create Optimization Output JSON
           │   ├─> Add optimization results
           │   ├─> Add metrics
           │   ├─> Add execution status
           │   └─> Save to inventory_optimization_output.json
           │
           ├─> Create Summary Report
           │   ├─> Workflow timestamp
           │   ├─> Status of each step
           │   ├─> Key metrics
           │   ├─> File locations
           │   └─> Save to optimization_summary.json
           │
           ├─> Log Output Locations
           │   ├─> Print file paths
           │   ├─> Print summary metrics
           │   └─> Confirm success
           │
           └─> ✓ Step 4 Complete

2. Return Results
   └─> Return overall status: "completed"

3. main.py Prints Final Summary
   ├─> Status: COMPLETED
   ├─> Files created (list all 4 JSON files)
   └─> Exit with code 0 (success)
```

## Data Transformations

### **Sales Data → Features (Step 1)**

```
sales_data_2024_expanded.csv (1000s of rows)
        │
        ├─ Temporal Extraction
        │  ├─ Month, Day, DayOfWeek, Quarter
        │  ├─ WeekOfYear, IsWeekend
        │  └─ Holiday season flags
        │
        ├─ Price Feature Engineering
        │  ├─ Price_Ratio = Our_Price / Comp_Price
        │  ├─ Effective_Price = Our_Price × (1 - Discount)
        │  ├─ Price_Discount_Interaction
        │  └─ Log transformations
        │
        ├─ Category/SKU Statistics
        │  ├─ Category_Avg_Price
        │  ├─ SKU_Std_Sales
        │  └─ All stat combinations
        │
        └─> 32 Final Features Ready for Training
```

### **Predictions → Optimization (Step 2-3)**

```
Demand Predictions (90 days × num_skus)
        │
        ├─ Aggregate by SKU
        │  └─> Total demand per SKU for 90 days
        │
        ├─ Integrate Inventory Data
        │  └─> Current levels per SKU
        │
        ├─ Integrate Supplier Data
        │  ├─> Lead times per SKU
        │  └─> Order costs per SKU
        │
        └─> Optimize Each SKU
           ├─> Safety stock calculation
           ├─> Reorder point determination
           ├─> EOQ computation
           └─> Cost analysis
```

## Key Algorithm Details

### **Demand Forecasting (Step 1)**

**Input:** Historical sales data with features
**Algorithm:** XGBoost Regression
**Output:** Predicted daily units for each product

```python
model = XGBRegressor(
    n_estimators=500,      # 500 decision trees
    learning_rate=0.05,    # Small steps for stability
    max_depth=8,           # Tree complexity
    subsample=0.8,         # 80% data per tree
    colsample_bytree=0.8   # 80% features per tree
)
```

### **Inventory Optimization (Step 3)**

**For each SKU:**

1. **Safety Stock:**
   ```
   SS = safety_factor × σ(demand)
   ```

2. **Reorder Point:**
   ```
   ROP = (μ(demand) × lead_time) + SS
   ```

3. **Economic Order Quantity:**
   ```
   EOQ = √(2DS/H)
   where:
     D = annual demand
     S = order cost per order
     H = holding cost per unit per year
   ```

4. **Optimal Inventory Level:**
   ```
   Q* = ROP + (EOQ/2)
   ```

5. **Daily Cost:**
   ```
   Daily Cost = Q* × holding_cost_per_day
   ```

## Time Complexity

- **Step 1:** O(n log n) for training (n = num records) + O(n) for evaluation
- **Step 2:** O(n) where n = 90 days × num_skus
- **Step 3:** O(k×m) where k = num_skus, m = forecast period (90)
- **Step 4:** O(k + m) for JSON serialization

**Total Time:** Usually 1-3 minutes for typical datasets

## File I/O Operations

```
DISK READS:
├─ sales_data_2024_expanded.csv        (once in Step 1)
├─ sales_data_2024_expanded.csv        (again in Step 2 - for stats)
├─ inventory_cleaned.csv               (Step 3)
└─ supplier_master.csv                 (Step 3)

DISK WRITES:
├─ models/demand_model.json            (Step 1)
├─ models/demand_model_artifacts.pkl   (Step 1)
├─ data/outputs/demand_predictions_90d.json             (Step 2)
├─ data/outputs/inventory_optimization_input.json       (Step 4)
├─ data/outputs/inventory_optimization_output.json      (Step 4)
└─ data/outputs/optimization_summary.json               (Step 4)
```

## Memory Usage

```
During Step 1:
├─ Original data: ~500 MB
├─ Features: +32 new columns
├─ Train/test split: 2 copies
└─ Total: ~2-3 GB

During Step 2:
├─ Future data (90×skus): ~100 MB
├─ Predictions: +1 column
└─ Total: ~200 MB

During Step 3:
├─ Aggregated data: ~50 MB
├─ Results: ~20 MB
└─ Total: ~100 MB
```

## Error Handling

At each step, if an error occurs:
1. Exception is caught
2. Error message is logged
3. Status is set to 'failed'
4. Workflow stops (doesn't proceed to next step)
5. User sees detailed error message

## Logging

Every major action is logged with:
- Timestamp
- Component name
- Log level (INFO/WARNING/ERROR)
- Message and details

Example logs:
```
2026-01-07 10:15:23 - root - INFO - Loading data from data/raw/sales_data_2024_expanded.csv
2026-01-07 10:15:45 - root - INFO - ✓ Loaded 50000 records with 50 features
2026-01-07 10:16:30 - root - INFO - Train: 40000, Test: 10000
2026-01-07 10:20:15 - root - INFO - ✓ Model trained - R²: 0.8234, RMSE: 12.45
...
```

This execution flow is deterministic and repeatable, making it ideal for production deployments and automation!
