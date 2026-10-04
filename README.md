# Predictive Modeling and Optimization in Logistics Systems

A Python-based logistics analytics project that forecasts shipment delivery time and translates predictive insights into operational optimization strategies.

## Project Overview

This project demonstrates an end-to-end machine learning workflow for a logistics forecasting problem:

1. Simulate a realistic shipment dataset.
2. Prepare numerical and categorical logistics features.
3. Train and compare Linear Regression, Random Forest, and Gradient Boosting models.
4. Evaluate models using MAE, RMSE, and R².
5. Apply 5-fold cross-validation.
6. Tune Gradient Boosting hyperparameters.
7. Perform a what-if optimization scenario.
8. Generate evaluation charts and a detailed report.

> **Important:** The dataset is simulated. Model performance figures are illustrative and should not be interpreted as results from a real logistics company.

## Target Variable

`delivery_time_hours` — predicted shipment delivery duration in hours.

## Main Features

- `distance_km`
- `traffic`
- `weather`
- `weight_kg`
- `stops`
- `dispatch_hour`
- `warehouse_congestion`
- `priority`
- `carrier`

## Results

The baseline model comparison showed Gradient Boosting as the strongest of the tested models on the simulated test set.

The tuned Gradient Boosting model achieved approximately:

- MAE: 2.30 hours
- RMSE: 2.88 hours
- R²: 0.708

A model-based optimization scenario combining reduced traffic exposure, lower warehouse congestion, and one fewer feasible stop produced an illustrative predicted reduction of approximately 14.4% in average delivery time.

## Project Structure

```text
logistics-predictive-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   └── logistics_model.py
├── data/
│   └── simulated_logistics_dataset.csv
├── results/
│   ├── model_test_results.csv
│   ├── cross_validation_results.csv
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── optimization_scenario.png
└── report/
    └── Predictive_Modeling_and_Optimization_in_Logistics_Systems.docx
```

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd logistics-predictive-optimization

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Model

```bash
python src/logistics_model.py
```

The script trains the models, evaluates them, performs cross-validation and hyperparameter tuning, and runs the optimization scenario.

## Evaluation Metrics

### MAE
Mean Absolute Error measures the average absolute prediction error in hours.

### RMSE
Root Mean Squared Error penalizes larger prediction errors more strongly.

### R²
R² measures the proportion of target variability explained by the model.

## Optimization Approach

The predictive model is used as a decision-support component. Potential operational levers include:

- Dynamic dispatch timing
- Warehouse workload balancing
- Stop consolidation
- Carrier allocation
- Delivery exception management
- Capacity planning

A production implementation should combine the prediction model with a constrained optimization engine that considers vehicle capacity, driver availability, time windows, service-level agreements, route constraints, and logistics cost.

## Limitations

The data is simulated and therefore does not represent real operational performance. A real deployment would require historical shipment records, timestamped delivery milestones, traffic/GPS information, warehouse processing times, carrier performance, and other relevant operational variables.

For real time-series deployment, time-based train/validation/test splits should generally be preferred over random splitting to reduce temporal leakage.

## Academic Use

This repository is suitable as a project submission demonstrating predictive modeling, model evaluation, Python implementation, and logistics optimization.

