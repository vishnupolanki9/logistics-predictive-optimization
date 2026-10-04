"""
Predictive Modeling and Optimization in Logistics Systems

Reproducible training/evaluation script based on a simulated logistics dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, KFold, cross_validate, RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulated_logistics_dataset.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    df = pd.read_csv(DATA)

    target = "delivery_time_hours"
    X = df.drop(columns=[target])
    y = df[target]

    categorical = [
        "traffic", "weather", "warehouse_congestion",
        "priority", "carrier"
    ]
    numeric = [
        "distance_km", "weight_kg", "stops", "dispatch_hour"
    ]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            loss="huber",
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    test_results = []
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", preprocess),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        test_results.append({
            "Model": name,
            "MAE (hours)": mean_absolute_error(y_test, pred),
            "RMSE (hours)": rmse(y_test, pred),
            "R²": r2_score(y_test, pred)
        })
        fitted[name] = pipe

    test_results = pd.DataFrame(test_results).sort_values("RMSE (hours)")
    test_results.to_csv(RESULTS / "model_test_results_reproduced.csv", index=False)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", preprocess),
            ("model", estimator)
        ])
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            },
            n_jobs=-1
        )
        cv_results.append({
            "Model": name,
            "CV MAE (hours)": -scores["test_mae"].mean(),
            "CV RMSE (hours)": -scores["test_rmse"].mean(),
            "CV R²": scores["test_r2"].mean()
        })

    pd.DataFrame(cv_results).sort_values("CV RMSE (hours)").to_csv(
        RESULTS / "cross_validation_results_reproduced.csv",
        index=False
    )

    # Hyperparameter tuning
    gb_pipe = Pipeline([
        ("prep", preprocess),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    param_dist = {
        "model__n_estimators": [150, 200, 250, 300, 400],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [2, 3, 5, 8],
        "model__subsample": [0.8, 1.0]
    }

    search = RandomizedSearchCV(
        gb_pipe,
        param_distributions=param_dist,
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    tuned_pred = search.best_estimator_.predict(X_test)
    print("\nTuned Gradient Boosting")
    print(f"MAE : {mean_absolute_error(y_test, tuned_pred):.3f}")
    print(f"RMSE: {rmse(y_test, tuned_pred):.3f}")
    print(f"R²  : {r2_score(y_test, tuned_pred):.3f}")
    print("\nBest parameters:")
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    # Illustrative optimization scenario
    scenario = X_test.copy()
    scenario["traffic"] = np.where(
        scenario["traffic"] == "High", "Medium", scenario["traffic"]
    )
    scenario["warehouse_congestion"] = np.where(
        scenario["warehouse_congestion"] == "High",
        "Medium",
        scenario["warehouse_congestion"]
    )
    scenario["stops"] = np.maximum(scenario["stops"] - 1, 0)

    baseline = search.best_estimator_.predict(X_test).mean()
    optimized = search.best_estimator_.predict(scenario).mean()
    reduction = baseline - optimized
    reduction_pct = reduction / baseline * 100

    print("\nOptimization scenario")
    print(f"Baseline predicted average: {baseline:.2f} hours")
    print(f"Optimized predicted average: {optimized:.2f} hours")
    print(f"Predicted reduction: {reduction:.2f} hours ({reduction_pct:.1f}%)")


if __name__ == "__main__":
    main()
