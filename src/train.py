"""
train.py — Train multiple models, tune hyperparameters, track with MLflow.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Tracking: MLflow logs params, metrics, and model artifacts for each run.

Usage:
    python src/train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_validate
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engineering import build_feature_matrix

warnings.filterwarnings("ignore")


# ── MODEL DEFINITIONS & HYPERPARAMETER GRIDS ────────────────────────────────

MODELS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
            "class_weight": ["balanced", None],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "scale_pos_weight": [1, 5, 8],
        },
    },
    "LightGBM": {
        "estimator": LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "is_unbalance": [True, False],
        },
    },
}


# ── TRAINING FUNCTION ───────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test, feature_names,
    n_iter=30, cv_folds=5
):
    """
    Train all models with RandomizedSearchCV, log to MLflow, return results.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    best_score = 0
    best_model = None
    best_name = None

    mlflow.set_experiment("olist-satisfaction-prediction")

    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=name):
            # Randomized search
            search = RandomizedSearchCV(
                estimator=cfg["estimator"],
                param_distributions=cfg["params"],
                n_iter=min(n_iter, _param_combinations(cfg["params"])),
                scoring="roc_auc",
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "cv_roc_auc_mean": search.best_score_,
            }

            # Business metric: top 20% risk capture rate
            threshold_idx = int(len(y_prob) * 0.8)
            top20_threshold = np.sort(y_prob)[threshold_idx]
            flagged = y_prob >= top20_threshold
            capture_rate = y_test[flagged].sum() / y_test.sum() if y_test.sum() > 0 else 0
            metrics["top20_capture_rate"] = capture_rate

            # Log to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name)

            # Print results
            print(f"  Best params: {search.best_params_}")
            for k, v in metrics.items():
                print(f"  {k:25s}: {v:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=["Positive", "Negative"]))
            print(f"  Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            results[name] = {
                "model": model,
                "metrics": metrics,
                "best_params": search.best_params_,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

            # Track best
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model
                best_name = name

    # ── Save best model ──
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    # Save feature names alongside model
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print(f"\n{'='*60}")
    print(f"  🏆 Best Model: {best_name} (ROC-AUC: {best_score:.4f})")
    print(f"  Saved to models/best_model.pkl")
    print(f"{'='*60}")

    return results, best_model, best_name


def _param_combinations(params):
    """Estimate total combinations in a param grid."""
    total = 1
    for v in params.values():
        total *= len(v)
    return total


# ── COMPARISON TABLE ────────────────────────────────────────────────────────

def results_summary(results: dict) -> pd.DataFrame:
    """Create a comparison DataFrame of all model metrics."""
    rows = []
    for name, res in results.items():
        row = {"model": name, **res["metrics"]}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    return df.round(4)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    print("Working directory:", os.getcwd())

    X_train, X_test, y_train, y_test, feat_names = build_feature_matrix()
    results, best_model, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_names
    )
    summary = results_summary(results)
    print("\n📊 Model Comparison:")
    print(summary.to_string())