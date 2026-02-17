"""
evaluate.py — Generate evaluation plots for model comparison.

Produces:
  - ROC curves for all models
  - Precision-Recall curves
  - Confusion matrix heatmaps
  - Model comparison bar chart

Usage:
    from src.evaluate import plot_all_evaluations
    plot_all_evaluations(results, y_test)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "LogisticRegression": "#3498db",
    "RandomForest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "LightGBM": "#9b59b6",
}


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    print(f"  💾 Saved {FIG_DIR}/{name}.png")


def plot_roc_curves(results: dict, y_test: np.ndarray):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS.get(name, "gray"),
                linewidth=2, label=f"{name} (AUC={score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    savefig("11_roc_curves")
    plt.show()


def plot_precision_recall_curves(results: dict, y_test: np.ndarray):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ap = average_precision_score(y_test, res["y_prob"])
        ax.plot(rec, prec, color=COLORS.get(name, "gray"),
                linewidth=2, label=f"{name} (AP={ap:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.4, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    savefig("12_pr_curves")
    plt.show()


def plot_confusion_matrices(results: dict, y_test: np.ndarray):
    """Plot confusion matrix for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Positive", "Negative"],
                    yticklabels=["Positive", "Negative"])
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    savefig("13_confusion_matrices")
    plt.show()


def plot_model_comparison(results: dict):
    """Bar chart comparing key metrics across models."""
    import pandas as pd

    metrics_to_plot = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    data = []
    for name, res in results.items():
        for m in metrics_to_plot:
            data.append({"Model": name, "Metric": m, "Value": res["metrics"][m]})
    comp = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    models = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    w = 0.8 / len(models)

    for i, name in enumerate(models):
        vals = comp[comp["Model"] == name]["Value"].values
        ax.bar(x + i * w, vals, w, label=name, color=COLORS.get(name, "gray"), edgecolor="white")

    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics_to_plot])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    savefig("14_model_comparison")
    plt.show()


def plot_top20_capture(results: dict, y_test: np.ndarray):
    """Show what % of negative reviews each model catches by flagging top 20% riskiest."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names, rates = [], []

    for name, res in results.items():
        y_prob = res["y_prob"]
        threshold_idx = int(len(y_prob) * 0.8)
        top20_threshold = np.sort(y_prob)[threshold_idx]
        flagged = y_prob >= top20_threshold
        capture = y_test[flagged].sum() / y_test.sum() if y_test.sum() > 0 else 0
        names.append(name)
        rates.append(capture * 100)

    bars = ax.bar(names, rates, color=[COLORS.get(n, "gray") for n in names], edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", fontweight="bold")

    ax.set_ylabel("% of Negative Reviews Captured")
    ax.set_title("Business Metric: Top 20% Risk Flag → Capture Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    savefig("15_top20_capture")
    plt.show()


def plot_all_evaluations(results: dict, y_test: np.ndarray):
    """Run all evaluation plots."""
    print("\n📊 Generating evaluation plots...\n")
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_model_comparison(results)
    plot_top20_capture(results, y_test)
    print("\n✅ All evaluation plots saved to reports/figures/")