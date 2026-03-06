"""
model/evaluate.py
══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS
    Comprehensive model evaluation and SHAP explainability.
    Generates all metric tables and plots after training runs.

CALLED FROM
    model/train.py   evaluate_regression, evaluate_classification,
                      plot_feature_importance, compute_shap,
                      print_final_summary

CALLS INTO
    model/config.py   PLOT_DIR, SHAP_SAMPLE_SIZE, SHAP_TOP_N_FEATURES
    matplotlib, seaborn, sklearn, shap
    Nothing else from this project.

WHAT IT PROVIDES
    evaluate_regression(model, X_test, y_test) → dict
        Metrics: R², MAE, RMSE, MAPE
        Plots:   actual vs predicted, residual distribution,
                 residuals vs predicted

    evaluate_classification(model, X_test, y_test) → dict
        Metrics: ROC-AUC, PR-AUC, F1, Precision, Recall, Accuracy
        Plots:   ROC curve, PR curve, confusion matrix, calibration curve

    plot_feature_importance(reg_model, clf_model, feature_names)
        Gain + split importance for both models.
        Saves: outputs/plots/feature_importance_*.png

    compute_shap(model, X_sample, task)
        Beeswarm, bar, dependence plots for top features.
        Saves: outputs/plots/shap_*.png

    print_final_summary(reg_metrics, clf_metrics, training_time)

MIGRATED FROM
    satellite_lgbm/evaluate.py  — replace  import config
                                   with     from model import config
══════════════════════════════════════════════════════════════════════════════
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from typing import List, Dict

from model import config

logger = logging.getLogger(__name__)

# ── Plot style ──────────────────────────────────────────────────────────────
PALETTE   = ["#0D47A1", "#1976D2", "#42A5F5", "#90CAF9", "#E3F2FD"]
ACCENT    = "#FF6F00"
BG        = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})


def _save(fig, name: str):
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    path = os.path.join(config.PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(model, X_test, y_test) -> Dict:
    """Full regression evaluation with plots."""
    y_pred = model.predict(X_test)

    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
        "MAPE": np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-9))) * 100,
    }

    logger.info("\n" + "="*50)
    logger.info("  REGRESSION TEST METRICS")
    logger.info("="*50)
    for k, v in metrics.items():
        logger.info(f"  {k:<8}: {v:.6f}")

    # ── Plot 1: Actual vs Predicted ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regression Evaluation — Risk Score Prediction", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.2, s=8, color=PALETTE[1])
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT, lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Risk Score")
    ax.set_ylabel("Predicted Risk Score")
    ax.set_title(f"Actual vs Predicted  (R²={metrics['R2']:.4f})")
    ax.legend()

    # ── Plot 2: Residuals distribution ───────────────────────────────────
    residuals = y_test.values - y_pred
    ax = axes[1]
    ax.hist(residuals, bins=80, color=PALETTE[1], edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=ACCENT, lw=1.5, linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution  (RMSE={metrics['RMSE']:.4f})")
    _save(fig, "regression_evaluation.png")

    # ── Plot 3: Residuals vs Predicted (heteroskedasticity check) ────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.2, s=6, color=PALETTE[2])
    ax.axhline(0, color=ACCENT, lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted — Heteroskedasticity Check")
    _save(fig, "residuals_vs_predicted.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification(model, X_test, y_test) -> Dict:
    """Full classification evaluation with plots."""
    y_proba = model.predict_proba(X_test)
    y_pred  = model.predict(X_test)

    metrics = {
        "ROC_AUC":    roc_auc_score(y_test, y_proba),
        "PR_AUC":     average_precision_score(y_test, y_proba),
        "F1":         f1_score(y_test, y_pred),
        "Threshold":  model.optimal_threshold,
    }

    logger.info("\n" + "="*50)
    logger.info("  CLASSIFICATION TEST METRICS")
    logger.info("="*50)
    for k, v in metrics.items():
        logger.info(f"  {k:<12}: {v:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred,
                 target_names=["Normal", "High-Risk"]))

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    fig.suptitle("Classification Evaluation — High-Risk Collision Alert",
                 fontsize=14, fontweight="bold")

    # ── ROC Curve ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax = fig.add_subplot(gs[0])
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
            label=f"AUC = {metrics['ROC_AUC']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=10)

    # ── Precision-Recall Curve ────────────────────────────────────────────
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_proba)
    ax = fig.add_subplot(gs[1])
    ax.plot(recall_arr, precision_arr, color=PALETTE[1], lw=2,
            label=f"AP = {metrics['PR_AUC']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=10)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    ax  = fig.add_subplot(gs[2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "High-Risk"],
                yticklabels=["Normal", "High-Risk"],
                linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\n(threshold={metrics['Threshold']:.2f})")

    # ── Calibration Curve ─────────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=15)
    ax = fig.add_subplot(gs[3])
    ax.plot(prob_pred, prob_true, marker="o", color=PALETTE[0],
            lw=2, label="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=10)

    _save(fig, "classification_evaluation.png")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str = "Feature Importance",
    filename: str = "feature_importance.png",
    top_n: int = 30,
):
    """Plot LightGBM native feature importances (gain + split)."""
    imp_gain  = model.model.feature_importance(importance_type="gain")
    imp_split = model.model.feature_importance(importance_type="split")

    df = pd.DataFrame({
        "feature": feature_names,
        "gain":    imp_gain,
        "split":   imp_split,
    }).sort_values("gain", ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.28)))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, col, label in zip(axes, ["gain", "split"], ["Gain (total information)", "Split count"]):
        df_sorted = df.sort_values(col)
        colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(df_sorted)))
        ax.barh(df_sorted["feature"], df_sorted[col], color=colors)
        ax.set_xlabel(label)
        ax.set_title(f"Top {top_n} Features by {label}")

    _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    model,
    X_sample: pd.DataFrame,
    task: str = "regression",
    top_n: int = config.SHAP_TOP_N_FEATURES,
):
    """
    Compute SHAP values and produce:
      1. Beeswarm summary plot
      2. Bar importance plot
      3. Dependence plots for top 3 features
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis. Run: pip install shap")
        return None

    logger.info(f"[SHAP] Computing values on {len(X_sample)} samples...")

    explainer   = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification lgb returns list [neg, pos] — take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── 1. Beeswarm summary ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=top_n,
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Feature Impact — {task.title()}", fontsize=13, fontweight="bold")
    _save(plt.gcf(), f"shap_beeswarm_{task}.png")

    # ── 2. Bar importance ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Mean |Value| — {task.title()}", fontsize=13, fontweight="bold")
    _save(plt.gcf(), f"shap_bar_{task}.png")

    # ── 3. Top-3 dependence plots ─────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    top3_feats = [X_sample.columns[i] for i in top3_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"SHAP Dependence — Top 3 Features ({task.title()})",
                 fontsize=13, fontweight="bold")
    for ax, feat in zip(axes, top3_feats):
        shap.dependence_plot(
            feat, shap_values, X_sample,
            ax=ax, show=False,
            dot_size=10, alpha=0.4,
        )
        ax.set_title(feat, fontsize=11)
    _save(fig, f"shap_dependence_{task}.png")

    logger.info("[SHAP] All plots saved.")
    return shap_values


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED METRICS SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(reg_metrics: Dict, clf_metrics: Dict,
                        reg_cv: Dict = None, clf_cv: Dict = None):
    """Print a clean final metrics summary table."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("  FINAL MODEL PERFORMANCE SUMMARY")
    print(sep)

    print("\n  ── Regression (Risk Score Prediction) ──────────────")
    for k, v in reg_metrics.items():
        print(f"    {k:<10}: {v:.6f}")
    if reg_cv:
        print(f"    CV R²     : {reg_cv['R2']['mean']:.4f} ± {reg_cv['R2']['std']:.4f}")
        print(f"    CV RMSE   : {reg_cv['RMSE']['mean']:.4f} ± {reg_cv['RMSE']['std']:.4f}")

    print("\n  ── Classification (High-Risk Alert) ─────────────────")
    for k, v in clf_metrics.items():
        print(f"    {k:<12}: {v:.4f}")
    if clf_cv:
        print(f"    CV AUC    : {clf_cv['ROC_AUC']['mean']:.4f} ± {clf_cv['ROC_AUC']['std']:.4f}")
        print(f"    CV F1     : {clf_cv['F1']['mean']:.4f} ± {clf_cv['F1']['std']:.4f}")

    print(f"\n  Output plots → {config.PLOT_DIR}/")
    print(f"  Saved models  → {config.MODEL_DIR}/")
    print(sep)
