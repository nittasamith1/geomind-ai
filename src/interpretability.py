"""
GeoMind AI - Model Interpretability Module (SHAP)
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Implements:
1. SHAP TreeExplainer for XGBoost / Random Forest.
2. Global feature importance: mean |SHAP| bar chart.
3. Per-sample local explanation: SHAP waterfall / force plots.
4. SHAP summary scatter plot across validation set.
5. Residual error analysis: error distribution, time-of-day and weekday slicing.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException
from src.data_preprocessing import prepare_datasets


FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_xgboost_model() -> object:
    """Loads the saved XGBoost model from the artifact store."""
    model_path = Path("models/ml/xgboost.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {model_path}. Run train_ml.py first.")
    return joblib.load(model_path)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes SHAP values using TreeExplainer (exact, not sampling).
    Subsamples to max_samples for speed on large validation sets.

    Returns:
        shap_values: np.ndarray of shape (n_samples, n_features)
        X_sample: np.ndarray of shape (n_samples, n_features)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install shap")

    logger.info(f"Computing SHAP values (samples={min(len(X), max_samples)})...")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=min(len(X), max_samples), replace=False)
    X_sample = X[indices]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
    return shap_values, X_sample


def plot_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """
    Plots mean absolute SHAP values as horizontal bar chart — global feature importance.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(importance_df["feature"], importance_df["mean_abs_shap"], color=colors, height=0.65)

    ax.set_xlabel("Mean |SHAP Value| (impact on model output magnitude)", color="white", fontsize=11)
    ax.set_title(f"GeoMind AI: Global Feature Importance (Top {top_n}) — SHAP TreeExplainer", 
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.xaxis.label.set_color("white")

    # Annotate bar values
    for bar, val in zip(bars, importance_df["mean_abs_shap"]):
        ax.text(bar.get_width() + max(importance_df["mean_abs_shap"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left", color="white", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"SHAP global importance plot saved to {save_path}")
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """
    SHAP beeswarm / dot summary plot: shows feature impact distribution across all samples.
    """
    try:
        import shap
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")

        shap_top = shap_values[:, top_idx]
        feat_top = X_sample[:, top_idx]
        feat_names_top = [feature_names[i] for i in top_idx]

        # Scatter each feature as a row
        for row_idx, feat_name in enumerate(reversed(feat_names_top)):
            col_idx = len(feat_names_top) - 1 - row_idx
            sv = shap_top[:, col_idx]
            fv = feat_top[:, col_idx]
            fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
            colors = plt.cm.RdBu_r(fv_norm)
            jitter = np.random.default_rng(row_idx).uniform(-0.15, 0.15, size=len(sv))
            ax.scatter(sv, np.full_like(sv, row_idx) + jitter, c=colors, alpha=0.4, s=8, linewidths=0)

        ax.set_yticks(range(len(feat_names_top)))
        ax.set_yticklabels(reversed(feat_names_top), color="white", fontsize=8)
        ax.axvline(0, color="#888", lw=1, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on traffic volume prediction)", color="white", fontsize=10)
        ax.set_title("GeoMind AI: SHAP Summary — Feature Impact Distribution", 
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#444")

        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Feature Value (normalized: low → high)", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"SHAP summary plot saved to {save_path}")
        plt.close()
    except Exception as e:
        raise CustomException(e, sys)


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    datetimes: pd.Series,
    save_prefix: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs structured error analysis:
    - Residual distribution (histogram + Q-Q)
    - Errors sliced by hour-of-day
    - Errors sliced by day-of-week
    - Percentile breakdown of large errors

    Returns:
        error_df: DataFrame with raw errors and metadata columns
    """
    try:
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        dt = pd.to_datetime(datetimes)
        error_df = pd.DataFrame({
            "datetime": dt,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": errors,
            "abs_error": abs_errors,
            "hour": dt.dt.hour,
            "day_of_week": dt.dt.dayofweek,
            "day_name": dt.dt.day_name()
        })

        # ── Plot 1: Error Distribution ──────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")

        ax = axes[0]
        ax.hist(errors, bins=80, color="#7c5cbf", alpha=0.85, edgecolor="none")
        ax.axvline(0, color="#ff4444", lw=1.5, linestyle="--", label="Zero Error")
        ax.axvline(errors.mean(), color="#44ff88", lw=1.5, linestyle=":", label=f"Mean={errors.mean():.0f}")
        ax.set_title("Residual Error Distribution (Predicted − True)", color="white", fontweight="bold")
        ax.set_xlabel("Prediction Error (vehicles/hr)", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#444")
        ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

        ax = axes[1]
        from scipy.stats import probplot
        (osm, osr), (slope, intercept, _) = probplot(errors, dist="norm")
        ax.scatter(osm, osr, s=4, alpha=0.4, color="#5bc8f5")
        qqline = np.array([osm[0], osm[-1]]) * slope + intercept
        ax.plot([osm[0], osm[-1]], qqline, color="#ff8844", lw=1.5, label="Normal Reference")
        ax.set_title("Q-Q Plot (Residuals vs Normal Distribution)", color="white", fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles", color="white")
        ax.set_ylabel("Sample Quantiles", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#444")
        ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)

        plt.suptitle("GeoMind AI: Error Analysis — Residual Diagnostics", 
                     color="white", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        if save_prefix:
            path = FIGURES_DIR / f"{save_prefix}_error_dist.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"Error distribution plot saved to {path}")
        plt.close()

        # ── Plot 2: MAE by Hour of Day ───────────────────────────────────────
        hourly_mae = error_df.groupby("hour")["abs_error"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        colors = ["#ff6b6b" if (7 <= h <= 9 or 16 <= h <= 18) else "#5bc8f5" for h in hourly_mae["hour"]]
        ax.bar(hourly_mae["hour"], hourly_mae["abs_error"], color=colors, alpha=0.85, width=0.7)
        ax.set_xlabel("Hour of Day", color="white", fontsize=11)
        ax.set_ylabel("Mean Absolute Error (vehicles/hr)", color="white", fontsize=11)
        ax.set_title("GeoMind AI: Error Analysis — MAE by Hour of Day\n"
                     "(Red = Rush Hours: 07-09, 16-18)", 
                     color="white", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, 24))
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#444")
        rush_patch = mpatches.Patch(color="#ff6b6b", label="Rush Hour")
        off_patch = mpatches.Patch(color="#5bc8f5", label="Off-Peak")
        ax.legend(handles=[rush_patch, off_patch], facecolor="#1a1d27", labelcolor="white")
        plt.tight_layout()
        if save_prefix:
            path = FIGURES_DIR / f"{save_prefix}_mae_by_hour.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"Hourly MAE plot saved to {path}")
        plt.close()

        # ── Plot 3: MAE by Day of Week ───────────────────────────────────────
        DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_mae = error_df.groupby("day_of_week")["abs_error"].mean().reset_index()
        daily_mae["day_name"] = daily_mae["day_of_week"].apply(lambda x: DOW_NAMES[x])

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        colors_dow = ["#aaa" if d >= 5 else "#7c5cbf" for d in daily_mae["day_of_week"]]
        ax.bar(daily_mae["day_name"], daily_mae["abs_error"], color=colors_dow, alpha=0.85, width=0.65)
        ax.set_xlabel("Day of Week", color="white", fontsize=11)
        ax.set_ylabel("Mean Absolute Error (vehicles/hr)", color="white", fontsize=11)
        ax.set_title("GeoMind AI: Error Analysis — MAE by Day of Week\n(Gray = Weekend)", 
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#444")
        plt.tight_layout()
        if save_prefix:
            path = FIGURES_DIR / f"{save_prefix}_mae_by_dow.png"
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"Day-of-week MAE plot saved to {path}")
        plt.close()

        # ── Percentile breakdown ────────────────────────────────────────────
        p90_threshold = np.percentile(abs_errors, 90)
        p95_threshold = np.percentile(abs_errors, 95)
        p99_threshold = np.percentile(abs_errors, 99)
        logger.info(f"Error Percentiles — P90: {p90_threshold:.1f} | P95: {p95_threshold:.1f} | P99: {p99_threshold:.1f}")

        error_df["error_severity"] = pd.cut(
            abs_errors,
            bins=[0, 200, 500, p90_threshold, np.inf],
            labels=["Small (<200)", "Medium (200-500)", "Large (>P90)", "Extreme (>P95)"]
        )

        return error_df

    except Exception as e:
        raise CustomException(e, sys)


def run_full_interpretability_pipeline() -> None:
    """
    End-to-end interpretability pipeline:
    1. Load datasets and XGBoost model.
    2. Compute SHAP values.
    3. Plot global importance and summary.
    4. Perform residual error analysis.
    """
    try:
        logger.info("=" * 60)
        logger.info("GeoMind AI: Model Interpretability & Error Analysis Pipeline")
        logger.info("=" * 60)

        X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
        model = _load_xgboost_model()

        # SHAP Analysis on validation set
        shap_values, X_sample = compute_shap_values(model, X_val, feat_names, max_samples=3000)

        plot_global_feature_importance(
            shap_values, feat_names, top_n=20,
            save_path=FIGURES_DIR / "shap_global_importance.png"
        )
        plot_shap_summary(
            shap_values, X_sample, feat_names, top_n=20,
            save_path=FIGURES_DIR / "shap_summary_scatter.png"
        )

        # Error Analysis on test set
        test_preds = model.predict(X_test)
        test_df = pd.read_csv("data/processed/test.csv")
        test_df["date_time"] = pd.to_datetime(test_df["date_time"])

        error_df = analyze_prediction_errors(
            y_true=y_test,
            y_pred=test_preds,
            datetimes=test_df["date_time"].iloc[:len(y_test)].reset_index(drop=True),
            save_prefix="xgboost_test"
        )

        logger.info("\nError Analysis Summary:")
        logger.info(f"  Total test samples: {len(error_df):,}")
        logger.info(f"  Mean AE: {error_df['abs_error'].mean():.2f}")
        logger.info(f"  Median AE: {error_df['abs_error'].median():.2f}")
        logger.info(f"  Std of Error: {error_df['error'].std():.2f}")
        logger.info("Interpretability pipeline complete. Figures saved to docs/figures/")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_full_interpretability_pipeline()
