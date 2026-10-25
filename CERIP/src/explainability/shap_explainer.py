"""
CERIP — SHAP Explainability (M5)
==================================
Global and per-instance SHAP explanations for both modules.

Functions
---------
explain_heat_model(model, X_df, feature_names, ...)
    Compute SHAP values for the heat risk model.
    Returns global summary + per-cell explanations.

explain_aqi_model(model, X_df, feature_names, ...)
    Compute SHAP values for the AQI Gradient Boosting model.

plot_global_summary(shap_values, feature_names, ...)
    Save beeswarm / bar plot to reports/figures/.

plot_cell_waterfall(shap_values, idx, feature_names, ...)
    Single-cell waterfall explanation.

get_top_drivers(shap_values, feature_names, idx, n=5)
    Return top N contributing features for one prediction.
"""

import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import shap
import yaml

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

MAX_DISPLAY = CFG["shap"]["max_display_features"]
BG_SAMPLE   = CFG["shap"]["background_sample_size"]


# =============================================================
# Core SHAP helpers
# =============================================================

def _get_tree_explainer(model, X_background: np.ndarray) -> shap.Explainer:
    """Create a TreeExplainer (fast, exact for tree-based models)."""
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def _get_kernel_explainer(model, X_background: np.ndarray) -> shap.KernelExplainer:
    """Fallback KernelExplainer for non-tree models."""
    # Summarise background to speed up
    bg = shap.sample(X_background, BG_SAMPLE)
    return shap.KernelExplainer(model.predict, bg)


def compute_shap_values(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    use_tree_explainer: bool = True,
    max_evals: int = 500,
) -> tuple:
    """
    Compute SHAP values for a fitted model.

    Returns
    -------
    explainer  : fitted shap.Explainer
    shap_vals  : np.ndarray of shape (n_samples, n_features)
    base_value : float (expected model output)
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    if use_tree_explainer:
        try:
            explainer  = _get_tree_explainer(model, X_arr)
            shap_vals  = explainer.shap_values(X_arr)
            # For multi-class classifiers, shap_vals is a list; use class-average
            if isinstance(shap_vals, list):
                shap_vals = np.abs(np.stack(shap_vals)).mean(axis=0)
            base_value = (
                explainer.expected_value[0]
                if hasattr(explainer.expected_value, "__len__")
                else float(explainer.expected_value)
            )
            return explainer, shap_vals, base_value
        except Exception as e:
            log.warning("TreeExplainer failed (%s). Falling back to KernelExplainer.", e)

    # Kernel fallback
    explainer  = _get_kernel_explainer(model, X_arr)
    shap_vals  = explainer.shap_values(X_arr[:BG_SAMPLE])
    base_value = float(explainer.expected_value)
    return explainer, shap_vals, base_value


# =============================================================
# Heat Risk SHAP
# =============================================================

def explain_heat_model(
    model,
    X_df: pd.DataFrame,
    feature_names: list[str],
    label: str = "heat_regressor",
    save_plots: bool = True,
) -> dict:
    """
    Compute and optionally plot SHAP values for the heat risk model.

    Returns
    -------
    dict with:
        shap_values  : np.ndarray (n_samples, n_features)
        base_value   : float
        global_importance : pd.DataFrame sorted by mean |SHAP|
        top_features : list[str]
    """
    log.info("Computing SHAP for %s …", label)

    X_arr = X_df[feature_names].values if isinstance(X_df, pd.DataFrame) else X_df
    explainer, shap_vals, base_value = compute_shap_values(
        model, X_arr, use_tree_explainer=True
    )

    # ── Global importance ─────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    global_imp = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_features = global_imp["feature"].head(MAX_DISPLAY).tolist()

    result = {
        "shap_values":       shap_vals,
        "base_value":        base_value,
        "global_importance": global_imp,
        "top_features":      top_features,
        "feature_names":     feature_names,
        "X_arr":             X_arr,
    }

    if save_plots:
        _plot_beeswarm(shap_vals, X_arr, feature_names, label)
        _plot_bar(global_imp, label)
        log.info("SHAP plots saved to %s", FIGURES_DIR)

    return result


def explain_aqi_model(
    model,
    X_df: pd.DataFrame,
    feature_names: list[str],
    label: str = "aqi_gb",
    save_plots: bool = True,
) -> dict:
    """Compute SHAP for the AQI Gradient Boosting model (same interface)."""
    return explain_heat_model(model, X_df, feature_names, label, save_plots)


# =============================================================
# Per-instance explanations
# =============================================================

def get_top_drivers(
    shap_values: np.ndarray,
    feature_names: list[str],
    idx: int,
    n: int = 5,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top N driving features for prediction `idx`.
    Positive SHAP = increases risk, Negative SHAP = decreases risk.
    """
    vals = shap_values[idx]
    df = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": vals,
        "direction":  ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in vals],
    })
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(n).reset_index(drop=True)


def explain_single_prediction(
    shap_result: dict,
    idx: int,
    n_top: int = 5,
    feature_value_df: pd.DataFrame | None = None,
) -> dict:
    """
    Return a structured explanation for one prediction.

    Returns
    -------
    dict with:
        top_drivers : pd.DataFrame
        base_value  : float
        prediction_delta : float (sum of top-N SHAP values)
        narrative   : str (human-readable explanation)
    """
    shap_values  = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    base_value   = shap_result["base_value"]

    top_df = get_top_drivers(shap_values, feature_names, idx, n_top)

    # Build narrative
    lines = [f"Base expected output: {base_value:.2f}"]
    for _, row in top_df.iterrows():
        feat_val = ""
        if feature_value_df is not None and row["feature"] in feature_value_df.columns:
            feat_val = f" = {feature_value_df.iloc[idx][row['feature']]:.3f}"
        lines.append(
            f"  • {row['feature']}{feat_val}: "
            f"{row['direction']} by {abs(row['shap_value']):.3f}"
        )
    narrative = "\n".join(lines)

    return {
        "top_drivers":       top_df,
        "base_value":        base_value,
        "prediction_delta":  float(top_df["shap_value"].sum()),
        "narrative":         narrative,
    }


# =============================================================
# Plotting helpers
# =============================================================

def _plot_beeswarm(
    shap_vals: np.ndarray,
    X_arr: np.ndarray,
    feature_names: list[str],
    label: str,
):
    """Save a SHAP beeswarm (summary) plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_arr,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"shap_beeswarm_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved beeswarm: %s", out)


def _plot_bar(global_imp: pd.DataFrame, label: str):
    """Save a SHAP global feature importance bar chart."""
    top = global_imp.head(MAX_DISPLAY)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top["mean_abs_shap"]]
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance — {label}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / f"shap_bar_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved bar plot: %s", out)


def plot_cell_waterfall(
    shap_result: dict,
    idx: int,
    title: str = "Per-Cell SHAP Explanation",
    save_path: Path | None = None,
):
    """
    Plot waterfall chart for a single prediction and optionally save.
    Returns the figure path or None.
    """
    shap_vals    = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    base_value   = shap_result["base_value"]
    X_arr        = shap_result.get("X_arr")

    exp = shap.Explanation(
        values=shap_vals[idx],
        base_values=base_value,
        data=X_arr[idx] if X_arr is not None else None,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp, max_display=10, show=False)
    ax.set_title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"shap_waterfall_cell_{idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved waterfall: %s", save_path)
    return str(save_path)
