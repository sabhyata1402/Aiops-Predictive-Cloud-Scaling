"""
SHAP Analysis — H9MLAI Project
Sabhyata Kumari | X24283142

Computes and saves SHAP values for XGBoost, Random Forest, and LSTM.
Generates all figures needed for the report.

Run:  python src/explainability/shap_analysis.py
"""

import numpy as np
import pandas as pd
import joblib
import shap
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless rendering
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR  = Path("data/models")
RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("data/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure `from src...` imports work when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = 'target_cpu'


def load_models_and_data():
    train = pd.read_parquet("data/processed/alibaba_train.parquet")
    test  = pd.read_parquet("data/processed/alibaba_test.parquet")
    fcols = joblib.load("data/processed/alibaba_feature_cols.pkl")

    X_train = train[fcols].values.astype(np.float32)
    X_test  = test[fcols].values.astype(np.float32)
    y_test  = test[TARGET].values.astype(np.float32)

    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    rf_model  = joblib.load(MODELS_DIR / "rf_model.pkl")

    # Load LSTM
    from src.models.train_all import LSTMForecaster
    ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location='cpu')
    n_features = ckpt['n_features']
    lstm_model = LSTMForecaster(n_features=n_features)
    lstm_model.load_state_dict(ckpt['model_state'])
    lstm_model.eval()

    return (X_train, X_test, y_test, fcols,
            xgb_model, rf_model, lstm_model)


def shap_xgboost(model, X_train, X_test, feature_names):
    print("Computing SHAP for XGBoost...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:2000])   # sample for speed
    joblib.dump({'shap_values': shap_values,
                 'feature_names': feature_names},
                RESULTS_DIR / "shap_xgboost.pkl")

    # Global importance plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:2000],
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title("XGBoost — SHAP Feature Importance (Global)",
              fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_xgb_global.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:2000],
                      feature_names=feature_names, show=False)
    plt.title("XGBoost — SHAP Beeswarm (Impact on CPU Forecast)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_xgb_beeswarm.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # Waterfall for one example
    idx = np.argmax(np.abs(shap_values).sum(axis=1))  # most extreme
    shap_exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test[idx],
        feature_names=feature_names
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, max_display=12, show=False)
    plt.title("XGBoost — SHAP Waterfall (Single High-CPU Prediction)",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_xgb_waterfall.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"  ✅ XGBoost SHAP saved. Top features:")
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:5]
    for i in top_idx:
        print(f"     {feature_names[i]}: {mean_abs[i]:.4f}")
    return shap_values


def shap_random_forest(model, X_train, X_test, feature_names):
    print("Computing SHAP for Random Forest...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:1000])
    joblib.dump({'shap_values': shap_values, 'feature_names': feature_names},
                RESULTS_DIR / "shap_rf.pkl")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:1000],
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title("Random Forest — SHAP Feature Importance (Global)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_rf_global.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Random Forest SHAP saved")
    return shap_values


def shap_lstm_gradient(model, X_test, feature_names, seq_len=6):
    """
    Gradient-based feature importance for LSTM.
    For each anomalous window, compute gradient of output w.r.t. input features.
    Mean absolute gradient = proxy for feature importance.
    """
    print("Computing gradient-based importance for LSTM...")

    n_samples = min(500, len(X_test) - seq_len)
    importances = np.zeros((n_samples, len(feature_names)))

    for i in range(n_samples):
        x = torch.tensor(X_test[i:i+seq_len], dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)
        out = model(x)
        out.backward()
        grad = x.grad.squeeze(0).abs().mean(dim=0)  # mean across timesteps
        importances[i] = grad.detach().numpy()

    mean_imp = importances.mean(axis=0)
    top_idx  = np.argsort(mean_imp)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2196F3' if v > mean_imp.mean() else '#90CAF9'
              for v in mean_imp[top_idx]]
    ax.barh([feature_names[i] for i in top_idx[::-1]],
            mean_imp[top_idx[::-1]], color=colors[::-1])
    ax.set_xlabel("Mean |Gradient| (feature importance proxy)")
    ax.set_title("LSTM — Gradient-Based Feature Importance (Top 20)", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_lstm_gradient.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    joblib.dump({'importances': importances, 'feature_names': feature_names},
                RESULTS_DIR / "shap_lstm.pkl")
    print(f"  ✅ LSTM gradient importance saved. Top features:")
    for i in top_idx[:5]:
        print(f"     {feature_names[i]}: {mean_imp[i]:.4f}")
    return importances


def plot_feature_comparison(xgb_shap, rf_shap, lstm_imp, feature_names):
    """Side-by-side top-10 comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    for ax, (name, vals) in zip(axes, [
        ("XGBoost", np.abs(xgb_shap).mean(axis=0)),
        ("Random Forest", np.abs(rf_shap).mean(axis=0)),
        ("LSTM", lstm_imp.mean(axis=0)),
    ]):
        top_idx = np.argsort(vals)[-10:]
        ax.barh([feature_names[i].replace('_', ' ') for i in top_idx],
                vals[top_idx], color='#1565C0')
        ax.set_title(f"{name}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Mean |SHAP| / |Gradient|")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Feature Importance Comparison Across Models — Alibaba Dataset",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature comparison plot saved")


def plot_learning_curves():
    """Plot learning curves from saved results."""
    curves = joblib.load(RESULTS_DIR / "learning_curves.pkl")

    fig, ax = plt.subplots(figsize=(9, 6))
    fractions = [f * 100 for f in curves['fraction']]
    colors = {'xgb': '#E53935', 'rf': '#1E88E5', 'lstm': '#43A047'}
    labels = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'lstm': 'LSTM'}

    for key in ['xgb', 'rf', 'lstm']:
        vals = curves[key]
        valid = [(f, v) for f, v in zip(fractions, vals) if v is not None]
        if valid:
            fx, vy = zip(*valid)
            ax.plot(fx, vy, 'o-', color=colors[key], label=labels[key],
                    linewidth=2, markersize=7)

    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("Test RMSE (CPU %)", fontsize=12)
    ax.set_title("Learning Curves — All Models on Alibaba Dataset", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Learning curves plot saved")


def plot_cross_cloud():
    """Heatmap of cross-cloud generalisation."""
    import seaborn as sns
    results = joblib.load(RESULTS_DIR / "cross_cloud_results.pkl")
    if not results:
        print("⚠  No cross-cloud results to plot")
        return

    models  = ['xgboost_rmse', 'rf_rmse', 'lstm_rmse']
    model_labels = ['XGBoost', 'Random Forest', 'LSTM']
    providers = ['alibaba'] + list(results.keys())

    # Load Alibaba in-distribution results
    all_res = joblib.load(RESULTS_DIR / "all_results.pkl")
    ali_row = {
        'xgboost_rmse': all_res['model_performance']['xgboost']['rmse'],
        'rf_rmse':       all_res['model_performance']['random_forest']['rmse'],
        'lstm_rmse':     all_res['model_performance']['lstm']['rmse'],
    }

    data = []
    for m in models:
        row = [ali_row.get(m, 0)]
        for p in list(results.keys()):
            row.append(results[p].get(m, 0))
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pd.DataFrame(data, index=model_labels,
                     columns=['Alibaba (train)'] +
                              [p.capitalize() for p in results.keys()]),
        annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, linewidths=0.5
    )
    ax.set_title("Cross-Cloud Generalisation — Test RMSE (lower = better)",
                 fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_cloud_heatmap.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print("✅ Cross-cloud heatmap saved")


def plot_predictions_vs_actual():
    """Time-series plot: actual vs predicted CPU for all 3 models."""
    test  = pd.read_parquet("data/processed/alibaba_test.parquet")
    fcols = joblib.load("data/processed/alibaba_feature_cols.pkl")
    X     = test[fcols].values.astype(np.float32)
    y     = test[TARGET].values.astype(np.float32)

    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    rf_model  = joblib.load(MODELS_DIR / "rf_model.pkl")

    n_show = 500
    xgb_p = xgb_model.predict(X[:n_show])
    rf_p  = rf_model.predict(X[:n_show])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y[:n_show], color='black', lw=1.2, label='Actual CPU%', alpha=0.8)
    ax.plot(xgb_p, color='#E53935', lw=1, label='XGBoost', alpha=0.75)
    ax.plot(rf_p,  color='#1E88E5', lw=1, label='Random Forest', alpha=0.75)
    ax.axhline(85, color='grey', linestyle='--', lw=0.8, label='SLA threshold (85%)')
    ax.set_xlabel("Time steps (10-min intervals)", fontsize=12)
    ax.set_ylabel("CPU Utilisation (%)", fontsize=12)
    ax.set_title("Prediction vs Actual CPU% — Alibaba Test Set (500 windows)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predictions_vs_actual.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print("✅ Predictions vs actual plot saved")


def plot_cost_analysis():
    """Bar chart of daily EUR savings per model."""
    all_res = joblib.load(RESULTS_DIR / "all_results.pkl")
    cost    = all_res.get('cost_analysis', {})
    if not cost:
        print("⚠  No cost results to plot")
        return

    models  = list(cost.keys())
    savings = [cost[m]['mean_daily_saving_eur'] for m in models]
    ci_lo   = [cost[m]['ci_lower'] for m in models]
    ci_hi   = [cost[m]['ci_upper'] for m in models]
    errs    = [[s - l for s, l in zip(savings, ci_lo)],
               [h - s for h, s in zip(ci_hi, savings)]]
    labels  = ['XGBoost', 'Random Forest', 'LSTM']
    colors  = ['#43A047', '#1E88E5', '#E53935']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, savings, color=colors, width=0.5,
                  yerr=errs, capsize=6, error_kw={'linewidth': 1.5})
    ax.set_ylabel("Estimated Daily Saving (€)", fontsize=12)
    ax.set_title("Cost Saving: Proactive ML Scaling vs Reactive Auto-Scaling\n"
                 "(AWS m5.xlarge, EU-West-1 Dublin, €0.192/hr)",
                 fontsize=11)
    ax.axhline(0, color='black', lw=0.8)
    for bar, val in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"€{val:.2f}", ha='center', va='bottom', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cost_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Cost analysis plot saved")


def main():
    print("=" * 60)
    print("H9MLAI — SHAP Analysis and Figure Generation")
    print("=" * 60)

    (X_train, X_test, y_test, feature_names,
     xgb_model, rf_model, lstm_model) = load_models_and_data()

    xgb_shap  = shap_xgboost(xgb_model,  X_train, X_test, feature_names)
    rf_shap   = shap_random_forest(rf_model, X_train, X_test, feature_names)
    lstm_imp  = shap_lstm_gradient(lstm_model, X_test, feature_names)

    plot_feature_comparison(xgb_shap, rf_shap, lstm_imp, feature_names)
    plot_learning_curves()
    plot_cross_cloud()
    plot_predictions_vs_actual()
    plot_cost_analysis()

    print(f"\n✅ All figures saved to {FIGURES_DIR}/")
    print("   Next: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
