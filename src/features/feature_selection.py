"""
Feature Selection — H9MLAI Project
Sabhyata Kumari | X24283142

Applies two-stage feature selection:
  1. Correlation filter  — remove features with pairwise correlation > 0.95
  2. RFE                 — Recursive Feature Elimination with XGBoost estimator
                          (selects top-N features by importance rank)

Saves:
  data/processed/{dataset}_selected_features.pkl   — list of selected column names
  data/processed/{dataset}_rfe_importances.pkl     — RFE ranking array

Run:  python src/features/feature_selection.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CORR_THRESHOLD  = 0.95   # drop one of any pair more correlated than this
N_FEATURES_RFE  = 25     # keep top-25 features after RFE
RFE_STEP        = 3      # eliminate 3 features per iteration (faster)
DATASETS        = ['alibaba', 'azure', 'google']
DATA_DIR        = Path("data/processed")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Correlation Filter
# ─────────────────────────────────────────────────────────────────────────────

def correlation_filter(df: pd.DataFrame, feature_cols: list, threshold: float = CORR_THRESHOLD):
    """
    Remove one feature from each pair whose absolute Pearson correlation
    exceeds `threshold`.  Returns the surviving feature column names.
    """
    X = df[feature_cols]
    corr_matrix = X.corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Columns to drop: if any correlation > threshold, drop the second column
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    surviving = [c for c in feature_cols if c not in to_drop]

    print(f"  Correlation filter: {len(feature_cols)} → {len(surviving)} features "
          f"(dropped {len(to_drop)} with |r| > {threshold})")
    return surviving


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Recursive Feature Elimination
# ─────────────────────────────────────────────────────────────────────────────

def rfe_selection(df: pd.DataFrame, feature_cols: list, n_features: int = N_FEATURES_RFE):
    """
    Run RFE with a fast XGBoost estimator on the training data.
    Returns (selected_feature_names, rfe_ranking_array).
    """
    n_select = min(n_features, len(feature_cols))

    X = df[feature_cols].values
    y = df['target_cpu'].values

    estimator = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        tree_method='hist',
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=RFE_STEP)
    rfe.fit(X, y)

    selected = [col for col, support in zip(feature_cols, rfe.support_) if support]
    print(f"  RFE: {len(feature_cols)} → {len(selected)} features selected")
    return selected, rfe.ranking_


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_selection(dataset: str):
    """Run full two-stage selection for one dataset and save results."""
    train_path = DATA_DIR / f"{dataset}_train.parquet"
    if not train_path.exists():
        print(f"  ⚠️  {train_path} not found — run preprocessor.py first")
        return None

    print(f"\n── Feature Selection: {dataset} ──────────────────────────")
    train_df = pd.read_parquet(train_path)

    # Load feature column list produced by preprocessor
    feat_cols_path = DATA_DIR / f"{dataset}_feature_cols.pkl"
    if feat_cols_path.exists():
        feature_cols = joblib.load(feat_cols_path)
    else:
        # Derive from dataframe
        exclude = {'node_id', 'ts', 'provider',
                   'target_cpu', 'target_cpu_h1', 'target_cpu_h6',
                   'target_mem', 'sla_breach'}
        feature_cols = [c for c in train_df.columns if c not in exclude]

    print(f"  Starting features: {len(feature_cols)}")

    # Stage 1 — correlation filter
    after_corr = correlation_filter(train_df, feature_cols)

    # Stage 2 — RFE
    selected, ranking = rfe_selection(train_df, after_corr)

    # Save
    joblib.dump(selected, DATA_DIR / f"{dataset}_selected_features.pkl")
    joblib.dump(ranking,  DATA_DIR / f"{dataset}_rfe_importances.pkl")

    # Summary table
    print(f"  ✅ Final selected ({len(selected)}): {selected[:8]}{'...' if len(selected) > 8 else ''}")
    return selected


def main():
    print("=" * 60)
    print("H9MLAI — Feature Selection Pipeline")
    print("=" * 60)

    summary = {}
    for ds in DATASETS:
        selected = run_feature_selection(ds)
        if selected:
            summary[ds] = len(selected)

    print("\n── Summary ───────────────────────────────────────────────")
    for ds, n in summary.items():
        print(f"  {ds:10s}: {n} features selected (from 40+)")
    print("\nNext: python src/models/train_all.py  (uses selected features)")


if __name__ == "__main__":
    main()
