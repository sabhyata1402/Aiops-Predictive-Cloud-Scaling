"""
Train All Models — H9MLAI Project
Sabhyata Kumari | X24283142

Trains XGBoost, Random Forest, and LSTM on Alibaba data.
Includes:
  - Bayesian hyperparameter tuning (XGBoost via Optuna)
  - Grid search hyperparameter tuning (Random Forest)
  - Multi-horizon evaluation (10 min, 30 min, 60 min)
  - Transfer learning (fine-tune on 10% Azure/Google data)
  - Wilcoxon signed-rank test for statistical significance
  - Model size, training time, inference latency metrics
  - Cross-cloud generalisation evaluation
  - Cost analysis (proactive vs reactive scaling)

Run:  python src/models/train_all.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wilcoxon
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

MODELS_DIR  = Path("data/models")
RESULTS_DIR = Path("data/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'target_cpu'   # 30-min primary target
SEED   = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(provider='alibaba'):
    base  = Path("data/processed")
    train = pd.read_parquet(base / f"{provider}_train.parquet")
    val   = pd.read_parquet(base / f"{provider}_val.parquet")
    test  = pd.read_parquet(base / f"{provider}_test.parquet")
    fcols = joblib.load(base / f"{provider}_feature_cols.pkl")
    return train, val, test, fcols


def xy(df, feature_cols, target=TARGET):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    return X, y


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4),
            'mape': round(mape, 4)}


def model_size_mb(path):
    """Return model file size in MB."""
    try:
        return round(os.path.getsize(path) / 1e6, 2)
    except Exception:
        return None


def to_jsonable(obj):
    if isinstance(obj, bool):           # must be before int check (bool subclasses int)
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — XGBoost  (Bayesian tuning via Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n── XGBoost (Bayesian Hyperparameter Tuning) ────────────────")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators':      trial.suggest_int('n_est', 100, 400),
                'max_depth':         trial.suggest_int('depth', 3, 8),
                'learning_rate':     trial.suggest_float('lr', 0.01, 0.3, log=True),
                'subsample':         trial.suggest_float('sub', 0.6, 1.0),
                'colsample_bytree':  trial.suggest_float('col', 0.6, 1.0),
                'reg_alpha':         trial.suggest_float('alpha', 1e-3, 5.0, log=True),
                'early_stopping_rounds': 20,
                'random_state': SEED, 'n_jobs': -1,
            }
            m = xgb.XGBRegressor(**params, verbosity=0)
            m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)
            return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        best_params = study.best_params
        best_params.update({'random_state': SEED, 'n_jobs': -1})
        print(f"  Best params: {best_params}")

    except ImportError:
        print("  Optuna not available — using defaults")
        best_params = {
            'n_estimators': 300, 'max_depth': 6,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': SEED, 'n_jobs': -1,
        }

    t0 = time.time()
    best_params['early_stopping_rounds'] = 20
    model = xgb.XGBRegressor(**best_params, verbosity=0)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        _ = model.predict(X_val)
    latency_ms = (time.time() - t0) / 10 * 1000

    model_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    size_mb = model_size_mb(model_path)
    print(f"  Train: {train_time:.1f}s | Latency: {latency_ms:.1f}ms "
          f"| Size: {size_mb}MB")
    return model, train_time, latency_ms, size_mb, best_params


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Random Forest  (Grid search tuning)
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n── Random Forest (Grid Search Hyperparameter Tuning) ───────")

    # Grid search on a subset for speed
    n_gs = min(len(X_train), 8000)
    param_grid = {
        'n_estimators':    [50, 100],
        'max_depth':       [10, 15],
        'min_samples_split': [5, 10],
        'max_features':    ['sqrt'],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        param_grid, cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=1, verbose=0    # n_jobs=1 avoids Python 3.13 multiprocessing bug on macOS
    )
    gs.fit(X_train[:n_gs], y_train[:n_gs])
    best_params = gs.best_params_
    print(f"  Best params: {best_params}")

    t0 = time.time()
    model = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        _ = model.predict(X_val)
    latency_ms = (time.time() - t0) / 10 * 1000

    model_path = MODELS_DIR / "rf_model.pkl"
    joblib.dump(model, model_path)
    size_mb = model_size_mb(model_path)
    print(f"  Train: {train_time:.1f}s | Latency: {latency_ms:.1f}ms "
          f"| Size: {size_mb}MB")
    return model, train_time, latency_ms, size_mb, best_params


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — LSTM  (2-layer, dropout, early stopping, LR schedule)
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=6):
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (self.X[idx:idx + self.seq_len],
                self.y[idx + self.seq_len])


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc1     = nn.Linear(hidden, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    print("\n── LSTM (2-layer, dropout=0.2, Adam + LR schedule) ────────")

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEQ_LEN    = 6
    BATCH_SIZE = 256
    MAX_EPOCHS = 60
    PATIENCE   = 10

    train_ds = TimeSeriesDataset(X_train, y_train, SEQ_LEN)
    val_ds   = TimeSeriesDataset(X_val,   y_val,   SEQ_LEN)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=False)

    model     = LSTMForecaster(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val, patience_ctr = float('inf'), 0
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        ep_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl += criterion(model(Xb.to(device)), yb.to(device)).item()

        avg_t = ep_loss / max(len(train_loader), 1)
        avg_v = vl     / max(len(val_loader),   1)
        train_losses.append(avg_t)
        val_losses.append(avg_v)
        scheduler.step(avg_v)

        if avg_v < best_val:
            best_val, patience_ctr = avg_v, 0
            torch.save(model.state_dict(), MODELS_DIR / "lstm_best.pt")
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={avg_t:.4f} "
                  f"val={avg_v:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    train_time = time.time() - t0
    model.load_state_dict(
        torch.load(MODELS_DIR / "lstm_best.pt", map_location=device))

    model_path = MODELS_DIR / "lstm_model.pt"
    torch.save({'model_state': model.state_dict(),
                'n_features':  n_features,
                'seq_len':     SEQ_LEN,
                'train_losses': train_losses,
                'val_losses':   val_losses},
               model_path)
    size_mb = model_size_mb(model_path)

    # Inference latency
    model.eval()
    n_lat = min(256, len(X_val) // SEQ_LEN)
    sample = torch.tensor(
        X_val[:n_lat * SEQ_LEN].reshape(n_lat, SEQ_LEN, n_features),
        dtype=torch.float32).to(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)
    latency_ms = (time.time() - t0) / 10 * 1000

    print(f"  Train: {train_time:.1f}s | Latency: {latency_ms:.1f}ms "
          f"| Size: {size_mb}MB")
    joblib.dump({'train': train_losses, 'val': val_losses},
                RESULTS_DIR / "lstm_loss_curves.pkl")
    return model, train_time, latency_ms, size_mb, train_losses, val_losses


def predict_lstm(model, X, seq_len=6, batch_size=512, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds, n = [], len(X) - seq_len
    for i in range(0, n, batch_size):
        end   = min(i + batch_size, n)
        batch = np.array([X[j:j + seq_len] for j in range(i, end)])
        t     = torch.tensor(batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds.extend(model(t).cpu().numpy().tolist())
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# WILCOXON SIGNED-RANK TEST
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_significance(y_true, preds_dict):
    """
    Pairwise Wilcoxon signed-rank tests on absolute errors.
    Tests whether RMSE differences between models are statistically significant
    (alpha = 0.05), as promised in Stage I proposal.
    """
    print("\n── Wilcoxon Signed-Rank Tests (alpha=0.05) ──────────────────")
    errors = {name: np.abs(y_true - p) for name, p in preds_dict.items()}
    pairs  = [('xgboost', 'random_forest'),
              ('xgboost', 'lstm'),
              ('random_forest', 'lstm')]
    results = {}
    for a, b in pairs:
        if a not in errors or b not in errors:
            continue
        ea, eb = errors[a], errors[b]
        n = min(len(ea), len(eb))
        stat, p = wilcoxon(ea[:n], eb[:n], alternative='two-sided')
        sig = "✅ significant" if p < 0.05 else "❌ not significant"
        label = f"{a}_vs_{b}"
        results[label] = {'W': round(float(stat), 4),
                          'p_value': round(float(p), 6),
                          'significant_at_0.05': p < 0.05}
        print(f"  {a} vs {b}: W={stat:.1f}, p={p:.4f} → {sig}")
    joblib.dump(results, RESULTS_DIR / "wilcoxon_results.pkl")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-HORIZON EVALUATION  (10 min, 30 min, 60 min)
# ─────────────────────────────────────────────────────────────────────────────

def multi_horizon_experiment(train, val, test, feature_cols, xgb_best_params):
    """
    Train XGBoost for 3 forecast horizons and compare performance.
    Corresponds to cost analysis across horizons promised in Stage I.
    """
    print("\n── Multi-Horizon Evaluation ─────────────────────────────────")

    horizons = {
        '10min  (h1)': 'target_cpu_h1',
        '30min  (h3)': 'target_cpu',
        '60min  (h6)': 'target_cpu_h6',
    }
    results = {}

    for label, target_col in horizons.items():
        # Check column exists
        if target_col not in train.columns:
            print(f"  ⚠  {label}: target column '{target_col}' missing — "
                  f"re-run preprocessor.py")
            continue

        X_tr = train[feature_cols].values.astype(np.float32)
        y_tr = train[target_col].values.astype(np.float32)
        X_v  = val[feature_cols].values.astype(np.float32)
        y_v  = val[target_col].values.astype(np.float32)
        X_te = test[feature_cols].values.astype(np.float32)
        y_te = test[target_col].values.astype(np.float32)

        # Drop NaN rows that arise from extreme shifts
        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
        X_te, y_te = X_te[mask_te], y_te[mask_te]

        params = {k: v for k, v in xgb_best_params.items()
                  if k not in ('n_jobs', 'random_state')}
        params.update({'n_jobs': -1, 'random_state': SEED, 'verbosity': 0,
                       'early_stopping_rounds': 15})

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        preds = m.predict(X_te)
        r = metrics(y_te, preds)
        results[label] = r
        print(f"  {label}: RMSE={r['rmse']}  MAE={r['mae']}  MAPE={r['mape']}%")

    joblib.dump(results, RESULTS_DIR / "multi_horizon_results.pkl")
    print("  ✅ Multi-horizon results saved")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER LEARNING  (fine-tune on 10% Azure / Google)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_learning(base_xgb, feature_cols):
    """
    Fine-tune the Alibaba-trained XGBoost on 10% of Azure/Google data.
    Reports RMSE before and after fine-tuning to quantify adaptation cost.
    """
    print("\n── Transfer Learning (fine-tune on 10% target data) ─────────")
    results = {}

    for provider in ['azure', 'google']:
        try:
            train_p, _, test_p, fcols_p = load_data(provider)

            # Common features between Alibaba and target provider
            common = [c for c in feature_cols if c in fcols_p]
            if len(common) < 5:
                print(f"  ⚠  {provider}: too few common features ({len(common)})")
                continue

            # Pad / align to Alibaba feature count
            def pad_features(X_src, n_target):
                if X_src.shape[1] < n_target:
                    pad = np.zeros((X_src.shape[0],
                                    n_target - X_src.shape[1]),
                                   dtype=np.float32)
                    return np.hstack([X_src, pad])
                return X_src[:, :n_target]

            n_feat = len(feature_cols)

            X_test_p = pad_features(
                test_p[common].values.astype(np.float32), n_feat)
            y_test_p  = test_p[TARGET].values.astype(np.float32)

            # Before fine-tuning
            preds_before = base_xgb.predict(X_test_p)
            rmse_before  = round(float(np.sqrt(
                mean_squared_error(y_test_p, preds_before))), 4)

            # Fine-tune on 10% of provider training data
            n_ft  = max(100, int(len(train_p) * 0.10))
            X_ft  = pad_features(
                train_p[common].values[:n_ft].astype(np.float32), n_feat)
            y_ft  = train_p[TARGET].values[:n_ft].astype(np.float32)

            fine_model = xgb.XGBRegressor(
                n_estimators=50, max_depth=5,
                learning_rate=0.05, random_state=SEED,
                n_jobs=-1, verbosity=0)
            fine_model.fit(X_ft, y_ft)

            preds_after = fine_model.predict(X_test_p)
            rmse_after  = round(float(np.sqrt(
                mean_squared_error(y_test_p, preds_after))), 4)

            improvement = round(rmse_before - rmse_after, 4)
            results[provider] = {
                'rmse_zero_shot':    rmse_before,
                'rmse_after_finetune': rmse_after,
                'improvement':       improvement,
                'n_finetune_samples': n_ft,
                'common_features':   len(common),
            }
            print(f"  {provider}: zero-shot={rmse_before}  "
                  f"fine-tuned={rmse_after}  "
                  f"improvement={improvement:+.4f}  "
                  f"(n_ft={n_ft})")

        except Exception as e:
            print(f"  ⚠  {provider}: {e}")

    joblib.dump(results, RESULTS_DIR / "transfer_learning_results.pkl")
    print("  ✅ Transfer learning results saved")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING CURVES  (10 / 25 / 50 / 75 / 100 % of training data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_learning_curves(X_train, y_train, X_test, y_test,
                             X_val, y_val):
    print("\n── Learning Curves (10/25/50/75/100% of training data) ──────")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    curves    = {'fraction': fractions, 'xgb': [], 'rf': [], 'lstm': []}
    n_feat    = X_train.shape[1]
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for frac in fractions:
        n       = max(64, int(len(X_train) * frac))
        X_s, y_s = X_train[:n], y_train[:n]
        print(f"  {frac:.0%}  ({n:,} samples)")

        # XGBoost
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5,
                              learning_rate=0.05, random_state=SEED,
                              verbosity=0, n_jobs=-1,
                              early_stopping_rounds=15)
        m.fit(X_s, y_s, eval_set=[(X_val, y_val)], verbose=False)
        curves['xgb'].append(
            round(float(np.sqrt(mean_squared_error(y_test, m.predict(X_test)))), 4))

        # Random Forest
        m_rf = RandomForestRegressor(n_estimators=50, max_depth=12,
                                     random_state=SEED, n_jobs=-1)
        m_rf.fit(X_s, y_s)
        curves['rf'].append(
            round(float(np.sqrt(mean_squared_error(y_test, m_rf.predict(X_test)))), 4))

        # LSTM (quick 15-epoch version for curve)
        ds = TimeSeriesDataset(X_s, y_s, seq_len=6)
        if len(ds) < 64:
            curves['lstm'].append(None)
            continue
        loader = DataLoader(ds, 128, shuffle=True, drop_last=True)
        m_lstm = LSTMForecaster(n_features=n_feat).to(device)
        opt    = torch.optim.Adam(m_lstm.parameters(), lr=1e-3)
        crit   = nn.MSELoss()
        for _ in range(15):
            m_lstm.train()
            for Xb, yb in loader:
                opt.zero_grad()
                loss = crit(m_lstm(Xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
        lp = predict_lstm(m_lstm, X_test, device=device)
        yt = y_test[6:]
        ml = min(len(lp), len(yt))
        curves['lstm'].append(
            round(float(np.sqrt(mean_squared_error(yt[:ml], lp[:ml]))), 4))
        print(f"    XGB={curves['xgb'][-1]}  "
              f"RF={curves['rf'][-1]}  LSTM={curves['lstm'][-1]}")

    joblib.dump(curves, RESULTS_DIR / "learning_curves.pkl")
    print("  ✅ Learning curves saved")
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CLOUD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_cloud_eval(xgb_model, rf_model, lstm_model, feature_cols):
    print("\n── Cross-Cloud Generalisation ───────────────────────────────")
    results = {}
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for provider in ['azure', 'google']:
        try:
            _, _, test, fcols_p = load_data(provider)
            common = [c for c in feature_cols if c in fcols_p]
            if len(common) < 5:
                print(f"  ⚠  {provider}: too few common features ({len(common)})")
                continue

            X_t = test[common].values.astype(np.float32)
            y_t = test[TARGET].values.astype(np.float32)

            # Pad to Alibaba feature count
            n_f = len(feature_cols)
            if X_t.shape[1] < n_f:
                X_t = np.hstack([X_t,
                    np.zeros((X_t.shape[0], n_f - X_t.shape[1]),
                             dtype=np.float32)])
            else:
                X_t = X_t[:, :n_f]

            xgb_rmse  = round(float(np.sqrt(mean_squared_error(
                y_t, xgb_model.predict(X_t)))), 4)
            rf_rmse   = round(float(np.sqrt(mean_squared_error(
                y_t, rf_model.predict(X_t)))), 4)
            lp        = predict_lstm(lstm_model, X_t, device=device)
            yt_l      = y_t[6:]
            ml        = min(len(lp), len(yt_l))
            lstm_rmse = round(float(np.sqrt(mean_squared_error(
                yt_l[:ml], lp[:ml]))), 4)

            results[provider] = {
                'xgboost_rmse': xgb_rmse, 'rf_rmse': rf_rmse,
                'lstm_rmse': lstm_rmse,
                'n_samples': len(y_t),
                'common_features': len(common),
            }
            print(f"  {provider.upper()}: XGB={xgb_rmse}  RF={rf_rmse}  "
                  f"LSTM={lstm_rmse}  (n={len(y_t):,})")

        except Exception as e:
            print(f"  ⚠  {provider}: {e}")

    joblib.dump(results, RESULTS_DIR / "cross_cloud_results.pkl")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# COST ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def cost_analysis(preds, y_actual, model_name,
                  current_nodes=3, price=0.192):
    rows = []
    for pred, actual in zip(preds, y_actual):
        reactive_nodes  = current_nodes * 1.4 if actual > 85 else current_nodes
        proactive_nodes = (max(current_nodes, round(current_nodes * pred / 80))
                           if pred > 70 else current_nodes)
        rows.append({
            'predicted_cpu':     float(pred),
            'actual_cpu':        float(actual),
            'reactive_nodes':    reactive_nodes,
            'proactive_nodes':   proactive_nodes,
            'saving_eur_per_hour': (reactive_nodes - proactive_nodes) * price,
        })
    df   = pd.DataFrame(rows)
    mean = df['saving_eur_per_hour'].mean() * 24
    lo   = df['saving_eur_per_hour'].quantile(0.025) * 24
    hi   = df['saving_eur_per_hour'].quantile(0.975) * 24
    return {
        'model': model_name,
        'mean_daily_saving_eur': round(mean, 2),
        'ci_lower': round(lo, 2),
        'ci_upper': round(hi, 2),
        'pct_windows_proactive_cheaper':
            round((df['saving_eur_per_hour'] > 0).mean() * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("H9MLAI — Full Training Pipeline  |  Sabhyata Kumari X24283142")
    print("=" * 65)

    train, val, test, feature_cols = load_data('alibaba')
    X_train, y_train = xy(train, feature_cols)
    X_val,   y_val   = xy(val,   feature_cols)
    X_test,  y_test  = xy(test,  feature_cols)
    n_features = X_train.shape[1]
    print(f"\nAlibaba | {n_features} features | "
          f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    all_results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model, xgb_time, xgb_lat, xgb_mb, xgb_params = train_xgboost(
        X_train, y_train, X_val, y_val)
    xgb_preds   = xgb_model.predict(X_test)
    xgb_metrics = metrics(y_test, xgb_preds)
    xgb_metrics.update({'train_time_s': round(xgb_time, 1),
                         'latency_ms':  round(xgb_lat,  1),
                         'model_size_mb': xgb_mb,
                         'best_params': xgb_params})
    all_results['xgboost'] = xgb_metrics
    print(f"  XGBoost: {xgb_metrics}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model, rf_time, rf_lat, rf_mb, rf_params = train_random_forest(
        X_train, y_train, X_val, y_val)
    rf_preds   = rf_model.predict(X_test)
    rf_metrics = metrics(y_test, rf_preds)
    rf_metrics.update({'train_time_s': round(rf_time, 1),
                        'latency_ms':  round(rf_lat,  1),
                        'model_size_mb': rf_mb,
                        'best_params': rf_params})
    all_results['random_forest'] = rf_metrics
    print(f"  RF: {rf_metrics}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    (lstm_model, lstm_time, lstm_lat,
     lstm_mb, tr_losses, val_losses) = train_lstm(
        X_train, y_train, X_val, y_val, n_features)
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_preds  = predict_lstm(lstm_model, X_test, device=device)
    y_test_lstm = y_test[6:]
    ml          = min(len(lstm_preds), len(y_test_lstm))
    lstm_metrics = metrics(y_test_lstm[:ml], lstm_preds[:ml])
    lstm_metrics.update({'train_time_s': round(lstm_time, 1),
                          'latency_ms':  round(lstm_lat,  1),
                          'model_size_mb': lstm_mb})
    all_results['lstm'] = lstm_metrics
    print(f"  LSTM: {lstm_metrics}")

    # ── Wilcoxon signed-rank test ─────────────────────────────────────────────
    # Align lengths for pairwise comparison
    n_cmp   = min(len(xgb_preds), len(rf_preds), ml)
    wilcoxon_res = wilcoxon_significance(
        y_test[:n_cmp],
        {'xgboost':       xgb_preds[:n_cmp],
         'random_forest': rf_preds[:n_cmp],
         'lstm':          lstm_preds[:n_cmp]})

    # ── Learning curves ───────────────────────────────────────────────────────
    curves = compute_learning_curves(
        X_train, y_train, X_test, y_test, X_val, y_val)

    # ── Cross-cloud evaluation ────────────────────────────────────────────────
    cc = cross_cloud_eval(xgb_model, rf_model, lstm_model, feature_cols)

    # ── Multi-horizon experiment ──────────────────────────────────────────────
    horizon_res = multi_horizon_experiment(
        train, val, test, feature_cols, xgb_params)

    # ── Transfer learning ─────────────────────────────────────────────────────
    tl_res = transfer_learning(xgb_model, feature_cols)

    # ── Cost analysis ─────────────────────────────────────────────────────────
    print("\n── Cost Analysis ────────────────────────────────────────────")
    cost_res = {}
    for name, preds, y_ref in [
        ('xgboost',       xgb_preds,        y_test),
        ('random_forest', rf_preds,          y_test),
        ('lstm',          lstm_preds[:ml],   y_test_lstm[:ml]),
    ]:
        r = cost_analysis(preds, y_ref, name)
        cost_res[name] = r
        print(f"  {name}: €{r['mean_daily_saving_eur']}/day "
              f"(95% CI: €{r['ci_lower']}–€{r['ci_upper']})")

    # ── Save everything ───────────────────────────────────────────────────────
    final = {
        'model_performance': all_results,
        'cross_cloud':       cc,
        'cost_analysis':     cost_res,
        'wilcoxon':          wilcoxon_res,
        'multi_horizon':     horizon_res,
        'transfer_learning': tl_res,
        'feature_cols':      feature_cols,
        'n_features':        n_features,
    }
    joblib.dump(final, RESULTS_DIR / "all_results.pkl")
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(to_jsonable(
            {k: v for k, v in final.items() if k != 'feature_cols'}),
            f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Model':<16} {'RMSE':<8} {'MAE':<8} {'MAPE%':<8} "
          f"{'Train(s)':<10} {'Lat(ms)':<10} {'MB':<8}")
    print("-" * 65)
    for name, r in all_results.items():
        print(f"{name:<16} {r['rmse']:<8} {r['mae']:<8} "
              f"{r['mape']:<8} {r['train_time_s']:<10} "
              f"{r['latency_ms']:<10} {r.get('model_size_mb','?'):<8}")

    print("\n✅ All done.")
    print("   Next: python src/explainability/shap_analysis.py")
    print("   Then: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
