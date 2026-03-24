"""
Train All Models — H9MLAI Project
Sabhyata Kumari | X24283142

Trains XGBoost, Random Forest, and LSTM on Alibaba data.
Cross-evaluates all models on Azure and Google.
Saves all trained models, results, and learning curves.

Run:  python src/models/train_all.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

MODELS_DIR = Path("data/models")
RESULTS_DIR = Path("data/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'target_cpu'   # primary target (also train mem variant)
SEED   = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(provider='alibaba'):
    base = Path("data/processed")
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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4),
            'mape': round(mape, 4)}


def to_jsonable(obj):
    """Recursively convert numpy/pandas values into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n── XGBoost ─────────────────────────────────────")

    # Hyperparameter search with Optuna
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
                'random_state': SEED,
                'n_jobs': -1,
            }
            m = xgb.XGBRegressor(**params, verbosity=0)
            m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20,
                  verbose=False)
            preds = m.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        best_params = study.best_params
        best_params['random_state'] = SEED
        best_params['n_jobs'] = -1
        print(f"  Best params: {best_params}")

    except ImportError:
        print("  Optuna not available — using defaults")
        best_params = {
            'n_estimators': 300, 'max_depth': 6,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'random_state': SEED, 'n_jobs': -1
        }

    # Final training
    t0 = time.time()
    model = xgb.XGBRegressor(**best_params, verbosity=0)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=20,
              verbose=False)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Inference latency
    t0 = time.time()
    for _ in range(10):
        preds = model.predict(X_val)
    latency_ms = (time.time() - t0) / 10 * 1000
    print(f"  Inference latency: {latency_ms:.1f}ms per batch")

    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
    return model, train_time, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n── Random Forest ───────────────────────────────")

    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    t0 = time.time()
    for _ in range(10):
        preds = model.predict(X_val)
    latency_ms = (time.time() - t0) / 10 * 1000
    print(f"  Inference latency: {latency_ms:.1f}ms per batch")

    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    return model, train_time, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — LSTM
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
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last timestep
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    print("\n── LSTM ─────────────────────────────────────────")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    SEQ_LEN    = 6
    BATCH_SIZE = 256
    MAX_EPOCHS = 60
    PATIENCE   = 10

    train_ds = TimeSeriesDataset(X_train, y_train, SEQ_LEN)
    val_ds   = TimeSeriesDataset(X_val,   y_val,   SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, drop_last=False)

    model = LSTMForecaster(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader),   1)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "lstm_best.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={avg_train:.4f} "
                  f"val={avg_val:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Load best weights
    model.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt",
                                     map_location=device))
    torch.save({'model_state': model.state_dict(),
                'n_features': n_features,
                'seq_len': SEQ_LEN,
                'train_losses': train_losses,
                'val_losses': val_losses},
               MODELS_DIR / "lstm_model.pt")

    # Inference latency
    model.eval()
    sample = torch.tensor(X_val[:256 * SEQ_LEN].reshape(256, SEQ_LEN, n_features),
                          dtype=torch.float32).to(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)
    latency_ms = (time.time() - t0) / 10 * 1000
    print(f"  Inference latency: {latency_ms:.1f}ms per batch")

    return model, train_time, latency_ms, train_losses, val_losses


def predict_lstm(model, X, seq_len=6, batch_size=512, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    n = len(X) - seq_len
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = np.array([X[j:j+seq_len] for j in range(i, end)])
        t = torch.tensor(batch, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds.extend(model(t).cpu().numpy().tolist())
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def compute_learning_curves(X_train, y_train, X_test, y_test,
                             X_val, y_val):
    """Train each model on 10/25/50/75/100% of data, record test RMSE."""
    print("\n── Learning Curves ─────────────────────────────")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    curves = {'fraction': fractions, 'xgb': [], 'rf': [], 'lstm': []}

    n_features = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for frac in fractions:
        n = int(len(X_train) * frac)
        X_sub, y_sub = X_train[:n], y_train[:n]
        print(f"  Fraction {frac:.0%} ({n:,} samples)")

        # XGBoost
        m_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.05, random_state=SEED,
                                   verbosity=0, n_jobs=-1)
        m_xgb.fit(X_sub, y_sub, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=15, verbose=False)
        curves['xgb'].append(
            round(np.sqrt(mean_squared_error(y_test, m_xgb.predict(X_test))), 4))

        # Random Forest
        m_rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                                      random_state=SEED, n_jobs=-1)
        m_rf.fit(X_sub, y_sub)
        curves['rf'].append(
            round(np.sqrt(mean_squared_error(y_test, m_rf.predict(X_test))), 4))

        # LSTM (quick version for learning curve)
        ds = TimeSeriesDataset(X_sub, y_sub, seq_len=6)
        if len(ds) < 64:
            curves['lstm'].append(None)
            continue
        loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
        m_lstm = LSTMForecaster(n_features=n_features).to(device)
        opt    = torch.optim.Adam(m_lstm.parameters(), lr=1e-3)
        crit   = nn.MSELoss()
        for _ in range(15):   # quick 15 epochs for curve
            m_lstm.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(m_lstm(Xb), yb)
                loss.backward()
                opt.step()
        lstm_preds = predict_lstm(m_lstm, X_test, device=device)
        y_test_lstm = y_test[6:]
        min_len = min(len(lstm_preds), len(y_test_lstm))
        rmse = round(np.sqrt(mean_squared_error(
            y_test_lstm[:min_len], lstm_preds[:min_len])), 4)
        curves['lstm'].append(rmse)
        print(f"    XGB={curves['xgb'][-1]}  RF={curves['rf'][-1]}  LSTM={rmse}")

    joblib.dump(curves, RESULTS_DIR / "learning_curves.pkl")
    print("  ✅ Learning curves saved")
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CLOUD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_cloud_eval(xgb_model, rf_model, lstm_model, feature_cols):
    """Evaluate Alibaba-trained models on Azure and Google test sets."""
    print("\n── Cross-Cloud Generalisation ──────────────────")
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for provider in ['azure', 'google']:
        try:
            _, _, test, fcols_other = load_data(provider)

            # Align features (use common columns, fill missing with 0)
            common = [c for c in feature_cols if c in fcols_other]
            if len(common) < 5:
                print(f"  ⚠  {provider}: too few common features ({len(common)})")
                continue

            X_test = test[common].values.astype(np.float32)
            y_test = test[TARGET].values.astype(np.float32)

            # Pad to match training feature count
            if X_test.shape[1] < len(feature_cols):
                pad = np.zeros((X_test.shape[0],
                                len(feature_cols) - X_test.shape[1]),
                               dtype=np.float32)
                X_test_padded = np.hstack([X_test, pad])
            else:
                X_test_padded = X_test[:, :len(feature_cols)]

            # XGBoost
            xgb_preds = xgb_model.predict(X_test_padded)
            xgb_rmse  = round(np.sqrt(mean_squared_error(y_test, xgb_preds)), 4)

            # RF
            rf_preds  = rf_model.predict(X_test_padded)
            rf_rmse   = round(np.sqrt(mean_squared_error(y_test, rf_preds)), 4)

            # LSTM
            lstm_preds   = predict_lstm(lstm_model, X_test_padded, device=device)
            y_test_lstm  = y_test[6:]
            min_len = min(len(lstm_preds), len(y_test_lstm))
            lstm_rmse = round(np.sqrt(mean_squared_error(
                y_test_lstm[:min_len], lstm_preds[:min_len])), 4)

            results[provider] = {
                'xgboost_rmse': xgb_rmse,
                'rf_rmse':      rf_rmse,
                'lstm_rmse':    lstm_rmse,
                'n_samples':    len(y_test),
                'common_features': len(common)
            }
            print(f"  {provider.upper()}: XGB={xgb_rmse}  RF={rf_rmse}  "
                  f"LSTM={lstm_rmse}  (n={len(y_test):,})")

        except Exception as e:
            print(f"  ⚠  {provider}: {e}")

    joblib.dump(results, RESULTS_DIR / "cross_cloud_results.pkl")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# COST ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def cost_analysis(xgb_preds, y_actual, model_name,
                  current_nodes=3,
                  price_eur_per_node_hour=0.192,   # AWS m5.xlarge EU-West-1
                  provisioning_delay_hours=2/60):   # 2-minute delay

    """
    Calculate daily EUR saving: proactive ML scaling vs reactive threshold.

    Reactive baseline:
    - Waits until CPU > 85% (threshold breach)
    - Over-provisions by 40% safety buffer
    - Node not available for 2 minutes after trigger

    Proactive ML:
    - Uses ML prediction 30 min ahead
    - Right-sizes exactly to predicted demand
    - Node ready before breach
    """
    results_per_window = []

    for pred, actual in zip(xgb_preds, y_actual):
        # Reactive: over-provision when breached
        if actual > 85:
            reactive_nodes = current_nodes * 1.4
        else:
            reactive_nodes = current_nodes
        reactive_cost_hour = reactive_nodes * price_eur_per_node_hour

        # Proactive: right-size based on ML prediction
        if pred > 70:   # pre-emptively scale when forecast high
            proactive_nodes = max(current_nodes,
                                  round(current_nodes * pred / 80))
        else:
            proactive_nodes = current_nodes
        proactive_cost_hour = proactive_nodes * price_eur_per_node_hour

        saving_hour = reactive_cost_hour - proactive_cost_hour
        results_per_window.append({
            'predicted_cpu': pred,
            'actual_cpu': actual,
            'reactive_nodes': reactive_nodes,
            'proactive_nodes': proactive_nodes,
            'saving_eur_per_hour': saving_hour,
        })

    df = pd.DataFrame(results_per_window)
    daily_saving_mean = df['saving_eur_per_hour'].mean() * 24
    daily_saving_ci   = (
        df['saving_eur_per_hour'].quantile(0.025) * 24,
        df['saving_eur_per_hour'].quantile(0.975) * 24,
    )

    result = {
        'model': model_name,
        'mean_daily_saving_eur': round(daily_saving_mean, 2),
        'ci_lower': round(daily_saving_ci[0], 2),
        'ci_upper': round(daily_saving_ci[1], 2),
        'pct_windows_proactive_cheaper':
            round((df['saving_eur_per_hour'] > 0).mean() * 100, 1),
        'detail_df': df,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — TRAIN EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("H9MLAI — Training All Models")
    print("=" * 60)

    # Load Alibaba (primary training dataset)
    train, val, test, feature_cols = load_data('alibaba')
    X_train, y_train = xy(train, feature_cols)
    X_val,   y_val   = xy(val,   feature_cols)
    X_test,  y_test  = xy(test,  feature_cols)
    n_features = X_train.shape[1]
    print(f"\nAlibaba: {n_features} features | "
          f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    all_results = {}

    # ── Train XGBoost ──────────────────────────────────────────────
    xgb_model, xgb_time, xgb_lat = train_xgboost(
        X_train, y_train, X_val, y_val)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = metrics(y_test, xgb_preds)
    xgb_metrics.update({'train_time_s': round(xgb_time, 1),
                         'latency_ms': round(xgb_lat, 1)})
    all_results['xgboost'] = xgb_metrics
    print(f"  Results: {xgb_metrics}")

    # ── Train Random Forest ────────────────────────────────────────
    rf_model, rf_time, rf_lat = train_random_forest(
        X_train, y_train, X_val, y_val)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = metrics(y_test, rf_preds)
    rf_metrics.update({'train_time_s': round(rf_time, 1),
                        'latency_ms': round(rf_lat, 1)})
    all_results['random_forest'] = rf_metrics
    print(f"  Results: {rf_metrics}")

    # ── Train LSTM ─────────────────────────────────────────────────
    lstm_model, lstm_time, lstm_lat, tr_losses, val_losses = train_lstm(
        X_train, y_train, X_val, y_val, n_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_preds = predict_lstm(lstm_model, X_test, device=device)
    y_test_lstm = y_test[6:]
    min_len = min(len(lstm_preds), len(y_test_lstm))
    lstm_metrics = metrics(y_test_lstm[:min_len], lstm_preds[:min_len])
    lstm_metrics.update({'train_time_s': round(lstm_time, 1),
                          'latency_ms': round(lstm_lat, 1)})
    all_results['lstm'] = lstm_metrics
    print(f"  Results: {lstm_metrics}")
    joblib.dump({'train': tr_losses, 'val': val_losses},
                RESULTS_DIR / "lstm_loss_curves.pkl")

    # ── Learning Curves ────────────────────────────────────────────
    curves = compute_learning_curves(
        X_train, y_train, X_test, y_test, X_val, y_val)

    # ── Cross-Cloud ────────────────────────────────────────────────
    cc = cross_cloud_eval(xgb_model, rf_model, lstm_model, feature_cols)

    # ── Cost Analysis ──────────────────────────────────────────────
    print("\n── Cost Analysis ───────────────────────────────")
    cost_results = {}
    for name, preds in [('xgboost', xgb_preds),
                         ('random_forest', rf_preds),
                         ('lstm', lstm_preds[:min_len])]:
        y_ref = y_test if name != 'lstm' else y_test_lstm[:min_len]
        res = cost_analysis(preds, y_ref, name)
        cost_results[name] = {k: v for k, v in res.items() if k != 'detail_df'}
        print(f"  {name}: €{res['mean_daily_saving_eur']}/day "
              f"(95% CI: €{res['ci_lower']}–€{res['ci_upper']})")

    # ── Save all results ───────────────────────────────────────────
    final = {
        'model_performance': all_results,
        'cross_cloud': cc,
        'cost_analysis': cost_results,
        'feature_cols': feature_cols,
        'n_features': n_features,
    }
    joblib.dump(final, RESULTS_DIR / "all_results.pkl")
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(
            to_jsonable({k: v for k, v in final.items()
                         if k != 'feature_cols'}),
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'MAPE%':<8} "
          f"{'Train(s)':<10} {'Latency(ms)':<12}")
    print("-" * 60)
    for model_name, r in all_results.items():
        print(f"{model_name:<15} {r['rmse']:<8} {r['mae']:<8} "
              f"{r['mape']:<8} {r['train_time_s']:<10} {r['latency_ms']:<12}")

    print("\n✅ All models trained and results saved.")
    print("   Next: python src/explainability/shap_analysis.py")
    print("   Then: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
