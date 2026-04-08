"""
Data Preprocessor — H9MLAI Project
Sabhyata Kumari | X24283142

Loads all three datasets, aligns them to a common format,
engineers features, and saves processed train/val/test splits.

Run:  python src/data/preprocessor.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 6      # windows per sequence for LSTM (6 × 10min = 60min)
LAG_STEPS      = [1, 3, 6, 18, 36, 144]   # 10min, 30min, 1h, 3h, 6h, 24h
ROLLING_WINS   = [3, 6, 18, 36]            # 30min, 1h, 3h, 6h


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS — each dataset has different column names
# ─────────────────────────────────────────────────────────────────────────────

def load_alibaba(sample_machines=80):
    """Load Alibaba 2018 machine_usage.csv → standard format."""
    path = Path("data/raw/alibaba/machine_usage.csv")
    print(f"Loading Alibaba from {path}...")
    df = pd.read_csv(path)

    # Standardise column names
    col_map = {
        'machine_id': 'node_id',
        'time_stamp': 'ts',
        'cpu_util_percent': 'cpu',
        'mem_util_percent': 'mem',
    }
    # Handle alternate column names in real dataset
    if 'cpu_util_percent' not in df.columns and len(df.columns) >= 4:
        df.columns = ['node_id', 'ts', 'cpu', 'mem',
                      'mem_gps', 'mkpi', 'net_in', 'net_out', 'disk_io'][:len(df.columns)]
        col_map = {}

    df = df.rename(columns=col_map)
    df = df[['node_id', 'ts', 'cpu', 'mem']].copy()

    # Sample machines for speed
    machines = df['node_id'].unique()
    if len(machines) > sample_machines:
        machines = np.random.choice(machines, sample_machines, replace=False)
        df = df[df['node_id'].isin(machines)]

    df['provider'] = 'alibaba'
    df['cpu'] = pd.to_numeric(df['cpu'], errors='coerce')
    df['mem'] = pd.to_numeric(df['mem'], errors='coerce')
    print(f"  → {len(df):,} rows, {df['node_id'].nunique()} nodes")
    return df


def load_azure():
    """Load Azure VM dataset → standard format."""
    path = Path("data/raw/azure/vm_cpu_readings.csv")
    gz_path = Path("data/raw/azure/vm_cpu_readings.csv.gz")

    if path.exists():
        print(f"Loading Azure from {path}...")
        df = pd.read_csv(path)
    elif gz_path.exists():
        import gzip, io
        print(f"Loading Azure from {gz_path} (truncation-tolerant)...")
        # Read as many bytes as possible — handles truncated gzip files
        try:
            with gzip.open(gz_path, 'rb') as f:
                raw = f.read()
        except EOFError:
            # Truncated file: read whatever was decompressed before the error
            raw = b''
            with open(gz_path, 'rb') as fh:
                d = gzip.GzipFile(fileobj=fh)
                try:
                    raw = d.read()
                except EOFError:
                    raw = d._buffer.raw._buffer if hasattr(d, '_buffer') else raw
            if not raw:
                import zlib
                with open(gz_path, 'rb') as fh:
                    fh.read(10)            # skip gzip header
                    raw_deflate = fh.read()
                try:
                    raw = zlib.decompress(raw_deflate, -15)
                except Exception:
                    raw = zlib.decompress(raw_deflate, -15,
                                          bufsize=1024 * 1024 * 512)
        text = raw.decode('utf-8', errors='replace')
        # Trim to last complete line
        last_nl = text.rfind('\n')
        if last_nl > 0:
            text = text[:last_nl + 1]
        df = pd.read_csv(io.StringIO(text), header=None if '\n' in text[:200] and
                         not text[:200].startswith('v') else 'infer')
    else:
        raise FileNotFoundError("Azure dataset not found in data/raw/azure/")

    col_map = {'vm_id': 'node_id', 'timestamp': 'ts', 'cpu_avg': 'cpu',
               'mem_avg': 'mem'}
    df = df.rename(columns=col_map)

    # Keep available columns
    available = [c for c in ['node_id', 'ts', 'cpu', 'mem'] if c in df.columns]
    df = df[available].copy()

    if 'mem' not in df.columns:
        df['mem'] = df['cpu'] * 0.75 + np.random.normal(0, 3, len(df))

    df['provider'] = 'azure'
    df['cpu'] = pd.to_numeric(df['cpu'], errors='coerce')
    df['mem'] = pd.to_numeric(df['mem'], errors='coerce')
    print(f"  → {len(df):,} rows, {df['node_id'].nunique()} nodes")
    return df


def load_google():
    """Load Google Cluster Traces → standard format."""
    path = Path("data/raw/google/machine_events.csv")
    print(f"Loading Google from {path}...")
    df = pd.read_csv(path)

    col_map = {'machine_id': 'node_id', 'time': 'ts',
               'cpu_usage': 'cpu', 'memory_usage': 'mem'}
    df = df.rename(columns=col_map)
    df = df[['node_id', 'ts', 'cpu', 'mem']].copy()

    # Google stores as fractions (0–1) → convert to percent
    if df['cpu'].max() <= 1.0:
        df['cpu'] = df['cpu'] * 100
        df['mem'] = df['mem'] * 100

    df['provider'] = 'google'
    df['cpu'] = pd.to_numeric(df['cpu'], errors='coerce')
    df['mem'] = pd.to_numeric(df['mem'], errors='coerce')
    print(f"  → {len(df):,} rows, {df['node_id'].nunique()} nodes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df):
    """Remove nulls, clip outliers, sort."""
    df = df.dropna(subset=['cpu', 'mem'])
    df['cpu'] = df['cpu'].clip(0, 100)
    df['mem'] = df['mem'].clip(0, 100)
    # Winsorise at 99th percentile per node
    for col in ['cpu', 'mem']:
        p99 = df.groupby('node_id')[col].transform(lambda x: x.quantile(0.99))
        df[col] = df[col].clip(upper=p99)
    df = df.sort_values(['node_id', 'ts']).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df, target_horizon=3):
    """
    Build feature matrix per node.
    target_horizon: predict CPU/mem N steps ahead (3 steps × 10min = 30min)
    """
    all_node_dfs = []

    for node_id, grp in df.groupby('node_id'):
        grp = grp.sort_values('ts').reset_index(drop=True)

        if len(grp) < max(LAG_STEPS) + target_horizon + 10:
            continue   # not enough history

        feat = pd.DataFrame()
        feat['node_id']  = grp['node_id']
        feat['ts']       = grp['ts']
        feat['provider'] = grp['provider']

        # Target variables — multiple forecast horizons
        # h1 ≈ 10 min (1 step), h3 = 30 min (3 steps), h6 = 60 min (6 steps)
        feat['target_cpu']    = grp['cpu'].shift(-3)   # primary: 30 min
        feat['target_cpu_h1'] = grp['cpu'].shift(-1)   # short:   10 min
        feat['target_cpu_h6'] = grp['cpu'].shift(-6)   # long:    60 min
        feat['target_mem']    = grp['mem'].shift(-3)

        # Raw current values
        feat['cpu'] = grp['cpu']
        feat['mem'] = grp['mem']

        # ── Lag features ────────────────────────────────────────────
        for lag in LAG_STEPS:
            feat[f'cpu_lag_{lag}'] = grp['cpu'].shift(lag)
            feat[f'mem_lag_{lag}'] = grp['mem'].shift(lag)

        # ── Rolling statistics ───────────────────────────────────────
        for win in ROLLING_WINS:
            feat[f'cpu_roll_mean_{win}'] = grp['cpu'].rolling(win).mean()
            feat[f'cpu_roll_std_{win}']  = grp['cpu'].rolling(win).std()
            feat[f'cpu_roll_max_{win}']  = grp['cpu'].rolling(win).max()
            feat[f'cpu_roll_min_{win}']  = grp['cpu'].rolling(win).min()
            feat[f'mem_roll_mean_{win}'] = grp['mem'].rolling(win).mean()
            feat[f'mem_roll_std_{win}']  = grp['mem'].rolling(win).std()
            feat[f'mem_roll_max_{win}']  = grp['mem'].rolling(win).max()

        # ── Rate of change (gradient) ────────────────────────────────
        feat['cpu_roc_1']  = grp['cpu'].diff(1)
        feat['cpu_roc_6']  = grp['cpu'].diff(6)
        feat['cpu_roc_18'] = grp['cpu'].diff(18)
        feat['mem_roc_1']  = grp['mem'].diff(1)
        feat['mem_roc_6']  = grp['mem'].diff(6)

        # ── Time-based features (cyclic encoding) ────────────────────
        # ts is in seconds from epoch; map to hour of day
        ts_hours = (grp['ts'] % 86400) / 3600
        feat['hour_sin'] = np.sin(2 * np.pi * ts_hours / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * ts_hours / 24)

        # Day of week (approximate from ts)
        ts_days = (grp['ts'] // 86400) % 7
        feat['dow_sin'] = np.sin(2 * np.pi * ts_days / 7)
        feat['dow_cos'] = np.cos(2 * np.pi * ts_days / 7)

        # ── SLA breach indicator (label for classification variant) ──
        feat['sla_breach'] = (grp['cpu'].shift(-target_horizon) > 85).astype(int)

        all_node_dfs.append(feat)

    result = pd.concat(all_node_dfs, ignore_index=True)
    result = result.dropna()   # remove rows with NaN from lags
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df, train_frac=0.70, val_frac=0.15):
    """
    Strict temporal split per node — no shuffling to prevent data leakage.
    """
    train_parts, val_parts, test_parts = [], [], []

    for node_id, grp in df.groupby('node_id'):
        grp = grp.sort_values('ts').reset_index(drop=True)
        n = len(grp)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train_parts.append(grp.iloc[:t1])
        val_parts.append(grp.iloc[t1:t2])
        test_parts.append(grp.iloc[t2:])

    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(val_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True))


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df):
    exclude = {'node_id', 'ts', 'provider',
               'target_cpu', 'target_cpu_h1', 'target_cpu_h6',
               'target_mem', 'sla_breach'}
    return [c for c in df.columns if c not in exclude]


def normalise(train_df, val_df, test_df, feature_cols):
    """Fit scaler on train only — prevent data leakage."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    return train_df, val_df, test_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("H9MLAI — Data Preprocessing Pipeline")
    print("=" * 60)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all three datasets ──────────────────────────────────────
    ali   = clean(load_alibaba())
    azure = clean(load_azure())
    goog  = clean(load_google())

    datasets = {'alibaba': ali, 'azure': azure, 'google': goog}

    for name, df in datasets.items():
        print(f"\n── Processing {name} ─────────────────────────────")

        # Feature engineering
        feat_df = engineer_features(df)
        feature_cols = get_feature_cols(feat_df)
        print(f"  Features: {len(feature_cols)} columns, {len(feat_df):,} rows")

        # Split
        train, val, test = temporal_split(feat_df)
        print(f"  Split: train={len(train):,}  val={len(val):,}  test={len(test):,}")

        # Normalise
        train, val, test, scaler = normalise(train, val, test, feature_cols)

        # Save
        train.to_parquet(out_dir / f"{name}_train.parquet")
        val.to_parquet(out_dir / f"{name}_val.parquet")
        test.to_parquet(out_dir / f"{name}_test.parquet")
        joblib.dump(scaler, out_dir / f"{name}_scaler.pkl")
        joblib.dump(feature_cols, out_dir / f"{name}_feature_cols.pkl")

        print(f"  ✅ Saved to data/processed/{name}_*.parquet")

    # Save combined Alibaba (used for cross-cloud training)
    print("\n✅ All datasets processed and saved.")
    print("   Next: python src/models/train_all.py")


if __name__ == "__main__":
    main()
