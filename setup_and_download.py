"""
H9MLAI PROJECT SETUP SCRIPT
============================
Student: Sabhyata Kumari | X24283142
NCI Dublin — MSc Artificial Intelligence

Run this FIRST before anything else:
    python setup_and_download.py

What this does:
1. Installs all required packages
2. Downloads real Alibaba 2018 Cluster Trace data
3. Downloads real Azure Public Dataset data
4. Downloads real Google cluster trace data
5. Verifies the datasets are real (not synthetic)
6. Runs preprocessing automatically
"""

import os
import sys
import subprocess
import gzip
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install packages
# ─────────────────────────────────────────────────────────────────────────────
def install_packages():
    print("Installing packages from requirements.txt...")
    req = Path("requirements.txt")
    if not req.exists():
        raise RuntimeError("requirements.txt not found in project root")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(req)
    ])
    print("✅ Packages installed")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Download datasets
# ─────────────────────────────────────────────────────────────────────────────
def download_file(url, dest_path, label):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✅ Already downloaded: {label}")
        return dest
    print(f"⬇  Downloading {label}...")
    try:
        import requests
        from tqdm import tqdm
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        first_chunk = b""
        with open(dest, 'wb') as f, tqdm(total=total, unit='B',
                                          unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if not first_chunk and chunk:
                    first_chunk = chunk[:64]
                f.write(chunk)
                bar.update(len(chunk))
        # Guard against downloading an HTML/XML error page as a .tar.gz/.csv.gz file.
        if first_chunk.lstrip().startswith((b"<", b"<?xml")):
            dest.unlink(missing_ok=True)
            raise RuntimeError("download returned HTML/XML instead of dataset bytes")
        print(f"✅ Downloaded: {label}")
        return dest
    except Exception as e:
        raise RuntimeError(
            f"Could not download {label}: {e}\n"
            f"URL: {url}\n"
            f"Expected path: {dest_path}"
        )


def _is_synthetic_alibaba(csv_path):
    import pandas as pd
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path, nrows=200)
    if 'machine_id' not in df.columns:
        return False
    machine_ids = df['machine_id'].astype(str).head(30)
    return machine_ids.str.match(r'^m_\d{4}$').mean() > 0.8


def _is_synthetic_azure(csv_path):
    import pandas as pd
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path, nrows=200)
    if 'vm_id' not in df.columns:
        return False
    vm_ids = df['vm_id'].astype(str).head(30)
    return vm_ids.str.match(r'^vm_\d{3}$').mean() > 0.8


def _is_synthetic_google(csv_path):
    import pandas as pd
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path, nrows=200)
    if 'machine_id' not in df.columns:
        return False
    machine_ids = df['machine_id'].astype(str).head(30)
    return machine_ids.str.match(r'^g_\d{4}$').mean() > 0.8


def assert_real_data():
    checks = [
        ('Alibaba', Path('data/raw/alibaba/machine_usage.csv'), _is_synthetic_alibaba),
        ('Azure', Path('data/raw/azure/vm_cpu_readings.csv'), _is_synthetic_azure),
        ('Google', Path('data/raw/google/machine_events.csv'), _is_synthetic_google),
    ]

    problems = []
    for name, path, is_synthetic in checks:
        if not path.exists():
            problems.append(f"{name}: missing file at {path}")
            continue
        if is_synthetic(path):
            problems.append(f"{name}: synthetic-looking dataset at {path}")

    if problems:
        raise RuntimeError(
            "Real data requirement failed:\n- " + "\n- ".join(problems)
        )


def run_preprocessing():
    print("\n── Running Preprocessing ─────────────────────")
    subprocess.check_call([sys.executable, "src/data/preprocessor.py"])
    print("✅ Preprocessing completed")


def download_alibaba():
    """
    Alibaba 2018 Cluster Trace
    Full data: https://github.com/alibaba/clusterdata
    We download the machine_usage.tar.gz from the v2018 release.
    The file contains per-machine CPU and memory every 10 seconds.
    """
    base = Path("data/raw/alibaba")
    base.mkdir(parents=True, exist_ok=True)

    tar_path = Path("data/raw/alibaba/machine_usage.tar.gz")
    csv_path = Path("data/raw/alibaba/machine_usage.csv")

    if csv_path.exists() and not _is_synthetic_alibaba(csv_path):
        print("✅ Already downloaded: Alibaba real dataset")
        return

    if csv_path.exists() and _is_synthetic_alibaba(csv_path):
        print("⚠️ Found synthetic Alibaba file. Replacing with real dataset...")
        csv_path.unlink(missing_ok=True)

    # Official link referenced by the Alibaba GitHub repo docs.
    url = (
        "https://aliopentrace.oss-cn-beijing.aliyuncs.com/"
        "v2018Traces/machine_usage.tar.gz"
    )
    download_file(url, str(tar_path), "Alibaba machine_usage.tar.gz")

    # Extract
    if tar_path.exists() and not csv_path.exists():
        print("Extracting Alibaba data...")
        import tarfile
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall("data/raw/alibaba/")

        if not csv_path.exists():
            matches = list(base.rglob("machine_usage.csv"))
            if matches:
                shutil.move(str(matches[0]), str(csv_path))

        if not csv_path.exists():
            raise RuntimeError(
                "Alibaba download succeeded but machine_usage.csv was not found after extraction"
            )

        print("✅ Alibaba data extracted")


def create_synthetic_alibaba():
    """
    Create a realistic synthetic dataset matching Alibaba 2018 format.
    Columns: machine_id, time_stamp, cpu_util_percent, mem_util_percent,
             mem_gps, mkpi, net_in, net_out, disk_io_percent
    500,000 rows across 100 machines over 8 days (10-sec intervals)
    """
    import pandas as pd
    import numpy as np

    print("Generating synthetic Alibaba-format data (500K rows)...")
    np.random.seed(42)

    n_machines = 100
    # 8 days × 24h × 360 intervals per hour (10-sec) = 69,120 per machine
    # Use 5,000 per machine for speed = 500,000 total
    n_intervals = 5000
    total_rows = n_machines * n_intervals

    machine_ids = np.repeat([f"m_{i:04d}" for i in range(n_machines)],
                             n_intervals)

    # Generate realistic workload patterns
    t = np.tile(np.arange(n_intervals), n_machines)
    hour_of_day = (t * 10 / 3600) % 24  # 10-sec intervals → hours

    # Daily cycle: peak at business hours (9-17), low at night
    daily_cycle = 0.3 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    daily_cycle = np.clip(daily_cycle, 0.1, 0.9)

    # Add machine-specific baseline (some machines consistently busier)
    machine_baseline = np.repeat(np.random.uniform(0.1, 0.5, n_machines),
                                  n_intervals)

    # CPU with noise, spikes, and trends
    cpu = (daily_cycle + machine_baseline +
           np.random.normal(0, 0.08, total_rows))
    # Add occasional spikes (anomalies for training variety)
    spike_mask = np.random.random(total_rows) < 0.02
    cpu[spike_mask] += np.random.uniform(0.3, 0.6, spike_mask.sum())
    cpu = np.clip(cpu * 100, 0.1, 99.9)

    # Memory — more stable than CPU, gradual trends
    mem = (0.4 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24) +
           machine_baseline * 0.5 +
           np.random.normal(0, 0.05, total_rows))
    mem = np.clip(mem * 100, 5, 95)

    timestamps = np.tile(np.arange(0, n_intervals * 10, 10), n_machines)

    df = pd.DataFrame({
        'machine_id': machine_ids,
        'time_stamp': timestamps,
        'cpu_util_percent': cpu.round(2),
        'mem_util_percent': mem.round(2),
        'mem_gps': (mem * 0.08).round(3),   # GB/s memory bandwidth proxy
        'mkpi': np.random.uniform(0, 1, total_rows).round(3),
        'net_in': np.random.exponential(10, total_rows).round(2),
        'net_out': np.random.exponential(8, total_rows).round(2),
        'disk_io_percent': np.clip(
            np.random.exponential(5, total_rows), 0, 80).round(2),
    })

    Path("data/raw/alibaba").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/alibaba/machine_usage.csv", index=False)
    print(f"✅ Synthetic Alibaba data created: {len(df):,} rows")


def download_azure():
    """
    Azure Public Dataset — VM CPU and memory utilisation
    Source: https://github.com/Azure/AzurePublicDataset
    """
    azure_dir = Path("data/raw/azure")
    azure_dir.mkdir(parents=True, exist_ok=True)

    # Azure hosts VM traces on GitHub releases
    url = ("https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
           "master/AzurePublicDatasetLinksV2.txt")

    csv_path = Path("data/raw/azure/vm_cpu_readings.csv")
    gz_path = Path("data/raw/azure/vm_cpu_readings.csv.gz")

    if csv_path.exists() and not _is_synthetic_azure(csv_path):
        print("✅ Already downloaded: Azure real dataset")
        return

    if csv_path.exists() and _is_synthetic_azure(csv_path):
        print("⚠️ Found synthetic Azure file. Replacing with real dataset...")
        csv_path.unlink(missing_ok=True)

    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    links = [l.strip() for l in r.text.split('\n')
             if 'vm_cpu_readings-file-1-of' in l and l.strip()]
    if not links:
        raise RuntimeError("Could not find Azure vm_cpu_readings link in AzurePublicDatasetLinksV2.txt")

    download_file(links[0], str(gz_path), "Azure VM CPU readings (part 1)")

    with gzip.open(gz_path, 'rb') as f_in:
        with open(csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("✅ Azure data decompressed")


def download_google():
    """
    Google Cluster Data from the source linked in google/cluster-data docs.
    We convert it to the project's expected machine_events.csv schema.
    """
    import pandas as pd

    google_dir = Path("data/raw/google")
    google_dir.mkdir(parents=True, exist_ok=True)

    out_csv = google_dir / "machine_events.csv"
    source_gz = google_dir / "google_cluster_data_v1.csv.gz"

    if out_csv.exists() and not _is_synthetic_google(out_csv):
        print("✅ Already downloaded: Google real dataset")
        return

    if out_csv.exists() and _is_synthetic_google(out_csv):
        print("⚠️ Found synthetic Google file. Replacing with real dataset...")
        out_csv.unlink(missing_ok=True)

    source_url = (
        "https://commondatastorage.googleapis.com/"
        "clusterdata-misc/google-cluster-data-1.csv.gz"
    )

    download_file(source_url, str(source_gz), "Google ClusterData v1 (csv.gz)")

    print("Converting Google dataset to machine_events.csv...")
    df = pd.read_csv(source_gz, sep=r"\s+", engine="python")

    expected = [
        "Time", "ParentID", "TaskID",
        "JobType", "NrmlTaskCores", "NrmlTaskMem"
    ]
    if any(c not in df.columns for c in expected):
        if df.shape[1] < 6:
            raise RuntimeError("Unexpected Google schema: fewer than 6 columns")
        df = df.iloc[:, :6].copy()
        df.columns = expected

    out = pd.DataFrame({
        "time": pd.to_numeric(df["Time"], errors="coerce"),
        "machine_id": (
            df["ParentID"].astype(str).str.strip() + "_" +
            df["TaskID"].astype(str).str.strip()
        ),
        "cpu_usage": pd.to_numeric(df["NrmlTaskCores"], errors="coerce") * 100,
        "memory_usage": pd.to_numeric(df["NrmlTaskMem"], errors="coerce") * 100,
    }).dropna(subset=["time", "cpu_usage", "memory_usage"])

    out["cpu_request"] = out["cpu_usage"]
    out["memory_request"] = out["memory_usage"]
    out.to_csv(out_csv, index=False)
    print(f"✅ Google data converted: {len(out):,} rows")


def create_synthetic_azure():
    """
    Synthetic Azure VM dataset.
    Columns: vm_id, timestamp, cpu_avg, cpu_max, mem_avg
    Different workload pattern from Alibaba (enterprise VM vs batch jobs)
    """
    import pandas as pd
    import numpy as np

    print("Generating synthetic Azure-format data (200K rows)...")
    np.random.seed(123)

    n_vms = 50
    n_intervals = 4000
    total = n_vms * n_intervals

    vm_ids = np.repeat([f"vm_{i:03d}" for i in range(n_vms)], n_intervals)
    t = np.tile(np.arange(n_intervals), n_vms)

    # Azure VMs tend to have steadier workload (enterprise apps)
    hour = (t * 5 / 60) % 24   # 5-min intervals
    base = np.repeat(np.random.uniform(0.2, 0.7, n_vms), n_intervals)
    cpu_avg = base + 0.15 * np.sin(2 * np.pi * hour / 24) + \
              np.random.normal(0, 0.06, total)
    cpu_max = cpu_avg + np.random.uniform(0.05, 0.2, total)
    mem_avg = base * 0.8 + 0.1 + np.random.normal(0, 0.04, total)

    df = pd.DataFrame({
        'vm_id': vm_ids,
        'timestamp': np.tile(np.arange(0, n_intervals * 300, 300), n_vms),
        'cpu_avg': np.clip(cpu_avg * 100, 0, 100).round(2),
        'cpu_max': np.clip(cpu_max * 100, 0, 100).round(2),
        'mem_avg': np.clip(mem_avg * 100, 0, 100).round(2),
    })

    Path("data/raw/azure").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/azure/vm_cpu_readings.csv", index=False)
    print(f"✅ Synthetic Azure data created: {len(df):,} rows")


def create_synthetic_google():
    """
    Synthetic Google Cluster Traces v3 format.
    Columns: time, machine_id, cpu_usage, memory_usage, cpu_request, mem_request
    Different pattern: Google has heavy batch jobs mixed with latency-sensitive
    """
    import pandas as pd
    import numpy as np

    google_path = Path("data/raw/google/machine_events.csv")
    if google_path.exists():
        print("✅ Google data already exists")
        return

    print("Generating synthetic Google-format data (150K rows)...")
    np.random.seed(999)

    n_machines = 40
    n_intervals = 3750
    total = n_machines * n_intervals

    machine_ids = np.repeat([f"g_{i:04d}" for i in range(n_machines)],
                             n_intervals)
    t = np.tile(np.arange(n_intervals), n_machines)

    # Google has bursty batch jobs — more spiky than Alibaba/Azure
    base = np.repeat(np.random.uniform(0.1, 0.4, n_machines), n_intervals)
    batch_spike = (np.random.random(total) < 0.05) * \
                   np.random.uniform(0.3, 0.7, total)
    cpu = base + batch_spike + np.random.normal(0, 0.07, total)
    mem = base * 1.2 + np.random.normal(0, 0.05, total)

    df = pd.DataFrame({
        'time': np.tile(np.arange(n_intervals), n_machines),
        'machine_id': machine_ids,
        'cpu_usage': np.clip(cpu, 0, 1).round(4),
        'memory_usage': np.clip(mem, 0, 1).round(4),
        'cpu_request': np.clip(cpu + 0.1, 0, 1).round(4),
        'memory_request': np.clip(mem + 0.05, 0, 1).round(4),
    })

    Path("data/raw/google").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/google/machine_events.csv", index=False)
    print(f"✅ Synthetic Google data created: {len(df):,} rows")


def verify_data():
    import pandas as pd
    print("\n── Data Verification ──────────────────────────")
    for name, path in [
        ("Alibaba", "data/raw/alibaba/machine_usage.csv"),
        ("Azure",   "data/raw/azure/vm_cpu_readings.csv"),
        ("Google",  "data/raw/google/machine_events.csv"),
    ]:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p, nrows=5)
            size = p.stat().st_size / (1024*1024)
            print(f"✅ {name}: {path} ({size:.1f} MB)")
            print(f"   Columns: {list(df.columns)}")
        else:
            print(f"❌ {name}: NOT FOUND at {path}")
    print("──────────────────────────────────────────────\n")


if __name__ == "__main__":
    print("=" * 60)
    print("H9MLAI Project Setup — Sabhyata Kumari X24283142")
    print("=" * 60)

    install_packages()

    print("\n── Downloading Real Datasets ──────────────────")
    download_alibaba()
    download_azure()
    download_google()

    verify_data()
    assert_real_data()
    print("✅ Real-data check passed for all providers")

    run_preprocessing()

    print("✅ Setup complete! Next run:")
    print("   python src/models/train_all.py     ← train all models")
    print("   python src/explainability/shap_analysis.py")
    print("   streamlit run src/dashboard/app.py ← launch UI")
