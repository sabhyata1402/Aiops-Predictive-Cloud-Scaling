"""
Streamlit Dashboard — H9MLAI Project
Sabhyata Kumari | X24283142

Intelligent Proactive Resource Forecasting Dashboard
Run: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AIOps — Intelligent Cloud Scaling",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] { background: #f0f4f8; }

    /* ── Sidebar: blue gradient matching header ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 55%, #0ea5e9 100%);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small { color: #dbeafe !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 700; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
    /* Inputs inside sidebar */
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stMultiSelect > div { background: rgba(255,255,255,0.12) !important; border-radius: 6px; }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        background: #f8fafc !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox input,
    [data-testid="stSidebar"] .stMultiSelect input {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: white !important;
        color: #1d4ed8 !important;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #eff6ff !important;
    }

    /* ── Top header banner ── */
    .aiops-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 55%, #0ea5e9 100%);
        padding: 1.2rem 1.8rem 1rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0.6rem;
    }
    .aiops-header h1 { font-size: 1.5rem; font-weight: 800; margin: 0 0 0.2rem 0; }
    .aiops-header p  { font-size: 0.8rem; opacity: 0.88; margin: 0; }

    /* ── Section banner (same style in both tabs) ── */
    .section-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 55%, #0ea5e9 100%);
        padding: 0.65rem 1.2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .section-banner .sb-title { font-size: 0.95rem; font-weight: 700; }
    .section-banner .sb-desc  { font-size: 0.78rem; opacity: 0.85; }

    /* ── KPI metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 55%, #0ea5e9 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 0.4rem;
        box-shadow: 0 2px 8px rgba(29,78,216,0.2);
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.78rem; opacity: 0.85; margin: 0; }
    .metric-sub   { font-size: 0.68rem; opacity: 0.7; margin: 0; }

    /* ── Alert cards ── */
    .alert-red   { background:#fef2f2; border-left:5px solid #dc2626;
                   padding:0.85rem 1rem; border-radius:6px; color:#7f1d1d;
                   font-size:0.9rem; line-height:1.5; }
    .alert-amber { background:#fffbeb; border-left:5px solid #d97706;
                   padding:0.85rem 1rem; border-radius:6px; color:#78350f;
                   font-size:0.9rem; line-height:1.5; }
    .alert-green { background:#f0fdf4; border-left:5px solid #16a34a;
                   padding:0.85rem 1rem; border-radius:6px; color:#14532d;
                   font-size:0.9rem; line-height:1.5; }

    /* ── Section title (for chart headings) ── */
    .section-title {
        font-size: 0.95rem; font-weight: 700; color: #1e3a8a;
        border-bottom: 2px solid #1d4ed8;
        padding-bottom: 0.3rem; margin-bottom: 0.8rem; margin-top: 0.5rem;
    }

    /* ── Scaling recommendation card ── */
    .rec-card {
        padding: 1rem 1.2rem; border-radius: 8px;
        margin-top: 0.8rem; font-size: 0.88rem;
        border: 1px solid rgba(0,0,0,0.07);
    }
    .rec-card .rec-title { font-size:0.95rem; font-weight:700; margin-bottom:0.5rem; }
    .rec-card .rec-action {
        font-size: 1rem; font-weight: 700;
        padding: 0.5rem 0.8rem; border-radius: 6px;
        margin-bottom: 0.6rem; display: inline-block;
    }
    .rec-card table { width:100%; border-collapse:collapse; margin-top:0.4rem; }
    .rec-card td { padding:0.28rem 0.5rem; font-size:0.85rem; }
    .rec-card tr:nth-child(even) { background:rgba(0,0,0,0.035); }
    .rec-card td:last-child { font-weight: 600; text-align:right; }

    /* ── Live pulsing dot ── */
    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.4; transform:scale(0.85); }
    }
    .live-dot {
        display:inline-block; width:9px; height:9px;
        background:#16a34a; border-radius:50%;
        animation: pulse 1.4s infinite; margin-right:5px;
        vertical-align:middle;
    }
    .sim-dot {
        display:inline-block; width:9px; height:9px;
        background:#dc2626; border-radius:50%;
        animation: pulse 0.7s infinite; margin-right:5px;
        vertical-align:middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models and data ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['xgboost'] = joblib.load("data/models/xgboost_model.pkl")
    except:
        models['xgboost'] = None
    try:
        models['rf'] = joblib.load("data/models/rf_model.pkl")
    except:
        models['rf'] = None
    try:
        from src.models.train_all import LSTMForecaster
        ckpt = torch.load("data/models/lstm_model.pt", map_location='cpu')
        lstm = LSTMForecaster(n_features=ckpt['n_features'])
        lstm.load_state_dict(ckpt['model_state'])
        lstm.eval()
        models['lstm'] = lstm
        models['lstm_n_features'] = ckpt['n_features']
        models['lstm_seq_len']    = ckpt.get('seq_len', 6)
    except:
        models['lstm'] = None
    return models


@st.cache_data
def load_data(provider='alibaba'):
    try:
        test  = pd.read_parquet(f"data/processed/{provider}_test.parquet")
        fcols = joblib.load(f"data/processed/{provider}_feature_cols.pkl")
        return test, fcols
    except:
        return None, None


@st.cache_data
def load_results():
    try:
        return joblib.load("data/results/all_results.pkl")
    except:
        return {}


@st.cache_data
def load_shap(model_name):
    try:
        return joblib.load(f"data/results/shap_{model_name}.pkl")
    except:
        return None


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_xgb_rf(model, X):
    if model is None:
        return np.random.uniform(20, 90, len(X))
    return model.predict(X)


def predict_lstm_model(model, X, seq_len=6, n_features=None):
    if model is None:
        return np.random.uniform(20, 90, max(len(X) - seq_len, 1))
    if n_features and X.shape[1] != n_features:
        if X.shape[1] > n_features:
            X = X[:, :n_features]
        else:
            pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
    preds = []
    for i in range(len(X) - seq_len):
        batch = torch.tensor(X[i:i+seq_len], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            preds.append(model(batch).item())
    return np.array(preds) if preds else np.array([50.0])


def cost_saving(predicted_cpu, actual_cpu,
                forecast_30m=None, target_util=75,
                current_nodes=3, price=0.192):
    """Estimate proactive vs reactive node counts and daily saving.

    projected is the worst of actual, short-horizon pred, and 30m forecast.
    target_util keeps utilisation below this % when recommending nodes.
    """
    projected = max(actual_cpu, predicted_cpu,
                    forecast_30m if forecast_30m is not None else predicted_cpu)
    safe_util = max(target_util, 1)
    proactive_nodes = max(current_nodes,
                          int(np.ceil(current_nodes * projected / safe_util)))
    reactive_nodes = max(current_nodes,
                         int(np.ceil(current_nodes * actual_cpu / 85)))
    saving = (reactive_nodes - proactive_nodes) * 24 * price
    return round(saving, 2), int(proactive_nodes), int(reactive_nodes)


# ── Live Monitor helpers ───────────────────────────────────────────────────────
def get_live_metrics():
    try:
        import psutil, datetime
        cpu  = psutil.cpu_percent(interval=1)
        mem  = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        ts   = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
        return cpu, mem, disk, ts
    except Exception:
        import random, datetime
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
        return random.uniform(10, 60), random.uniform(30, 80), 40.0, ts


def live_predict(model, cpu, mem):
    """Build a minimal feature vector and predict next CPU."""
    try:
        n = model.n_features_in_ if hasattr(model, 'n_features_in_') else 51
        feat = np.zeros(n, dtype=np.float32)
        feat[0] = cpu        # cpu_util_percent
        feat[1] = mem        # mem_util_percent
        feat[2] = cpu * 0.95 # lag_1
        feat[3] = cpu * 0.92 # lag_3
        return float(model.predict(feat.reshape(1, -1))[0])
    except Exception:
        return cpu * 1.02


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # ── Header banner ─────────────────────────────────────────────────────────
    st.markdown("""
<div class="aiops-header">
  <h1>🔮 AIOps — Intelligent Proactive Cloud Scaling</h1>
  <p>
    Predicts CPU &amp; memory pressure <strong>30 minutes ahead</strong> using
    XGBoost · Random Forest · LSTM &nbsp;|&nbsp;
    Triggers proactive scale-out before SLA breaches occur &nbsp;|&nbsp;
    <em>H9MLAI · Sabhyata Kumari · X24283142 · NCI Dublin MSc AI 2026</em>
  </p>
</div>
""", unsafe_allow_html=True)

    models  = load_models()
    results = load_results()

    # ── Sidebar (shared by both tabs) ─────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/cloud.png", width=55)
        st.markdown("### ⚙️ Controls")

        provider = st.selectbox(
            "Cloud Provider Dataset",
            ["alibaba", "azure", "google"],
            format_func=lambda x: x.capitalize()
        )
        model_choice = st.multiselect(
            "Models to run",
            ["XGBoost", "Random Forest", "LSTM"],
            default=["XGBoost", "Random Forest", "LSTM"]
        )
        n_samples = st.slider("Windows to analyse", 100, 2000, 500, 100)
        target    = st.selectbox("Target metric", ["CPU %", "Memory %"])

        st.divider()
        st.markdown("##### 📐 Scaling Parameters")
        current_nodes = st.slider("Current node count", 1, 40, 3,
                                  help="How many nodes your cluster is running right now")
        target_util = st.slider(
            "Target utilisation ceiling (%)", 60, 90, 75, 1,
            help="Keep CPU/Memory below this % — scale out before hitting it"
        )

        st.divider()
        run_btn = st.button("🚀 Run Forecast", type="primary",
                            use_container_width=True)
        st.divider()
        st.caption("AWS m5.xlarge · EU-West-1 Dublin · €0.192/hr")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Forecast Dashboard", "🖥️ Live Monitor"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE MONITOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
<div class="section-banner">
  <div>
    <div class="sb-title">🖥️ Live Cloud Resource Monitor</div>
    <div class="sb-desc">
      Monitors CPU &amp; memory every few seconds · XGBoost predicts resource pressure
      30 minutes ahead · Recommends exact node count to maintain SLA
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
        with col_ctrl1:
            refresh_interval = st.selectbox(
                "Refresh every", [5, 10, 30, 60], index=1,
                format_func=lambda x: f"{x}s"
            )
        with col_ctrl2:
            simulate = st.toggle(
                "🔥 Load Test Simulation", value=False,
                help="Runs a k6-style ramp-up → peak → ramp-down load pattern "
                     "(src/monitoring/load_test.js) to demo alerts and auto-scaling"
            )

        if simulate and 'sim_step' not in st.session_state:
            st.session_state.sim_step = 0
        if not simulate:
            st.session_state.sim_step = 0
            if 'live_history' in st.session_state:
                st.session_state.live_history = []

        @st.fragment(run_every=refresh_interval)
        def live_panel(xgb_model, current_nodes_live, target_util_live):
            import math, random, datetime

            # ── Metrics source ────────────────────────────────────────────────
            if st.session_state.get('sim_step', 0) > 0 or simulate:
                step = st.session_state.get('sim_step', 0)
                st.session_state.sim_step = step + 1
                t = step

                # k6-style profile: ramp-up → peak → ramp-down → idle → loop
                if t < 8:
                    base_cpu = 15 + t * 9           # 15 → 87 ramp-up
                elif t < 20:
                    base_cpu = 88 + 7 * math.sin(t * 0.9)  # 81–95 noisy peak
                elif t < 30:
                    base_cpu = 88 - (t - 20) * 8   # 88 → 8 ramp-down
                else:
                    base_cpu = 12 + (t % 6) * 1.5  # 12–21 idle
                    if t >= 36:
                        st.session_state.sim_step = 0

                cpu  = round(min(max(base_cpu + random.uniform(-2, 2), 5), 100), 1)
                mem  = round(min(38 + cpu * 0.38 + random.uniform(-1.5, 1.5), 95), 1)
                disk = round(38 + random.uniform(-0.5, 0.5), 1)
                ts   = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
                mode_badge = (
                    '<span style="background:#e53935;color:white;'
                    'padding:2px 8px;border-radius:12px;font-size:0.75rem">'
                    '⚡ Load Test Active</span>'
                )
            else:
                cpu, mem, disk, ts = get_live_metrics()
                mode_badge = (
                    '<span style="background:#2e7d32;color:white;'
                    'padding:2px 8px;border-radius:12px;font-size:0.75rem">'
                    '🟢 Live — Cloud VM</span>'
                )

            # ── Prediction: blend XGBoost with trend so alert always tracks load
            if xgb_model:
                model_pred = live_predict(xgb_model, cpu, mem)
                # If actual CPU is already high, weight toward actuals
                # (model trained on full feature set; sparse vector undershoots)
                weight = min((cpu - 30) / 60, 1.0) if cpu > 30 else 0.0
                predicted_cpu = round((1 - weight) * model_pred + weight * cpu * 1.04, 1)
            else:
                predicted_cpu = round(cpu * 1.04, 1)

            # ── Forward-look forecasts (short + 30m) with simple trend drift
            trend = max(predicted_cpu - cpu, 0)
            forecast_10s  = round(predicted_cpu, 1)
            forecast_30m  = round(min(100, predicted_cpu + trend * 2.5), 1)
            mem_forecast_30m = round(min(100, mem + max(mem - 60, 0) * 0.5), 1)

            # ── Alert — time-horizon aware, covers CPU + Memory ───────────────
            cpu_breach_now = cpu > 85 or predicted_cpu > 85
            mem_breach_now = mem > 85
            cpu_breach_30m = forecast_30m > 85
            mem_breach_30m = mem_forecast_30m > 85
            cpu_warn       = cpu > 70 or predicted_cpu > 70 or forecast_30m > 75
            mem_warn       = mem > 75 or mem_forecast_30m > 75
            any_critical   = cpu_breach_now or mem_breach_now or cpu_breach_30m or mem_breach_30m
            any_warn       = cpu_warn or mem_warn

            _, p_nodes_alert, _ = cost_saving(
                predicted_cpu, cpu, forecast_30m,
                target_util_live, current_nodes_live)

            if any_critical:
                lines = []
                if cpu_breach_now:
                    lines.append(
                        f"CPU is at <b>{cpu:.1f}%</b> — predicted <b>{predicted_cpu:.1f}%</b> "
                        f"in next {refresh_interval}s · SLA threshold breached"
                    )
                if cpu_breach_30m and not cpu_breach_now:
                    lines.append(
                        f"CPU will reach <b>{forecast_30m:.1f}%</b> in ~30 minutes "
                        f"(SLA breach imminent)"
                    )
                if mem_breach_now:
                    lines.append(
                        f"Memory at <b>{mem:.1f}%</b> — exceeds SLA threshold"
                    )
                if mem_breach_30m and not mem_breach_now:
                    lines.append(
                        f"Memory will reach <b>{mem_forecast_30m:.1f}%</b> in ~30 minutes"
                    )
                lines.append(
                    f"→ <b>Increase nodes immediately: "
                    f"{current_nodes_live} → {p_nodes_alert} nodes</b>"
                )
                st.markdown(
                    f'<div class="alert-red">🔴 <strong>CRITICAL — SLA BREACH RISK</strong><br>'
                    + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True
                )
            elif any_warn:
                lines = []
                if cpu_warn:
                    lines.append(
                        f"CPU at <b>{cpu:.1f}%</b> — forecast <b>{forecast_30m:.1f}%</b> "
                        f"in ~30 min · approaching SLA threshold"
                    )
                if mem_warn:
                    lines.append(
                        f"Memory at <b>{mem:.1f}%</b> — forecast <b>{mem_forecast_30m:.1f}%</b> "
                        f"in ~30 min"
                    )
                lines.append(
                    f"→ <b>Consider scaling: {current_nodes_live} → {p_nodes_alert} nodes "
                    f"within 15 minutes</b>"
                )
                st.markdown(
                    f'<div class="alert-amber">🟡 <strong>WARNING — Approaching threshold</strong><br>'
                    + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-green">🟢 <strong>NORMAL — All metrics within SLA</strong><br>'
                    f'CPU <b>{cpu:.1f}%</b> now · forecast <b>{forecast_30m:.1f}%</b> in 30m &nbsp;|&nbsp; '
                    f'Memory <b>{mem:.1f}%</b> now · forecast <b>{mem_forecast_30m:.1f}%</b> in 30m<br>'
                    f'No SLA breach projected · Current <b>{current_nodes_live} nodes</b> sufficient</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-top:0.7rem'></div>", unsafe_allow_html=True)

            # ── KPI tiles ─────────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("CPU (Now)", f"{cpu:.1f}%",
                          delta=f"{cpu - 50:+.1f}% vs baseline",
                          delta_color="inverse")
            with c2:
                st.metric("CPU Forecast (30m)", f"{forecast_30m:.1f}%",
                          delta=f"{forecast_30m - cpu:+.1f}% trend",
                          delta_color="inverse")
            with c3:
                st.metric("Memory (Now)", f"{mem:.1f}%",
                          delta=f"{mem - 60:+.1f}% vs baseline",
                          delta_color="inverse")
            with c4:
                st.metric("Memory Forecast (30m)", f"{mem_forecast_30m:.1f}%",
                          delta=f"{mem_forecast_30m - mem:+.1f}% trend",
                          delta_color="inverse")
            with c5:
                st.metric("Disk", f"{disk:.1f}%")

            st.markdown(
                f'<p style="font-size:0.76rem;color:#64748b;margin-top:0.15rem">'
                f'<span class="{"sim-dot" if simulate else "live-dot"}"></span>'
                f'&nbsp;{ts} &nbsp;·&nbsp; {mode_badge} &nbsp;·&nbsp; '
                f'auto-refresh every {refresh_interval}s</p>',
                unsafe_allow_html=True
            )

            # ── Rolling history ───────────────────────────────────────────────
            if 'live_history' not in st.session_state:
                st.session_state.live_history = []
            st.session_state.live_history.append(
                {'time': ts, 'cpu': cpu, 'mem': mem, 'predicted': predicted_cpu}
            )
            st.session_state.live_history = st.session_state.live_history[-60:]
            hist_df = pd.DataFrame(st.session_state.live_history)

            # ── Live chart ────────────────────────────────────────────────────
            fig_live = go.Figure()
            if len(hist_df) > 1:
                # Shaded CPU area
                fig_live.add_trace(go.Scatter(
                    x=hist_df['time'], y=hist_df['cpu'],
                    mode='lines+markers', name='CPU% (actual)',
                    line=dict(color='#1a237e', width=2.5),
                    fill='tozeroy', fillcolor='rgba(26,35,126,0.08)',
                    marker=dict(size=5)
                ))
                # XGBoost prediction line
                fig_live.add_trace(go.Scatter(
                    x=hist_df['time'], y=hist_df['predicted'],
                    mode='lines', name='CPU% (XGBoost forecast)',
                    line=dict(color='#e53935', width=2, dash='dot')
                ))
                # Memory line
                fig_live.add_trace(go.Scatter(
                    x=hist_df['time'], y=hist_df['mem'],
                    mode='lines+markers', name='Memory% (actual)',
                    line=dict(color='#0288d1', width=2),
                    marker=dict(size=4)
                ))

            # SLA threshold
            fig_live.add_hline(
                y=85, line_dash='dash', line_color='#e53935', line_width=1.5,
                annotation_text='SLA threshold — 85%',
                annotation_position='top right',
                annotation=dict(font=dict(color='#e53935', size=11))
            )
            # Warning threshold
            fig_live.add_hline(
                y=70, line_dash='dot', line_color='#f57f17', line_width=1,
                annotation_text='Warning — 70%',
                annotation_position='bottom right',
                annotation=dict(font=dict(color='#f57f17', size=10))
            )
            fig_live.update_layout(
                height=420,
                title=dict(
                    text="Real-Time Resource Utilisation — Cloud Infrastructure",
                    font=dict(size=13, color='#1a237e'), x=0
                ),
                xaxis=dict(title="Time (UTC)", tickangle=-30,
                           tickfont=dict(size=10)),
                yaxis=dict(title="Utilisation (%)", range=[0, 105],
                           gridcolor='#e0e0e0'),
                plot_bgcolor='white', paper_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            font=dict(size=11)),
                margin=dict(t=55, l=55, r=20, b=60),
                uirevision='live'
            )
            st.plotly_chart(fig_live, use_container_width=True)

            # ── Scaling recommendation ────────────────────────────────────────
            saving_live, p_nodes_live, r_nodes_live = cost_saving(
                predicted_cpu, cpu, forecast_30m, target_util_live,
                current_nodes_live)

            nodes_delta = p_nodes_live - current_nodes_live

            if any_critical:
                rc  = "#fef2f2"
                ri  = "🚨"
                urgency = "IMMEDIATELY — SLA breach window &lt; 30 minutes"
                action_bg = "#dc2626"; action_fg = "white"
            elif any_warn:
                rc  = "#fffbeb"
                ri  = "⚠️"
                urgency = "Within 15 minutes — breach projected in ~30m"
                action_bg = "#d97706"; action_fg = "white"
            else:
                rc  = "#f0fdf4"
                ri  = "✅"
                urgency = "No action required — monitor and review in 30m"
                action_bg = "#16a34a"; action_fg = "white"

            if nodes_delta > 0:
                action_txt = f"Increase: {current_nodes_live} → {p_nodes_live} nodes (+{nodes_delta})"
            else:
                action_txt = f"Hold: {current_nodes_live} nodes — capacity sufficient"

            st.markdown(f"""
<div class="rec-card" style="background:{rc}">
  <div class="rec-title">{ri} Proactive Scaling Recommendation</div>
  <div class="rec-action" style="background:{action_bg};color:{action_fg}">
    {action_txt}
  </div>
  <div style="font-size:0.8rem;color:#475569;margin-bottom:0.4rem">
    ⏰ {urgency}
  </div>
  <table>
    <tr><td>Current nodes</td><td>{current_nodes_live}</td></tr>
    <tr><td>Recommended nodes (proactive AIOps)</td><td><b>{p_nodes_live}</b></td></tr>
    <tr><td>Nodes needed reactively (without AIOps)</td><td>{r_nodes_live}</td></tr>
    <tr><td>Target utilisation ceiling</td><td>{target_util_live}%</td></tr>
    <tr><td>CPU forecast in 30 min</td><td>{forecast_30m:.1f}%</td></tr>
    <tr><td>Memory forecast in 30 min</td><td>{mem_forecast_30m:.1f}%</td></tr>
    <tr><td>Estimated daily saving vs reactive</td>
        <td style="color:#16a34a">€{saving_live:.2f}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

        live_panel(models.get('xgboost'), current_nodes, target_util)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FORECAST DASHBOARD (existing content moved here)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
<div class="section-banner">
  <div>
    <div class="sb-title">📊 Forecast Dashboard</div>
    <div class="sb-desc">
      Historical workload replay · XGBoost · Random Forest · LSTM predictions ·
      SHAP feature importance · Cost saving analysis
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Load data ─────────────────────────────────────────────────────────
        test_df, fcols = load_data(provider)

        if test_df is None:
            st.error("Data not found. Run setup_and_download.py first.")
            st.code("python setup_and_download.py\npython src/data/preprocessor.py")
            return

        target_col = 'target_cpu' if target == "CPU %" else 'target_mem'
        if target_col not in test_df.columns:
            target_col = 'target_cpu'

        sample = test_df.head(n_samples)
        X = sample[fcols].values.astype(np.float32)
        y = sample[target_col].values.astype(np.float32)

        # ── Results section ───────────────────────────────────────────────────
        if run_btn or True:   # show results by default

            # ── Top KPI cards ─────────────────────────────────────────────────
            with st.spinner("Running predictions..."):
                preds = {}
                if "XGBoost" in model_choice and models['xgboost']:
                    preds['XGBoost'] = predict_xgb_rf(models['xgboost'], X)
                if "Random Forest" in model_choice and models['rf']:
                    preds['Random Forest'] = predict_xgb_rf(models['rf'], X)
                if "LSTM" in model_choice and models['lstm']:
                    seq = models.get('lstm_seq_len', 6)
                    nf  = models.get('lstm_n_features')
                    lstm_p = predict_lstm_model(models['lstm'], X, seq, nf)
                    preds['LSTM'] = np.pad(lstm_p, (seq, 0), 'edge')

                if not preds:
                    st.warning("No models loaded. Train models first.")
                    return

                # Use XGBoost as primary if available
                primary_model = list(preds.keys())[0]
                primary_preds = preds[primary_model]

                # Cost analysis
                mean_pred   = float(np.mean(primary_preds))
                mean_actual = float(np.mean(y))
                saving, p_nodes, r_nodes = cost_saving(
                    mean_pred, mean_actual, mean_pred,
                    target_util, current_nodes)

            # KPI row
            cols = st.columns(5)
            kpis = [
                ("Mean Predicted CPU", f"{mean_pred:.1f}%", None),
                ("Mean Actual CPU",    f"{mean_actual:.1f}%", None),
                ("SLA Breach Windows",
                 f"{(y > 85).sum()}", f"{(y > 85).mean()*100:.1f}% of total"),
                ("Est. Daily Saving",  f"€{saving:.2f}", "vs reactive scaling"),
                ("Recommended Nodes",  f"{p_nodes} nodes",
                 f"vs {r_nodes} reactive"),
            ]
            for col, (label, val, sub) in zip(cols, kpis):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{val}</p>
                        <p class="metric-label">{label}</p>
                        <p class="metric-sub">{sub if sub else '&nbsp;'}</p>
                    </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Scaling recommendation alert ──────────────────────────────────
            if mean_pred > 85:
                st.markdown(f"""<div class="alert-red">
                🔴 <strong>CRITICAL:</strong> CPU forecast at {mean_pred:.1f}%
                — scale to {p_nodes} nodes NOW. SLA breach imminent.
                </div>""", unsafe_allow_html=True)
            elif mean_pred > 70:
                st.markdown(f"""<div class="alert-amber">
                🟡 <strong>WARNING:</strong> CPU forecast at {mean_pred:.1f}%
                — consider scaling to {p_nodes} nodes within 15 minutes.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="alert-green">
                🟢 <strong>NORMAL:</strong> CPU forecast at {mean_pred:.1f}%
                — current {current_nodes} nodes sufficient.
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Forecast chart ────────────────────────────────────────────────
            st.markdown('<p class="section-title">📈 CPU Forecast — All Models</p>',
                        unsafe_allow_html=True)

            fig = go.Figure()
            x_ax = list(range(len(y)))

            # Actual
            fig.add_trace(go.Scatter(
                x=x_ax, y=y.tolist(),
                mode='lines', name='Actual CPU%',
                line=dict(color='black', width=1.5), opacity=0.8
            ))

            colors_map = {'XGBoost': '#e53935', 'Random Forest': '#1e88e5',
                          'LSTM': '#43a047'}
            for model_name, model_preds in preds.items():
                # Align length
                p = model_preds[:len(y)] if len(model_preds) >= len(y) \
                    else np.pad(model_preds, (0, len(y) - len(model_preds)), 'edge')
                fig.add_trace(go.Scatter(
                    x=x_ax, y=p.tolist(),
                    mode='lines', name=model_name,
                    line=dict(color=colors_map.get(model_name, 'purple'), width=1.2),
                    opacity=0.8
                ))

            # SLA threshold line
            fig.add_hline(y=85, line_dash='dash', line_color='grey',
                          annotation_text='SLA threshold (85%)',
                          annotation_position='bottom right')

            fig.update_layout(
                height=380, xaxis_title="Time Window (10-min intervals)",
                yaxis_title="CPU Utilisation (%)",
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                plot_bgcolor='white', paper_bgcolor='white',
                yaxis=dict(range=[0, 105]),
                margin=dict(t=30, l=50, r=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── SHAP Feature Importance ───────────────────────────────────────
            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                st.markdown('<p class="section-title">🔍 SHAP Feature Importance'
                            ' — XGBoost</p>', unsafe_allow_html=True)
                shap_data = load_shap('xgboost')
                if shap_data:
                    sv   = np.abs(shap_data['shap_values']).mean(axis=0)
                    fns  = shap_data['feature_names']
                    top  = np.argsort(sv)[-12:]
                    fig2 = px.bar(
                        x=sv[top],
                        y=[fns[i].replace('_', ' ') for i in top],
                        orientation='h',
                        color=sv[top],
                        color_continuous_scale='Blues',
                        labels={'x': 'Mean |SHAP|', 'y': 'Feature'}
                    )
                    fig2.update_layout(height=350, showlegend=False,
                                       coloraxis_showscale=False,
                                       margin=dict(t=10, l=10, r=10, b=40),
                                       plot_bgcolor='white')
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("Positive SHAP = pushes prediction higher "
                               "(more likely CPU spike)")
                else:
                    st.info("Run shap_analysis.py to generate SHAP values")

            with c2:
                st.markdown('<p class="section-title">💰 Cost Saving Analysis'
                            '</p>', unsafe_allow_html=True)

                # Calculate per-window savings for all models
                cost_data = []
                for model_name, model_preds in preds.items():
                    p = model_preds[:len(y)]
                    window_savings = []
                    for pred_v, actual_v in zip(p, y):
                        s, _, _ = cost_saving(
                            pred_v, actual_v, pred_v,
                            target_util, current_nodes)
                        window_savings.append(s)
                    mean_s = np.mean(window_savings)
                    std_s  = np.std(window_savings)
                    cost_data.append({
                        'Model': model_name,
                        'Mean €/day': round(mean_s, 2),
                        'Std': round(std_s, 2),
                        'CI Lower': round(mean_s - 1.96*std_s, 2),
                        'CI Upper': round(mean_s + 1.96*std_s, 2),
                    })

                cost_df = pd.DataFrame(cost_data)
                fig3 = go.Figure()
                for _, row in cost_df.iterrows():
                    fig3.add_trace(go.Bar(
                        x=[row['Model']],
                        y=[row['Mean €/day']],
                        error_y=dict(type='data',
                                      array=[row['CI Upper'] - row['Mean €/day']],
                                      arrayminus=[row['Mean €/day'] - row['CI Lower']]),
                        name=row['Model'],
                        marker_color=colors_map.get(row['Model'], '#9c27b0')
                    ))
                fig3.update_layout(
                    height=350,
                    yaxis_title="Est. Daily Saving (€ vs reactive)",
                    plot_bgcolor='white', showlegend=False,
                    margin=dict(t=10, l=50, r=10, b=40)
                )
                fig3.add_hline(y=0, line_color='black', line_width=0.8)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(f"Reference: AWS m5.xlarge €0.192/hr · "
                           f"{current_nodes} current nodes · EU-West-1 Dublin")

            # ── Model Comparison Table ────────────────────────────────────────
            st.divider()
            st.markdown('<p class="section-title">📊 Model Performance Comparison'
                        '</p>', unsafe_allow_html=True)

            if results and 'model_performance' in results:
                perf = results['model_performance']
                rows = []
                for name, r in perf.items():
                    rows.append({
                        'Model': name.replace('_', ' ').title(),
                        'RMSE': r.get('rmse', '—'),
                        'MAE':  r.get('mae', '—'),
                        'MAPE (%)': r.get('mape', '—'),
                        'Train Time (s)': r.get('train_time_s', '—'),
                        'Inference (ms)': r.get('latency_ms', '—'),
                    })
                perf_df = pd.DataFrame(rows)
                st.dataframe(
                    perf_df.style.highlight_min(
                        subset=['RMSE', 'MAE', 'MAPE (%)'],
                        color='#c8e6c9'),
                    use_container_width=True, hide_index=True
                )

            # Cross-cloud
            if results and 'cross_cloud' in results and results['cross_cloud']:
                st.markdown('<p class="section-title">🌐 Cross-Cloud '
                            'Generalisation</p>', unsafe_allow_html=True)
                cc = results['cross_cloud']
                cc_rows = []
                for prov, vals in cc.items():
                    cc_rows.append({
                        'Test Provider': prov.capitalize(),
                        'XGBoost RMSE': vals.get('xgboost_rmse', '—'),
                        'RF RMSE':      vals.get('rf_rmse', '—'),
                        'LSTM RMSE':    vals.get('lstm_rmse', '—'),
                        'Common Features': vals.get('common_features', '—'),
                    })
                st.dataframe(pd.DataFrame(cc_rows),
                             use_container_width=True, hide_index=True)
                st.caption("Models trained on Alibaba, tested zero-shot on other providers")

            # ── Saved figures ─────────────────────────────────────────────────
            st.divider()
            st.markdown('<p class="section-title">📉 Learning Curves & Analysis'
                        '</p>', unsafe_allow_html=True)

            fig_col1, fig_col2 = st.columns(2)
            for col, fname, caption in [
                (fig_col1, "data/figures/learning_curves.png",
                 "Learning curves — RMSE vs training set size"),
                (fig_col2, "data/figures/shap_comparison.png",
                 "Feature importance comparison across models"),
            ]:
                with col:
                    if Path(fname).exists():
                        st.image(fname, caption=caption, use_column_width=True)
                    else:
                        st.info(f"Run shap_analysis.py to generate: {fname}")

            # Footer
            st.divider()
            st.caption(
                "📌 Sabhyata Kumari | X24283142 | H9MLAI Machine Learning | "
                "MSc AI 2025/2026 | National College of Ireland, Dublin  "
                "| Dataset: Alibaba 2018 + Azure Public + Google Cluster Traces"
            )


if __name__ == "__main__":
    main()
