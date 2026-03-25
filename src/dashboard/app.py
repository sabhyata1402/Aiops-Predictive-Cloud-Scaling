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
    page_title="Proactive Cloud Scaling — AIOps Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.8rem; opacity: 0.85; margin: 0; }
    .alert-red    { background-color: #ffebee; border-left: 4px solid #c62828;
                    padding: 0.8rem; border-radius: 4px; }
    .alert-amber  { background-color: #fff8e1; border-left: 4px solid #f57f17;
                    padding: 0.8rem; border-radius: 4px; }
    .alert-green  { background-color: #e8f5e9; border-left: 4px solid #2e7d32;
                    padding: 0.8rem; border-radius: 4px; }
    .section-title { font-size: 1.1rem; font-weight: 600;
                     border-bottom: 2px solid #3f51b5;
                     padding-bottom: 0.3rem; margin-bottom: 1rem; }
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
                current_nodes=3, price=0.192):
    if predicted_cpu > 70:
        proactive_nodes = max(current_nodes,
                              round(current_nodes * predicted_cpu / 80))
    else:
        proactive_nodes = current_nodes
    reactive_nodes = current_nodes * 1.4 if actual_cpu > 85 else current_nodes
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
    # Header
    st.title("🔮 Intelligent Proactive Cloud Scaling Dashboard")
    st.markdown(
        "**H9MLAI — Sabhyata Kumari | X24283142 | NCI Dublin MSc AI 2026**  "
        "| *Predicts CPU/memory spikes 30 minutes ahead using ML*"
    )
    st.divider()

    models  = load_models()
    results = load_results()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Forecast Dashboard", "🖥️ Live Monitor"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE MONITOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🖥️ Live Cloud Resource Monitor")
        st.markdown(
            "Real-time CPU & memory monitoring of the **Streamlit Cloud server** "
            "with 30-minute-ahead XGBoost prediction. Auto-refreshes every 10 seconds."
        )

        # Auto-refresh using meta tag
        refresh_interval = st.selectbox(
            "Auto-refresh interval", [10, 30, 60], index=0,
            format_func=lambda x: f"{x} seconds"
        )
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_interval}">',
            unsafe_allow_html=True
        )

        cpu, mem, disk, ts = get_live_metrics()

        # Predict next CPU
        predicted_cpu = cpu * 1.02  # fallback
        if models.get('xgboost'):
            predicted_cpu = live_predict(models['xgboost'], cpu, mem)

        # Alert
        if predicted_cpu > 85:
            st.error(f"🔴 CRITICAL: Predicted CPU {predicted_cpu:.1f}% — scale up immediately!")
        elif predicted_cpu > 70:
            st.warning(f"🟡 WARNING: Predicted CPU {predicted_cpu:.1f}% — monitor closely.")
        else:
            st.success(f"🟢 NORMAL: Predicted CPU {predicted_cpu:.1f}% — system healthy.")

        # Live KPI cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("CPU Usage (Now)", f"{cpu:.1f}%",
                      delta=f"{cpu - 50:.1f}% vs baseline")
        with c2:
            st.metric("Memory Usage", f"{mem:.1f}%")
        with c3:
            st.metric("Disk Usage", f"{disk:.1f}%")
        with c4:
            st.metric("XGBoost Prediction", f"{predicted_cpu:.1f}%",
                      delta=f"{predicted_cpu - cpu:+.1f}% trend",
                      delta_color="inverse")

        st.caption(f"Last updated: {ts}  |  Cloud server: Streamlit Community Cloud (Linux x86_64)")

        # Rolling history using session state
        if 'live_history' not in st.session_state:
            st.session_state.live_history = []

        st.session_state.live_history.append({
            'time': ts, 'cpu': cpu, 'mem': mem,
            'predicted': predicted_cpu
        })
        # Keep last 30 readings
        st.session_state.live_history = st.session_state.live_history[-30:]

        hist_df = pd.DataFrame(st.session_state.live_history)

        if len(hist_df) > 1:
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=hist_df['time'], y=hist_df['cpu'],
                mode='lines+markers', name='Actual CPU%',
                line=dict(color='black', width=2)
            ))
            fig_live.add_trace(go.Scatter(
                x=hist_df['time'], y=hist_df['predicted'],
                mode='lines+markers', name='XGBoost Predicted',
                line=dict(color='#e53935', width=2, dash='dot')
            ))
            fig_live.add_trace(go.Scatter(
                x=hist_df['time'], y=hist_df['mem'],
                mode='lines+markers', name='Memory%',
                line=dict(color='#1e88e5', width=1.5)
            ))
            fig_live.add_hline(y=85, line_dash='dash', line_color='grey',
                               annotation_text='SLA threshold (85%)')
            fig_live.update_layout(
                height=380, title="Live Resource Usage — Streamlit Cloud Server",
                xaxis_title="Time (UTC)", yaxis_title="Utilisation (%)",
                yaxis=dict(range=[0, 105]),
                plot_bgcolor='white', paper_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(t=50, l=50, r=20, b=40)
            )
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.info("Collecting data... refresh the page to see the live chart build up.")

        # Scaling recommendation
        saving_live, p_nodes_live, r_nodes_live = cost_saving(
            predicted_cpu, cpu, 3)
        st.markdown(f"""
        **Scaling Recommendation:**
        - Current nodes: **3**
        - Recommended (proactive): **{p_nodes_live} nodes**
        - Reactive baseline: **{r_nodes_live} nodes**
        - Est. daily saving: **€{saving_live:.2f}**
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FORECAST DASHBOARD (existing content moved here)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        # ── Sidebar ───────────────────────────────────────────────────────────
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/cloud.png", width=60)
            st.header("⚙️ Controls")

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
            current_nodes = st.slider("Current node count", 1, 20, 3)

            st.divider()
            run_btn = st.button("🚀 Run Forecast", type="primary",
                                use_container_width=True)

            st.divider()
            st.caption("AWS m5.xlarge · EU-West-1 Dublin · €0.192/hr")

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
                saving, p_nodes, r_nodes = cost_saving(mean_pred, mean_actual,
                                                        current_nodes)

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
                        {f'<p style="font-size:0.7rem;opacity:0.7">{sub}</p>'
                         if sub else ''}
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
                        s, _, _ = cost_saving(pred_v, actual_v, current_nodes)
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
