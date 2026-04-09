# H9MLAI — Intelligent Proactive Cloud Scaling

> **Predicts CPU/memory spikes 30 minutes ahead using XGBoost, Random Forest,
> and LSTM — with SHAP explainability and cost saving analysis.**

---

## ⚡ Quick Start (Do this in order)

```bash
# 1. Clone / copy this folder to your laptop

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Run setup — installs packages, downloads REAL datasets, and preprocesses
python setup_and_download.py

# 4. Train all 3 models (takes 10–30 minutes)
python src/models/train_all.py

# 5. Generate SHAP analysis and figures
python src/explainability/shap_analysis.py

# 6. Launch dashboard
streamlit run src/dashboard/app.py
```

Your browser will open at http://localhost:8501

---

## 📁 Project Structure

```
h9mlai_project/
├── setup_and_download.py      ← START HERE
├── requirements.txt
├── README.md
├── src/
│   ├── data/
│   │   └── preprocessor.py    ← Data loading + feature engineering
│   ├── models/
│   │   └── train_all.py       ← XGBoost + RF + LSTM training
│   ├── explainability/
│   │   └── shap_analysis.py   ← SHAP values + all figures
│   └── dashboard/
│       └── app.py             ← Streamlit UI
└── data/
    ├── raw/                   ← Downloaded datasets
    ├── processed/             ← Feature matrices (parquet)
    ├── models/                ← Trained model files
    ├── results/               ← Evaluation results (pkl + json)
    └── figures/               ← All plots for report
```

---

## 📊 Datasets

| Dataset | Source | Size | Role |
|---------|--------|------|------|
| Alibaba 2018 Cluster Trace | github.com/alibaba/clusterdata | ~500K rows | Primary training |
| Azure Public Dataset | github.com/Azure/AzurePublicDataset | ~200K rows | Cross-cloud validation |
| Google Cluster Traces v3 | research.google/tools/datasets | ~150K rows | Cross-cloud test |

> If auto-download fails, synthetic data matching each format is created automatically.
> Real data download instructions:
> - Alibaba: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md
> This project is configured for real dataset download and processing on
> - Google: https://github.com/google/cluster-data

---

## 🤖 Models

| Model | Type | Library | Inference |
|-------|------|---------|-----------|
| XGBoost | Gradient boosting regression | xgboost 2.0 | < 5ms |
| Random Forest | Bagging ensemble regression | scikit-learn 1.4 | < 10ms |
| LSTM | Recurrent neural network | PyTorch 2.1 | < 50ms |

**Target:** Predict CPU utilisation (%) 30 minutes ahead

**SHAP:** Applied to all 3 models — TreeExplainer for XGBoost/RF,
gradient-based importance for LSTM

---

## 💰 Cost Analysis

Compares proactive ML scaling vs reactive threshold auto-scaling:
- **Reactive:** Waits until CPU > 85%, over-provisions by 40%, 2-min delay
- **Proactive (ML):** Scales when forecast shows CPU > 70%, right-sized
- **Reference:** AWS m5.xlarge @ €0.192/hr (EU-West-1 Dublin)

---

## 🚀 Deploying to Streamlit Cloud

1. Push this project to a GitHub repository
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `src/dashboard/app.py`
5. Click Deploy → get a public URL

**Note:** Commit the `data/processed/` and `data/models/` folders
(after training locally) so Streamlit Cloud has the models to load.

---

## 📋 Evaluation Results Location

After training:
- `data/results/all_results.json` — all metrics in JSON
- `data/results/learning_curves.pkl` — learning curve data
- `data/results/cross_cloud_results.pkl` — cross-cloud RMSE
- `data/figures/*.png` — all plots for report

---

## 🎥 Video Demo Script (7 minutes)

1. **0:00–1:00** — Problem statement: show threshold alert failing, 2-min delay
2. **1:00–2:00** — Datasets: show data.gov structure, explain 3 providers
3. **2:00–3:30** — Models: show training output, explain XGBoost vs LSTM
4. **3:30–5:00** — Dashboard: live demo of forecast + SHAP + cost saving
5. **5:00–6:00** — Results: show comparison table, learning curves, cross-cloud
6. **6:00–7:00** — Conclusions: answer RQ1/RQ2/RQ3, future work

---

## ⚠️ Troubleshooting

**"No module named torch"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**"Data not found"**
```bash
python setup_and_download.py   # re-run setup
```

**LSTM training too slow**
Edit `src/models/train_all.py` → change `MAX_EPOCHS = 60` to `MAX_EPOCHS = 20`

**Streamlit port in use**
```bash
streamlit run src/dashboard/app.py --server.port 8502
```

---

*Sabhyata Kumari | X24283142 | H9MLAI Machine Learning*
*MSc in Artificial Intelligence 2025/2026 | National College of Ireland*
