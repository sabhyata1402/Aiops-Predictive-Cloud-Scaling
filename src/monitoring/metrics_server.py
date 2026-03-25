"""
metrics_server.py — Lightweight HTTP metrics endpoint for GCP/cloud VMs.

Deploy this on your VM so the AIOps dashboard can poll live resource data.

Usage:
    pip install flask psutil
    python src/monitoring/metrics_server.py

Then in the Streamlit dashboard set GCP_VM_IP to your VM's external IP.
Endpoint: http://<VM_IP>:8080/metrics  →  JSON with cpu, memory, disk, time
"""

from flask import Flask, jsonify
import psutil
import datetime

app = Flask(__name__)


@app.route("/metrics")
def metrics():
    """Return current CPU, memory, disk utilisation as JSON."""
    return jsonify({
        "cpu":    round(psutil.cpu_percent(interval=1), 1),
        "memory": round(psutil.virtual_memory().percent, 1),
        "disk":   round(psutil.disk_usage("/").percent, 1),
        "time":   datetime.datetime.utcnow().strftime("%H:%M:%S UTC"),
        "source": "gcp-vm"
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("AIOps Metrics Server running on http://0.0.0.0:8080")
    print("Endpoints: /metrics  /health")
    app.run(host="0.0.0.0", port=8080)
