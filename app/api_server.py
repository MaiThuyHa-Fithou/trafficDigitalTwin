"""
api_server.py – Flask REST API tích hợp Digital Twin loop.

Endpoints:
  GET  /status          → thông tin server
  GET  /data            → một bước TraCI + dự báo
  GET  /step?n=<int>    → chạy n bước, trả về lịch sử
  POST /reset           → khởi động lại mô phỏng
  GET  /metrics         → MAE/RMSE/MAPE tích lũy
"""
import os
import sys
import threading
import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EDGES, EDGE_NAMES, SUMO_CMD, TL_ID, SAMPLE_INTERVAL
from core.traci_env import SumoEnv
from model.inference import load, predict, push_state
from app.controller import control

# ── Init ─────────────────────────────────────────────────────
app   = Flask(__name__)
env   = SumoEnv(SUMO_CMD)
lock  = threading.Lock()

history   = []      # list of {step, flow, speed, density, prediction}
actuals   = []
preds     = []
_started  = False

def _ensure_started():
    global _started
    if not _started:
        env.start()
        _started = True

# ── Load model khi khởi động ─────────────────────────────────
load()


# ── Routes ───────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "server"  : "Traffic Digital Twin API",
        "location": "Ngã Tư Sở – Nguyễn Trãi, Hà Nội",
        "sim_step": env.sim_step,
        "running" : _started,
        "history_len": len(history),
    })


@app.route("/data")
def get_data():
    """Một bước mô phỏng + dự báo + điều khiển."""
    with lock:
        _ensure_started()

        # Chạy SAMPLE_INTERVAL bước (mỗi lần gọi = 1 chu kỳ lấy mẫu)
        state = None
        for _ in range(SAMPLE_INTERVAL):
            state = env.step()

        # Tổng hợp đặc trưng
        flows    = [state[e]["flow"]    for e in EDGES]
        speeds   = [state[e]["speed"]   for e in EDGES]
        densities= [state[e]["density"] for e in EDGES]

        # Cập nhật buffer dự báo
        push_state(state)
        pred_val = predict()

        # Điều khiển đèn
        ctrl_info = control(pred_val)

        # Ghi lịch sử
        record = {
            "step"      : env.sim_step,
            "flow"      : round(float(np.sum(flows)), 2),
            "speed"     : round(float(np.mean(speeds)), 2),
            "density"   : round(float(np.mean(densities)), 2),
            "prediction": round(pred_val, 2),
            "control"   : ctrl_info,
            "edges"     : {EDGES[i]: {
                "flow"   : round(flows[i], 2),
                "speed"  : round(speeds[i], 2),
                "density": round(densities[i], 2),
                "queue"  : state[EDGES[i]]["queue"],
                "name"   : EDGE_NAMES[i],
            } for i in range(len(EDGES))},
        }
        history.append(record)
        actuals.append(record["flow"])
        preds.append(pred_val)

    return jsonify(record)


@app.route("/step")
def step_n():
    """Chạy n chu kỳ lấy mẫu, trả về toàn bộ lịch sử."""
    n = int(request.args.get("n", 5))
    results = []
    for _ in range(n):
        with lock:
            _ensure_started()
            state = None
            for _ in range(SAMPLE_INTERVAL):
                state = env.step()
            push_state(state)
            pred_val = predict()
            control(pred_val)
            results.append({
                "step"      : env.sim_step,
                "flow"      : round(float(np.sum([state[e]["flow"] for e in EDGES])), 2),
                "prediction": round(pred_val, 2),
            })
    return jsonify({"steps": results})


@app.route("/metrics")
def get_metrics():
    """MAE, MAPE tích lũy từ đầu phiên."""
    if len(actuals) < 2:
        return jsonify({"error": "Chưa đủ dữ liệu"})
    a = np.array(actuals)
    p = np.array(preds)
    mae  = float(np.mean(np.abs(a - p)))
    mask = a > 0
    mape = float(np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100) if mask.any() else 0
    return jsonify({
        "n_steps": len(actuals),
        "MAE"    : round(mae, 4),
        "MAPE"   : round(mape, 2),
    })


@app.route("/reset", methods=["POST"])
def reset():
    global _started, history, actuals, preds
    with lock:
        env.close()
        _started = False
        history, actuals, preds = [], [], []
    return jsonify({"status": "reset ok"})


@app.route("/history")
def get_history():
    return jsonify({"history": history[-50:]})  # trả về 50 bước gần nhất


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    from core.config import API_HOST, API_PORT
    print(f"🚀 API đang chạy tại http://{API_HOST}:{API_PORT}")
    print(f"   Endpoints: /status  /data  /step?n=5  /metrics  /history")
    app.run(host=API_HOST, port=API_PORT, debug=False, threaded=True)
