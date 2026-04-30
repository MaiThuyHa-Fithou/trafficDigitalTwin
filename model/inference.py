"""
inference.py – Dự báo thời gian thực với sliding-window buffer.
Tích hợp vòng lặp Digital Twin: buffer → predict → feedback.
"""
import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.attn_rnn import AttnRNN
from core.config import (
    MODEL_PATH, SEQ_LEN, N_FEATURES, LSTM_HIDDEN, DROPOUT, DATA_DIR
)
from core.data_pipeline import load_scalers, SCALER_PATH

# ── Singleton state ──────────────────────────────────────────
_model   = None
_scalers = None
_buffer  = deque(maxlen=SEQ_LEN)   # buffer sliding-window


def load():
    """Nạp model và scalers vào bộ nhớ."""
    global _model, _scalers

    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model chưa được huấn luyện – sẽ dùng fallback.")
        return

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    n_feat = ckpt.get("n_features", N_FEATURES)
    hidden = ckpt.get("lstm_hidden", LSTM_HIDDEN)
    drop   = ckpt.get("dropout",     DROPOUT)

    _model = AttnRNN(n_feat, hidden, drop)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    if os.path.exists(SCALER_PATH):
        _scalers = load_scalers()

    m = ckpt.get("metrics", {})
    print(f"✅ Model nạp thành công  "
          f"MAE={m.get('MAE','?')}  MAPE={m.get('MAPE','?')}%")


def push_state(state_dict: dict) -> None:
    """
    Thêm một snapshot trạng thái giao thông vào buffer.
    state_dict: {edge_id: {flow, speed, density, ...}}
    """
    from core.config import EDGES, FEATURES
    vec = []
    for eid in EDGES:
        s = state_dict.get(eid, {})
        for feat in FEATURES:
            vec.append(float(s.get(feat, 0.0)))
    _buffer.append(vec)


def predict(state=None) -> float:
    """
    Dự báo lưu lượng (xe/phút) tại bước kế tiếp.

    Parameters
    ----------
    state : (optional) mảng numpy (n_edges, 3) từ api_server cũ.
            Nếu None → dùng buffer nội bộ.

    Returns
    -------
    float : lưu lượng dự báo (xe/phút)
    """
    # ── Fallback: dùng trung bình flow khi buffer chưa đủ ───
    if len(_buffer) < SEQ_LEN:
        if state is not None:
            return float(np.mean(state[:, 0]))
        return 0.0

    # ── Dự báo bằng Attn-RNN ────────────────────────────────
    if _model is None:
        # Không có model → trung bình n bước gần nhất
        arr = np.array(_buffer)
        return float(np.mean(arr[:, 0]))

    seq = torch.FloatTensor(np.array(_buffer)).unsqueeze(0)  # (1, T, F)
    with torch.no_grad():
        pred_norm, _ = _model(seq)
    val = float(pred_norm.item())

    # Inverse transform nếu có scaler
    if _scalers and "flow" in _scalers:
        val = float(_scalers["flow"].inverse_transform([[val]])[0][0])

    return max(0.0, val)


def predict_from_state(state_dict: dict) -> float:
    """Tiện ích: push state rồi predict ngay."""
    push_state(state_dict)
    return predict()
