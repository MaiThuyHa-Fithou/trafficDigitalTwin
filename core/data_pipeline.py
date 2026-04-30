"""
data_pipeline.py – Tiền xử lý dữ liệu & tạo sequences cho RNN.

Pipeline:
  CSV  →  clean  →  MinMaxScale  →  sliding-window sequences  →  (X, y)
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

from core.config import (
    SEQ_LEN, PRED_HORIZON, FEATURES,
    EDGES, TRAIN_RATIO, VAL_RATIO, DATA_DIR,
)

SCALER_PATH = os.path.join(DATA_DIR, "scalers.pkl")


# ── Làm sạch ────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["flow", "speed", "density", "queue"]:
        if col in df.columns:
            # Nội suy tuyến tính cho khoảng thiếu ngắn
            df[col] = df[col].interpolate(method="linear", limit=3)
            # Trung bình giãn nở cho phần còn thiếu
            df[col] = df[col].fillna(df[col].expanding().mean())
            df[col] = df[col].fillna(0)
    return df


# ── Chuẩn hóa ────────────────────────────────────────────────
def fit_scalers(df: pd.DataFrame) -> dict:
    """Fit một MinMaxScaler cho mỗi cột đặc trưng, lưu vào file."""
    scalers = {}
    for feat in FEATURES:
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.fit(df[[feat]])
        scalers[feat] = sc
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scalers, f)
    return scalers


def load_scalers() -> dict:
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for feat in FEATURES:
        if feat in df.columns and feat in scalers:
            df[feat] = scalers[feat].transform(df[[feat]])
    return df


def inverse_flow(arr: np.ndarray, scalers: dict) -> np.ndarray:
    """Khôi phục giá trị flow về đơn vị xe/phút."""
    return scalers["flow"].inverse_transform(
        arr.reshape(-1, 1)).flatten()


# ── Tạo sequences ────────────────────────────────────────────
def build_sequences(df: pd.DataFrame,
                    seq_len: int = SEQ_LEN,
                    pred_horizon: int = PRED_HORIZON):
    """
    Tạo sliding-window sequences đa cạnh.

    Mỗi bước thời gian t: vector đặc trưng = [flow,speed,density] × n_edges
    X shape: (N, seq_len, n_features)
    y shape: (N,)   ← flow tổng hợp tại t + pred_horizon

    Parameters
    ----------
    df : DataFrame đã chuẩn hóa, có cột [time, edge_id, flow, speed, density]
    """
    # Pivot: rows=time, cols=edge×feature
    pivot = df.pivot_table(
        index="time", columns="edge_id",
        values=FEATURES, aggfunc="first"
    )
    # Đặt tên cột phẳng: flow_A0B0, speed_A0B0, ...
    pivot.columns = [f"{feat}_{eid}" for feat, eid in pivot.columns]
    pivot = pivot.sort_index().fillna(0)

    data = pivot.values.astype(np.float32)
    n_feat = data.shape[1]

    # Cột target: trung bình flow tất cả cạnh (chuẩn hóa)
    flow_cols = [i for i, c in enumerate(pivot.columns) if c.startswith("flow_")]

    X, y = [], []
    for i in range(seq_len, len(data) - pred_horizon + 1):
        X.append(data[i - seq_len: i])
        y.append(np.mean(data[i + pred_horizon - 1, flow_cols]))

    X = np.array(X, dtype=np.float32)   # (N, seq_len, n_feat)
    y = np.array(y, dtype=np.float32)   # (N,)
    return X, y, list(pivot.columns)


# ── Train / Val / Test split ─────────────────────────────────
def split(X: np.ndarray, y: np.ndarray):
    n  = len(X)
    n1 = max(1, int(n * TRAIN_RATIO))
    n2 = max(1, int(n * VAL_RATIO))
    return (
        X[:n1],      y[:n1],
        X[n1:n1+n2], y[n1:n1+n2],
        X[n1+n2:],   y[n1+n2:],
    )


# ── Full pipeline ────────────────────────────────────────────
def run_pipeline(csv_path: str = None):
    csv_path = csv_path or os.path.join(DATA_DIR, "sumo_traffic_data.csv")
    print("[Pipeline] Đọc dữ liệu...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} hàng, {df['time'].nunique()} bước thời gian")

    print("[Pipeline] Làm sạch...")
    df = clean(df)

    print("[Pipeline] Fit scalers & chuẩn hóa...")
    scalers = fit_scalers(df)
    df_norm = apply_scalers(df, scalers)

    print("[Pipeline] Tạo sequences...")
    X, y, col_names = build_sequences(df_norm)
    print(f"  X={X.shape}  y={y.shape}  n_features={X.shape[2]}")

    Xtr, ytr, Xva, yva, Xte, yte = split(X, y)
    print(f"  Train={len(Xtr)}  Val={len(Xva)}  Test={len(Xte)}")

    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val"  : Xva, "y_val"  : yva,
        "X_test" : Xte, "y_test" : yte,
        "scalers": scalers,
        "col_names": col_names,
        "df_norm": df_norm,
        "df_raw" : pd.read_csv(csv_path),
    }
