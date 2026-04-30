"""
data_loader.py – Nạp và kiểm tra file CSV dữ liệu SUMO.
"""
import os
import pandas as pd
from core.config import DATA_DIR


def load_data(path: str = None) -> pd.DataFrame:
    """
    Nạp file CSV dữ liệu giao thông.
    Tự động đổi tên cột để tương thích nếu dùng file ngoài.
    """
    path = path or os.path.join(DATA_DIR, "sumo_traffic_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            "Hãy chạy  python run_system.py --collect  trước."
        )

    df = pd.read_csv(path)

    # Tương thích tên cột cũ
    rename_map = {
        "vehicle_count": "flow",
        "mean_speed"   : "speed",
        "t"            : "time",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"time", "edge_id", "flow", "speed", "density"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thiếu cột: {missing}")

    return df


def describe(df: pd.DataFrame):
    """In thống kê tóm tắt."""
    print(f"  Hàng: {len(df):,}  |  Bước thời gian: {df['time'].nunique()}")
    print(f"  Cạnh: {df['edge_id'].unique().tolist()}")
    for feat in ["flow", "speed", "density"]:
        if feat in df.columns:
            print(f"  {feat:8s}: min={df[feat].min():.2f}  "
                  f"max={df[feat].max():.2f}  "
                  f"mean={df[feat].mean():.2f}")
