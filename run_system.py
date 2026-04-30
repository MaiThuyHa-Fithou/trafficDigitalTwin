#!/usr/bin/env python3
"""
run_system.py – Điểm khởi động thống nhất của toàn hệ thống.

Sử dụng:
  python run_system.py --collect      # Bước 1: Thu thập dữ liệu SUMO
  python run_system.py --train        # Bước 2: Huấn luyện mô hình
  python run_system.py --api          # Bước 3: Chạy API server
  python run_system.py --dashboard    # Bước 4: Chạy Streamlit dashboard
  python run_system.py --all          # Bước 1+2+3+4 (tuần tự, trừ dashboard)
  python run_system.py --check        # Kiểm tra môi trường
"""
import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_env():
    print("=" * 60)
    print("  KIỂM TRA MÔI TRƯỜNG")
    print("=" * 60)
    ok = True

    # Python
    import platform
    print(f"  Python   : {platform.python_version()}")

    # SUMO
    import shutil
    sumo_bin = shutil.which("sumo")
    if sumo_bin:
        print(f"  SUMO     : ✅  {sumo_bin}")
    else:
        print("  SUMO     : ❌  Chưa cài đặt")
        ok = False

    # TraCI
    try:
        sys.path.append("/usr/share/sumo/tools")
        import traci
        print(f"  TraCI    : ✅")
    except ImportError:
        print("  TraCI    : ❌  Không tìm thấy")
        ok = False

    # PyTorch
    try:
        import torch
        print(f"  PyTorch  : ✅  {torch.__version__}  "
              f"(CUDA={'✅' if torch.cuda.is_available() else '❌'})")
    except ImportError:
        print("  PyTorch  : ❌  pip install torch")
        ok = False

    # Flask
    try:
        import flask
        print(f"  Flask    : ✅  {flask.__version__}")
    except ImportError:
        print("  Flask    : ❌  pip install flask")
        ok = False

    # Streamlit
    try:
        import streamlit
        print(f"  Streamlit: ✅  {streamlit.__version__}")
    except ImportError:
        print("  Streamlit: ❌  pip install streamlit")
        ok = False

    # sklearn
    try:
        import sklearn
        print(f"  Sklearn  : ✅  {sklearn.__version__}")
    except ImportError:
        print("  Sklearn  : ❌  pip install scikit-learn")
        ok = False

    # Files
    from core.config import SUMO_CFG
    net_file = os.path.join(BASE_DIR, "sumo", "nga_tu_so.net.xml")
    rou_file = os.path.join(BASE_DIR, "sumo", "nga_tu_so.rou.xml")
    print(f"\n  Mạng SUMO: {'✅' if os.path.exists(net_file) else '❌'}  {net_file}")
    print(f"  Route file: {'✅' if os.path.exists(rou_file) else '❌'}  {rou_file}")
    print(f"  SUMO cfg  : {'✅' if os.path.exists(SUMO_CFG) else '❌'}  {SUMO_CFG}")

    csv = os.path.join(BASE_DIR, "data", "sumo_traffic_data.csv")
    print(f"  Dữ liệu  : {'✅' if os.path.exists(csv) else '⚠️  Chưa có – chạy --collect'}")

    from core.config import MODEL_PATH
    print(f"  Mô hình  : {'✅' if os.path.exists(MODEL_PATH) else '⚠️  Chưa có – chạy --train'}")

    print("=" * 60)
    print(f"  Kết quả: {'✅ Sẵn sàng!' if ok else '❌ Có vấn đề – xem bên trên'}")
    print("=" * 60)
    return ok


def run_collect():
    print("\n▶  BƯỚC 1: THU THẬP DỮ LIỆU SUMO")
    from core.collect_data import collect_data
    collect_data()


def run_train():
    print("\n▶  BƯỚC 2: HUẤN LUYỆN MÔ HÌNH")
    # Thêm thư mục model vào path
    sys.path.insert(0, os.path.join(BASE_DIR, "model"))
    from model.train import main as train_main
    train_main()


def run_api():
    print("\n▶  BƯỚC 3: KHỞI ĐỘNG API SERVER")
    print("  API: http://127.0.0.1:8000")
    print("  Nhấn Ctrl+C để dừng\n")
    api_path = os.path.join(BASE_DIR, "app", "api_server.py")
    subprocess.run([sys.executable, api_path])


def run_dashboard():
    print("\n▶  BƯỚC 4: KHỞI ĐỘNG DASHBOARD")
    print("  Dashboard: http://localhost:8501")
    print("  Nhấn Ctrl+C để dừng\n")
    dash_path = os.path.join(BASE_DIR, "app", "realtime_dashboard.py")
    subprocess.run(["streamlit", "run", dash_path,
                    "--server.port", "8501",
                    "--server.headless", "true"])


def run_all():
    """Chạy tuần tự: collect → train (API và dashboard cần terminal riêng)."""
    run_collect()
    run_train()
    print("\n✅ Dữ liệu và mô hình đã sẵn sàng!")
    print("\n  Để hoàn tất hệ thống, mở 2 terminal và chạy:")
    print("    Terminal 1: python run_system.py --api")
    print("    Terminal 2: python run_system.py --dashboard")


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hệ thống Digital Twin giao thông – Ngã Tư Sở, Hà Nội"
    )
    parser.add_argument("--check",     action="store_true", help="Kiểm tra môi trường")
    parser.add_argument("--collect",   action="store_true", help="Thu thập dữ liệu SUMO")
    parser.add_argument("--train",     action="store_true", help="Huấn luyện mô hình")
    parser.add_argument("--api",       action="store_true", help="Chạy API server")
    parser.add_argument("--dashboard", action="store_true", help="Chạy Streamlit dashboard")
    parser.add_argument("--all",       action="store_true", help="Collect + Train")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.check:
        check_env()
    if args.collect:
        run_collect()
    if args.train:
        run_train()
    if args.all:
        run_all()
    if args.api:
        run_api()
    if args.dashboard:
        run_dashboard()
