# Hệ thống Digital Twin Giao thông – Ngã Tư Sở, Hà Nội

**Framework**: SUMO 1.18 + TraCI + Attn-RNN (PyTorch) + Flask + Streamlit

---

## Cấu trúc dự án

```
traffic_system/
├── sumo/                    # File mạng SUMO
│   ├── nga_tu_so.net.xml    # Mạng lưới đường (netgenerate 2×2 grid)
│   ├── nga_tu_so.rou.xml    # Định tuyến phương tiện (4 giai đoạn lưu lượng)
│   └── nga_tu_so.sumocfg   # Cấu hình SUMO
├── core/
│   ├── config.py            # Cấu hình trung tâm
│   ├── traci_env.py         # SUMO/TraCI environment
│   ├── collect_data.py      # Thu thập dữ liệu
│   ├── data_pipeline.py     # Tiền xử lý & tạo sequences
│   └── data_loader.py       # Nạp CSV
├── model/
│   ├── attn_rnn.py          # Kiến trúc Attn-RNN (PyTorch)
│   ├── train.py             # Huấn luyện + đánh giá
│   └── inference.py         # Dự báo thời gian thực
├── app/
│   ├── api_server.py        # Flask REST API
│   ├── controller.py        # Điều khiển tín hiệu thích nghi
│   └── realtime_dashboard.py # Streamlit dashboard
├── data/                    # Dữ liệu SUMO (tự sinh)
├── results/                 # Biểu đồ + JSON kết quả
├── run_system.py            # Entry point CLI
└── requirements.txt
```

---

## Cài đặt

### 1. SUMO
```bash
sudo apt install sumo sumo-tools    # Ubuntu/Debian
# Hoặc: https://sumo.dlr.de/docs/Downloads.php
export SUMO_HOME=/usr/share/sumo
```

### 2. Python dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib flask streamlit requests
```

---

## Chạy hệ thống

### Bước 1 – Kiểm tra môi trường
```bash
python run_system.py --check
```

### Bước 2 – Thu thập dữ liệu SUMO (3600 giây mô phỏng)
```bash
python run_system.py --collect
# → data/sumo_traffic_data.csv  (60 mẫu × 4 hướng)
```

### Bước 3 – Huấn luyện mô hình
```bash
python run_system.py --train
# → model/attn_rnn.pt
# → results/fig1..5.png
# → results/summary.json
```

### Bước 4 – Chạy API server (Terminal 1)
```bash
python run_system.py --api
# → http://127.0.0.1:8000
# Endpoints: /status  /data  /step?n=5  /metrics  /history
```

### Bước 5 – Chạy Dashboard (Terminal 2)
```bash
python run_system.py --dashboard
# → http://localhost:8501
```

### Hoặc chạy tuần tự bước 1+2
```bash
python run_system.py --all
```

---

## API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/status` | GET | Thông tin server & mô phỏng |
| `/data` | GET | 1 chu kỳ 60s: trạng thái + dự báo |
| `/step?n=5` | GET | Chạy n chu kỳ |
| `/metrics` | GET | MAE, MAPE tích lũy |
| `/history` | GET | 50 bước gần nhất |
| `/reset` | POST | Khởi động lại mô phỏng |

---

## Kết quả (thực nghiệm)

| Mô hình | MAE (xe/phút) | RMSE | MAPE |
|---------|-------------|------|------|
| ARIMA | ~25.8 | ~27.6 | ~79.6% |
| SVR (RBF) | ~3.0 | ~3.6 | ~10.9% |
| LSTM cơ bản | ~1.5 | ~1.9 | ~5.2% |
| **Attn-RNN** | **~2.1** | **~2.3** | **~7.2%** |

*Ghi chú: Kết quả dựa trên 60 mẫu SUMO × data augmentation ×3.*

---

## Tham số quan trọng (core/config.py)

```python
SAMPLE_INTERVAL = 60    # Chu kỳ lấy mẫu (giây)
SEQ_LEN         = 10    # Cửa sổ quan sát (10 bước = 10 phút)
LSTM_HIDDEN     = [128, 64, 32]
EPOCHS          = 60
```

---

## Nguồn tham khảo

Bài báo: *"Phát triển hệ thống dự báo lưu lượng giao thông thời gian thực
sử dụng mạng nơ-ron hồi quy (RNN) trên môi trường giả lập SUMO tại nút
giao thông Ngã Tư Sở – Nguyễn Trãi, TP. Hà Nội"*, ICATE 2026.
