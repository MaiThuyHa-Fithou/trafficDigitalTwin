"""
config.py – Cấu hình trung tâm toàn hệ thống
Ngã Tư Sở – Nguyễn Trãi, Hà Nội
Tương thích: Windows + Linux/macOS
"""
import os
import sys
import platform

# ── Đường dẫn ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CFG   = os.path.join(BASE_DIR, "sumo", "nga_tu_so.sumocfg")
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model", "attn_rnn.pt")
RESULT_DIR = os.path.join(BASE_DIR, "results")

for _d in [DATA_DIR, RESULT_DIR, os.path.join(BASE_DIR, "model")]:
    os.makedirs(_d, exist_ok=True)

# ── SUMO_HOME tự động theo OS ───────────────────────────────
def _detect_sumo_home():
    # 1. Ưu tiên biến môi trường (nếu người dùng đã set)
    env = os.environ.get("SUMO_HOME", "")
    if env and os.path.isdir(env):
        return env

    # 2. Theo hệ điều hành
    _os = platform.system()
    candidates = []
    if _os == "Windows":
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Sumo",
            r"C:\sumo",
        ]
    elif _os == "Darwin":  # macOS
        candidates = [
            "/opt/homebrew/share/sumo",
            "/usr/local/share/sumo",
            "/usr/share/sumo",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/sumo",
            "/usr/local/share/sumo",
            os.path.expanduser("~/sumo"),
        ]

    for p in candidates:
        if os.path.isdir(p):
            return p

    raise EnvironmentError(
        "Không tìm thấy SUMO!\n"
        "Hãy cài SUMO từ https://sumo.dlr.de/docs/Downloads.php\n"
        "Sau đó set biến môi trường SUMO_HOME, ví dụ:\n"
        r"  Windows: set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
        "\n  Linux:   export SUMO_HOME=/usr/share/sumo"
    )

SUMO_HOME = _detect_sumo_home()

# Thêm tools vào Python path ngay khi import config
_tools = os.path.join(SUMO_HOME, "tools")
if _tools not in sys.path:
    sys.path.insert(0, _tools)

# ── SUMO / TraCI ────────────────────────────────────────────
SUMO_CMD      = ["sumo", "-c", SUMO_CFG]
SIM_STEPS     = 3600
COLLECT_STEPS = 3600

EDGES = [
    "A0B0",
    "bottom1B0",
    "right0B0",
    "B1B0",
]
EDGE_NAMES = [
    "Nguyễn Trãi (Bắc→vào)",
    "Nguyễn Trãi (Nam→vào)",
    "Trường Chinh (Đông→vào)",
    "Tây Sơn (Tây→vào)",
]
EDGE_LEN_M      = 300.0
TL_ID           = "B0"
SAMPLE_INTERVAL = 60

# ── Mô hình RNN ─────────────────────────────────────────────
SEQ_LEN     = 10
FEATURES    = ["flow", "speed", "density"]
N_FEATURES  = len(FEATURES) * len(EDGES)   # 12
LSTM_HIDDEN = [128, 64, 32]
DROPOUT     = 0.2
PRED_HORIZON= 1
EPOCHS      = 60
BATCH_SIZE  = 16
LR          = 1e-3
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42

# ── API ─────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000

# ── Adaptive control ────────────────────────────────────────
FLOW_HIGH_THRESHOLD = 15.0
FLOW_LOW_THRESHOLD  = 5.0
TL_PHASE_NS = 0
TL_PHASE_EW = 2
