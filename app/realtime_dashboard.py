"""
realtime_dashboard.py – Streamlit Dashboard thời gian thực.

Hiển thị:
  - Lưu lượng thực tế vs dự báo (line chart)
  - Metrics KPI: Flow, Speed, Density, Prediction
  - Trạng thái đèn tín hiệu
  - Biểu đồ theo từng hướng đường

Chạy: streamlit run app/realtime_dashboard.py
"""
import time
import requests
import streamlit as st
import pandas as pd

# ── Cấu hình trang ──────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Digital Twin – Ngã Tư Sở",
    page_icon="🚦",
    layout="wide",
)

API_BASE = "http://127.0.0.1:8000"
REFRESH  = 1.0          # giây giữa các lần làm mới
MAX_HIST = 60           # số điểm hiển thị trên biểu đồ

# ── Header ──────────────────────────────────────────────────
st.title("🚦 AI Traffic Digital Twin Dashboard")
st.caption("Ngã Tư Sở – Nguyễn Trãi, Hà Nội  |  SUMO + TraCI + Attn-RNN")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    refresh_rate = st.slider("Tần suất làm mới (s)", 0.5, 5.0, REFRESH, 0.5)
    max_steps    = st.number_input("Số bước tối đa", 10, 500, 100)
    auto_run     = st.checkbox("Tự động chạy", value=True)
    st.divider()
    if st.button("🔄 Reset mô phỏng"):
        try:
            requests.post(f"{API_BASE}/reset", timeout=3)
            st.success("Đã reset!")
        except Exception:
            st.error("Không kết nối được API")

    st.divider()
    st.markdown("**Thành phần hệ thống:**")
    st.markdown("- 🟢 SUMO 1.18.0")
    st.markdown("- 🟢 TraCI")
    st.markdown("- 🟢 Attn-RNN (PyTorch)")
    st.markdown("- 🟢 Flask API :8000")

# ── Kiểm tra kết nối API ─────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=2)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}

ok, status_info = check_api()

if not ok:
    st.error("⚠️ Không kết nối được API server. Hãy chạy:")
    st.code("python app/api_server.py", language="bash")
    st.stop()

# ── Vùng hiển thị ────────────────────────────────────────────
col_kpi = st.columns(5)
ph_flow   = col_kpi[0].empty()
ph_pred   = col_kpi[1].empty()
ph_speed  = col_kpi[2].empty()
ph_density= col_kpi[3].empty()
ph_ctrl   = col_kpi[4].empty()

st.divider()
col_chart1, col_chart2 = st.columns([2, 1])
ph_main_chart = col_chart1.empty()
ph_edge_table = col_chart2.empty()

ph_metrics = st.empty()
ph_status  = st.empty()

# ── Lịch sử dữ liệu ─────────────────────────────────────────
hist_flow   = []
hist_pred   = []
hist_steps  = []

# ── Vòng lặp chính ──────────────────────────────────────────
step_count = 0
while auto_run and step_count < max_steps:
    try:
        resp = requests.get(f"{API_BASE}/data", timeout=10)
        if resp.status_code != 200:
            ph_status.warning("API trả lỗi – thử lại...")
            time.sleep(refresh_rate)
            continue

        data = resp.json()
        step_count += 1

        flow   = data.get("flow", 0)
        pred   = data.get("prediction", 0)
        speed  = data.get("speed", 0)
        dens   = data.get("density", 0)
        ctrl   = data.get("control", {})
        edges  = data.get("edges", {})
        sim_step = data.get("step", 0)

        # Cập nhật lịch sử
        hist_flow.append(flow)
        hist_pred.append(pred)
        hist_steps.append(sim_step)
        if len(hist_flow) > MAX_HIST:
            hist_flow.pop(0); hist_pred.pop(0); hist_steps.pop(0)

        # ── KPIs ────────────────────────────────────────────
        delta_flow = f"{flow - pred:+.1f}" if len(hist_pred) > 1 else "–"
        ph_flow.metric("🚗 Flow (xe/phút)", f"{flow:.1f}", delta=f"Dự báo: {pred:.1f}")
        ph_pred.metric("🔮 Dự báo", f"{pred:.1f} xe/min",
                       delta=f"Δ {float(flow-pred):+.1f}")
        ph_speed.metric("💨 Tốc độ TB", f"{speed:.1f} km/h")
        ph_density.metric("📊 Mật độ", f"{dens:.2f} xe/km")

        # Trạng thái đèn
        phase    = ctrl.get("phase", "?")
        action   = ctrl.get("action", "–")
        duration = ctrl.get("duration", "–")
        phase_label = "🟢 N-S" if phase == 0 else ("🟡 Chuyển" if phase == 1 else "🔵 Đ-T")
        ph_ctrl.metric("🚦 Đèn tín hiệu", phase_label,
                       delta=f"t={duration}s | {action}")

        # ── Biểu đồ chính ───────────────────────────────────
        df_chart = pd.DataFrame({
            "Thực tế (SUMO)"    : hist_flow,
            "Dự báo (Attn-RNN)" : hist_pred,
        }, index=hist_steps[-len(hist_flow):])

        with ph_main_chart:
            st.subheader("📈 Lưu lượng thời gian thực vs Dự báo")
            st.line_chart(df_chart, height=300)

        # ── Bảng theo hướng ─────────────────────────────────
        edge_rows = []
        for eid, edata in edges.items():
            edge_rows.append({
                "Hướng"     : edata.get("name", eid),
                "Flow"      : f"{edata['flow']:.1f}",
                "Speed(km/h)": f"{edata['speed']:.1f}",
                "Mật độ"    : f"{edata['density']:.2f}",
                "Hàng chờ"  : edata.get("queue", 0),
            })
        if edge_rows:
            with ph_edge_table:
                st.subheader("📍 Theo hướng đường")
                st.dataframe(pd.DataFrame(edge_rows), hide_index=True,
                             use_container_width=True)

        # ── Metrics tích lũy ─────────────────────────────────
        try:
            m_resp = requests.get(f"{API_BASE}/metrics", timeout=2).json()
            with ph_metrics:
                mc = st.columns(3)
                mc[0].metric("MAE (tích lũy)",  f"{m_resp.get('MAE',0):.3f}")
                mc[1].metric("MAPE (tích lũy)", f"{m_resp.get('MAPE',0):.1f}%")
                mc[2].metric("Bước đã chạy",    m_resp.get("n_steps", 0))
        except Exception:
            pass

        ph_status.caption(
            f"🕐 Bước mô phỏng: {sim_step}s  |  Chu kỳ giao diện: #{step_count}"
        )

    except requests.exceptions.Timeout:
        ph_status.warning("⏳ API timeout – đang chờ SUMO...")
    except Exception as e:
        ph_status.error(f"Lỗi: {e}")

    time.sleep(refresh_rate)

if step_count >= max_steps:
    st.success(f"✅ Hoàn thành {step_count} bước mô phỏng!")
    st.balloons()
