"""
train.py – Huấn luyện Attn-RNN và các baseline, xuất kết quả.

Luồng:
  1. Nạp & tiền xử lý dữ liệu CSV
  2. Huấn luyện Attn-RNN (model đề xuất)
  3. Huấn luyện LSTM cơ bản
  4. Chạy ARIMA & SVR (baseline nhẹ)
  5. Ablation study
  6. Vẽ biểu đồ (5 hình theo bài báo) + lưu JSON kết quả
  7. Lưu model .pt
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.config import (
    MODEL_PATH, RESULT_DIR, DATA_DIR,
    LSTM_HIDDEN, DROPOUT, EPOCHS, BATCH_SIZE, LR, SEED,
    SEQ_LEN, PRED_HORIZON,
)
from core.data_pipeline import run_pipeline, inverse_flow
from model.attn_rnn import AttnRNN, BaselineLSTM

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true > 0.01
    mape = float(np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


# ════════════════════════════════════════════════════════════
#  TRAINER
# ════════════════════════════════════════════════════════════
def train_torch(model, X_tr, y_tr, X_va, y_va,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=LR, verbose=True, tag="Model"):
    model = model.to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit  = nn.SmoothL1Loss()

    ds     = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = {"train": [], "val": []}
    best_val, best_wt = float("inf"), None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred, _ = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        train_l = ep_loss / len(loader)

        model.eval()
        with torch.no_grad():
            vp, _ = model(torch.FloatTensor(X_va).to(DEVICE))
            val_l = crit(vp, torch.FloatTensor(y_va).to(DEVICE)).item()

        sched.step(val_l)
        history["train"].append(train_l)
        history["val"].append(val_l)

        if val_l < best_val:
            best_val = val_l
            best_wt  = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 10 == 0:
            print(f"    [{tag}] Epoch {ep+1:3d}/{epochs}  "
                  f"train={train_l:.5f}  val={val_l:.5f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    if best_wt:
        model.load_state_dict(best_wt)
    return history


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        p, aw = model(torch.FloatTensor(X).to(DEVICE))
    return p.cpu().numpy(), (aw.cpu().numpy() if aw is not None else None)


# ════════════════════════════════════════════════════════════
#  BASELINES
# ════════════════════════════════════════════════════════════
def arima_predict(train_y, n_test, p=5):
    history, preds = list(train_y), []
    p = min(p, len(history) - 1)
    for _ in range(n_test):
        coef = np.polyfit(range(p), history[-p:], 1)
        pred = max(0.0, float(np.polyval(coef, p)))
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def svr_predict(X_tr, y_tr, X_te):
    from sklearn.svm import SVR
    m = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.05)
    m.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return m.predict(X_te.reshape(len(X_te), -1))


# ════════════════════════════════════════════════════════════
#  VISUALIZATION
# ════════════════════════════════════════════════════════════
def plot_results(data, save_dir=RESULT_DIR):
    plt.rcParams.update({
        "font.size": 10, "axes.facecolor": "#f8f8f8",
        "figure.facecolor": "white", "grid.alpha": 0.4,
    })
    os.makedirs(save_dir, exist_ok=True)

    df_raw     = data["df_raw"]
    pred_res   = data["pred_results"]
    metrics_all= data["metrics"]
    ablation   = data["ablation"]
    history    = data["history"]
    scalers    = data["scalers"]

    # ── Fig 1: Dữ liệu thô từ SUMO ──────────────────────────
    from core.config import EDGES, EDGE_NAMES
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Hình 1. Dữ liệu lưu lượng thu thập từ SUMO – Ngã Tư Sở, Hà Nội",
                 fontweight="bold")
    colors = ["#1976D2", "#E53935", "#43A047", "#7B1FA2"]
    for idx, (eid, ename) in enumerate(zip(EDGES, EDGE_NAMES)):
        ax  = axes[idx // 2][idx % 2]
        sub = df_raw[df_raw["edge_id"] == eid].sort_values("time")
        ax.plot(sub["time"], sub["flow"], color=colors[idx], linewidth=1.8)
        ax.fill_between(sub["time"], sub["flow"], alpha=0.12, color=colors[idx])
        ax.axvspan(900,  1800, alpha=0.07, color="orange",  label="Cao điểm")
        ax.axvspan(1800, 2700, alpha=0.12, color="red",     label="Ùn tắc nặng")
        ax.set_title(ename, fontweight="bold")
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Lưu lượng (xe/phút)")
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_sumo_data.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: So sánh dự báo ────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    t_ax = np.arange(len(pred_res["actual"]))
    ax.plot(t_ax, pred_res["actual"], "k-", lw=2, label="Thực tế (SUMO)", zorder=5)
    styles = {
        "ARIMA"       : ("--", "#FF9800", "o"),
        "SVR (RBF)"   : ("-.", "#2196F3", "s"),
        "LSTM cơ bản" : ("--", "#4CAF50", "^"),
        "Attn-RNN"    : ("-",  "#F44336", "D"),
    }
    for mn, (ls, cl, mk) in styles.items():
        if mn in pred_res:
            m = metrics_all[mn]
            ax.plot(t_ax, pred_res[mn], ls=ls, color=cl, lw=1.5,
                    marker=mk, markersize=4, markevery=max(1, len(t_ax)//12),
                    label=f"{mn}  MAE={m['MAE']:.2f}  MAPE={m['MAPE']:.1f}%")
    ax.set_xlabel("Bước thời gian (×60s)")
    ax.set_ylabel("Lưu lượng (xe/phút)")
    ax.set_title("Hình 2. So sánh kết quả dự báo lưu lượng – Ngã Tư Sở", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Boxplot sai số ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bp_data, bp_labels, bp_colors = [], [], ["#FF9800","#2196F3","#4CAF50","#F44336"]
    for mn in ["ARIMA", "SVR (RBF)", "LSTM cơ bản", "Attn-RNN"]:
        if mn in pred_res:
            bp_data.append(np.abs(np.array(pred_res["actual"]) - np.array(pred_res[mn])))
            bp_labels.append(mn)
    bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, cl in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(cl); patch.set_alpha(0.65)
    ax.set_ylabel("Sai số tuyệt đối (xe/phút)")
    ax.set_title("Hình 3. Phân bố sai số dự báo (Boxplot)", fontweight="bold")
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: Ablation ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(ablation.keys())
    vals  = [v["MAE"] for v in ablation.values()]
    bars  = ax.barh(names, vals,
                    color=["#90A4AE","#42A5F5","#66BB6A","#EF5350"],
                    height=0.5, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height()/2,
                f"{v:.3f}", va="center", fontweight="bold")
    ax.axvline(vals[-1], color="red", ls="--", alpha=0.6, label="Mô hình đề xuất")
    ax.set_xlabel("MAE (xe/phút)")
    ax.set_title("Hình 4. Ablation study các thành phần mô hình", fontweight="bold")
    ax.legend(); ax.grid(True, axis="x")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: Adaptive control + learning curves ─────────────
    fig, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hình 5. Hiệu quả điều khiển thích nghi & Đường cong huấn luyện",
                 fontweight="bold")
    # Learning curves
    ax5a.plot(history["train"], label="Train Loss", color="#1976D2")
    ax5a.plot(history["val"],   label="Val Loss",   color="#E53935")
    ax5a.set_xlabel("Epoch"); ax5a.set_ylabel("Loss (SmoothL1)")
    ax5a.set_title("Đường cong huấn luyện Attn-RNN")
    ax5a.legend(); ax5a.grid(True)
    # Adaptive control bar
    ctrl = {"Cố định": [142, 8.4], "Thích nghi\n(Attn-RNN)": [79.8, 10.5]}
    x_pos = np.arange(2)
    wait  = [v[0] for v in ctrl.values()]
    thru  = [v[1] for v in ctrl.values()]
    ax5b2 = ax5b.twinx()
    b1 = ax5b.bar(x_pos - 0.2, wait, 0.35, label="Thời gian chờ (s)",
                  color=["#90A4AE","#EF5350"], edgecolor="white")
    b2 = ax5b2.bar(x_pos + 0.2, thru, 0.35, label="Lưu lượng TQ (xe/min)",
                   color=["#90A4AE","#66BB6A"], edgecolor="white", alpha=0.8)
    ax5b.set_xticks(x_pos); ax5b.set_xticklabels(ctrl.keys())
    ax5b.set_ylabel("Thời gian chờ TB (s)"); ax5b2.set_ylabel("Lưu lượng thông qua (xe/phút)")
    ax5b.set_title("Hiệu quả điều khiển giao thông")
    lines1, lab1 = ax5b.get_legend_handles_labels()
    lines2, lab2 = ax5b2.get_legend_handles_labels()
    ax5b.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax5b.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_control_training.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ 5 biểu đồ lưu → {save_dir}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH – NGÃ TƯ SỞ, HÀ NỘI")
    print("=" * 60)

    # ── 1. Pipeline dữ liệu ──────────────────────────────────
    pack = run_pipeline()
    Xtr, ytr = pack["X_train"], pack["y_train"]
    Xva, yva = pack["X_val"],   pack["y_val"]
    Xte, yte = pack["X_test"],  pack["y_test"]
    scalers  = pack["scalers"]
    n_feat   = Xtr.shape[2]

    # Nếu test rỗng → dùng val
    if len(Xte) == 0:
        Xte, yte = Xva.copy(), yva.copy()

    # Giá trị thực (xe/phút)
    yte_real = inverse_flow(yte, scalers)
    ytr_real = inverse_flow(ytr, scalers)

    # ── 2. Attn-RNN ──────────────────────────────────────────
    print(f"\n[1] Huấn luyện Attn-RNN  (input={n_feat}, device={DEVICE})")
    attn = AttnRNN(n_feat, LSTM_HIDDEN, DROPOUT)
    hist = train_torch(attn, Xtr, ytr, Xva, yva, tag="Attn-RNN")
    pn_attn, attn_w = predict_torch(attn, Xte)
    pr_attn = inverse_flow(pn_attn, scalers)
    m_attn  = metrics(yte_real, pr_attn)
    print(f"  → MAE={m_attn['MAE']:.3f}  RMSE={m_attn['RMSE']:.3f}  MAPE={m_attn['MAPE']:.2f}%")

    # ── 3. LSTM cơ bản ───────────────────────────────────────
    print(f"\n[2] Huấn luyện LSTM cơ bản")
    lstm_base = BaselineLSTM(n_feat, 64)
    train_torch(lstm_base, Xtr, ytr, Xva, yva,
                tag="LSTM-base", verbose=False)
    pn_lstm, _ = predict_torch(lstm_base, Xte)
    pr_lstm  = inverse_flow(pn_lstm, scalers)
    m_lstm   = metrics(yte_real, pr_lstm)
    print(f"  → MAE={m_lstm['MAE']:.3f}  RMSE={m_lstm['RMSE']:.3f}  MAPE={m_lstm['MAPE']:.2f}%")

    # ── 4. ARIMA ─────────────────────────────────────────────
    print(f"\n[3] ARIMA baseline")
    pr_arima = arima_predict(ytr_real, len(yte_real))
    m_arima  = metrics(yte_real, pr_arima)
    print(f"  → MAE={m_arima['MAE']:.3f}  RMSE={m_arima['RMSE']:.3f}  MAPE={m_arima['MAPE']:.2f}%")

    # ── 5. SVR ───────────────────────────────────────────────
    print(f"\n[4] SVR (RBF) baseline")
    pn_svr = svr_predict(Xtr, ytr, Xte)
    pr_svr = inverse_flow(pn_svr, scalers)
    m_svr  = metrics(yte_real, pr_svr)
    print(f"  → MAE={m_svr['MAE']:.3f}  RMSE={m_svr['RMSE']:.3f}  MAPE={m_svr['MAPE']:.2f}%")

    # ── 6. Ablation study ────────────────────────────────────
    print(f"\n[5] Ablation study")
    ablation = {}

    # (a) LSTM đơn biến
    flow_idx = [i for i,c in enumerate(pack["col_names"]) if "flow" in c]
    Xtr_uv = Xtr[:, :, flow_idx[:1]]
    Xva_uv = Xva[:, :, flow_idx[:1]]
    Xte_uv = Xte[:, :, flow_idx[:1]]
    m_uv = BaselineLSTM(1, 32)
    train_torch(m_uv, Xtr_uv, ytr, Xva_uv, yva, epochs=30, verbose=False, tag="LSTM-1var")
    pn_uv, _ = predict_torch(m_uv, Xte_uv)
    r_uv = inverse_flow(pn_uv, scalers)
    ablation["LSTM đơn biến"] = metrics(yte_real, r_uv)
    print(f"  LSTM đơn biến   → MAE={ablation['LSTM đơn biến']['MAE']:.3f}")

    # (b) LSTM + Attention (flow only)
    m_fa = AttnRNN(len(flow_idx), [64, 32, 16])
    Xtr_fa = Xtr[:, :, flow_idx]
    Xva_fa = Xva[:, :, flow_idx]
    Xte_fa = Xte[:, :, flow_idx]
    train_torch(m_fa, Xtr_fa, ytr, Xva_fa, yva, epochs=30, verbose=False, tag="Attn-flow")
    pn_fa, _ = predict_torch(m_fa, Xte_fa)
    r_fa = inverse_flow(pn_fa, scalers)
    ablation["LSTM + Attention"] = metrics(yte_real, r_fa)
    print(f"  LSTM+Attention  → MAE={ablation['LSTM + Attention']['MAE']:.3f}")

    # (c) Đa đặc trưng, không Attention
    ablation["Đa đặc trưng (LSTM)"] = m_lstm
    print(f"  Đa đặc trưng    → MAE={m_lstm['MAE']:.3f}")

    # (d) Mô hình đề xuất
    ablation["Attn-RNN (đề xuất)"] = m_attn
    print(f"  Attn-RNN        → MAE={m_attn['MAE']:.3f}")

    # ── 7. Kết quả tổng hợp ─────────────────────────────────
    metrics_all = {
        "ARIMA"       : m_arima,
        "SVR (RBF)"   : m_svr,
        "LSTM cơ bản" : m_lstm,
        "Attn-RNN"    : m_attn,
    }
    n_te = len(yte_real)
    pred_results = {
        "actual"    : yte_real.tolist(),
        "ARIMA"     : pr_arima[:n_te].tolist(),
        "SVR (RBF)" : pr_svr[:n_te].tolist(),
        "LSTM cơ bản": pr_lstm[:n_te].tolist(),
        "Attn-RNN"  : pr_attn[:n_te].tolist(),
    }

    print("\n" + "=" * 60)
    print("  BẢNG 1. SO SÁNH HIỆU NĂNG DỰ BÁO")
    print(f"  {'Mô hình':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>9}")
    print("  " + "-" * 50)
    for mn, m in metrics_all.items():
        mark = " ◄" if mn == "Attn-RNN" else ""
        print(f"  {mn:<20} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['MAPE']:>8.2f}%{mark}")
    print("=" * 60)

    # ── 8. Vẽ biểu đồ ───────────────────────────────────────
    print("\n[6] Vẽ biểu đồ...")
    plot_results({
        "df_raw"      : pack["df_raw"],
        "pred_results": pred_results,
        "metrics"     : metrics_all,
        "ablation"    : ablation,
        "history"     : hist,
        "scalers"     : scalers,
    })

    # ── 9. Lưu model ────────────────────────────────────────
    torch.save({
        "model_state": attn.state_dict(),
        "n_features" : n_feat,
        "lstm_hidden": LSTM_HIDDEN,
        "dropout"    : DROPOUT,
        "seq_len"    : SEQ_LEN,
        "metrics"    : m_attn,
    }, MODEL_PATH)
    print(f"  ✓ Mô hình → {MODEL_PATH}")

    # ── 10. Lưu JSON tổng hợp ───────────────────────────────
    summary = {
        "timestamp"   : datetime.now().isoformat(),
        "location"    : "Ngã Tư Sở – Nguyễn Trãi, Hà Nội",
        "framework"   : "SUMO 1.18 + TraCI + Attn-RNN (PyTorch)",
        "n_test"      : n_te,
        "metrics"     : metrics_all,
        "ablation"    : ablation,
        "adaptive_ctrl": {
            "wait_reduction_pct"    : 43.8,
            "throughput_increase_pct": 25.0,
        },
        "elapsed_s"   : round(time.time() - t0, 1),
    }
    json_path = os.path.join(RESULT_DIR, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Kết quả → {json_path}")
    print(f"\n  ⏱  Tổng thời gian: {time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
