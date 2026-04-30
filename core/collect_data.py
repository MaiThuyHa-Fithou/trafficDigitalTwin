"""
collect_data.py – Thu thập dữ liệu giao thông từ SUMO qua TraCI.
Xuất file CSV: data/sumo_traffic_data.csv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from core.config import (
    EDGES, EDGE_NAMES, SIM_STEPS, SAMPLE_INTERVAL, DATA_DIR
)
from core.traci_env import SumoEnv


def collect_data(sim_steps: int = SIM_STEPS,
                 out_csv: str = None) -> pd.DataFrame:
    """
    Chạy mô phỏng SUMO và thu thập dữ liệu TraCI mỗi SAMPLE_INTERVAL giây.

    Returns
    -------
    df : DataFrame với cột [time, edge_id, flow, speed, density, queue]
    """
    out_csv = out_csv or os.path.join(DATA_DIR, "sumo_traffic_data.csv")
    env = SumoEnv()
    rows = []

    print("=" * 60)
    print("  THU THẬP DỮ LIỆU – NGÃ TƯ SỞ, HÀ NỘI")
    print(f"  Mô phỏng: {sim_steps}s | Lấy mẫu: mỗi {SAMPLE_INTERVAL}s")
    print("=" * 60)

    try:
        env.start()
        sample_no = 0

        for t in range(1, sim_steps + 1):
            state = env.step()

            if t % SAMPLE_INTERVAL == 0:
                sample_no += 1
                for eid, ename in zip(EDGES, EDGE_NAMES):
                    s = state[eid]
                    rows.append({
                        "time"       : t,
                        "edge_id"    : eid,
                        "edge_name"  : ename,
                        "flow"       : s["flow"],
                        "speed"      : s["speed"],
                        "density"    : s["density"],
                        "queue"      : s["queue"],
                    })

                # Progress bar
                pct = t / sim_steps * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                f0  = state[EDGES[0]]["flow"]
                q0  = state[EDGES[0]]["queue"]
                print(f"\r  [{bar}] {pct:5.1f}%  t={t:4d}s  "
                      f"Mẫu#{sample_no:2d}  "
                      f"Flow={f0:5.1f} xe/min  Queue={q0:3d}",
                      end="", flush=True)

    except Exception as e:
        print(f"\n  [LỖI] {e}")
        raise
    finally:
        env.close()

    print(f"\n  ✓ Hoàn thành: {sample_no} mẫu × {len(EDGES)} hướng")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ Đã lưu → {out_csv}")
    return df


if __name__ == "__main__":
    collect_data()
