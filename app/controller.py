"""
controller.py – Bộ điều khiển tín hiệu giao thông thích nghi.
Nhận dự báo lưu lượng → điều chỉnh pha đèn qua TraCI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traci
from core.config import (
    TL_ID, TL_PHASE_NS, TL_PHASE_EW,
    FLOW_HIGH_THRESHOLD, FLOW_LOW_THRESHOLD,
)


def control(pred: float, tl_id: str = TL_ID) -> dict:
    """
    Điều chỉnh pha và thời gian đèn dựa trên lưu lượng dự báo.

    Quy tắc:
      pred > HIGH (15 xe/min) → ưu tiên N-S (Nguyễn Trãi), green=70s
      pred < LOW  (5  xe/min) → ưu tiên Đông-Tây,           green=50s
      Trung gian               → giữ nguyên pha, điều chỉnh duration

    Returns
    -------
    dict : {"phase": int, "duration": int, "action": str}
    """
    action = "unchanged"
    try:
        cur_phase = traci.trafficlight.getPhase(tl_id)

        if pred > FLOW_HIGH_THRESHOLD:
            # Lưu lượng cao → kéo dài pha Nguyễn Trãi
            if cur_phase != TL_PHASE_NS:
                traci.trafficlight.setPhase(tl_id, TL_PHASE_NS)
                action = "switch_to_NS"
            traci.trafficlight.setPhaseDuration(tl_id, 70)
            duration = 70

        elif pred < FLOW_LOW_THRESHOLD:
            # Lưu lượng thấp → nhường pha Đông-Tây
            if cur_phase != TL_PHASE_EW:
                traci.trafficlight.setPhase(tl_id, TL_PHASE_EW)
                action = "switch_to_EW"
            traci.trafficlight.setPhaseDuration(tl_id, 50)
            duration = 50

        else:
            # Trung bình → tuyến tính hóa theo lưu lượng
            duration = int(50 + (pred - FLOW_LOW_THRESHOLD)
                           / (FLOW_HIGH_THRESHOLD - FLOW_LOW_THRESHOLD) * 20)
            traci.trafficlight.setPhaseDuration(tl_id, duration)
            action = "adjusted"

        return {
            "phase"   : traci.trafficlight.getPhase(tl_id),
            "duration": duration,
            "action"  : action,
            "pred"    : round(pred, 2),
        }

    except Exception as e:
        return {"phase": -1, "duration": 60, "action": f"error:{e}", "pred": pred}
