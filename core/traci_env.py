"""
traci_env.py – SUMO/TraCI environment
Tương thích Windows + Linux/macOS (SUMO_HOME tự động từ config).
"""
import os
import sys
import numpy as np

# Import config TRƯỚC để đảm bảo SUMO_HOME + sys.path được set
from core.config import (
    SUMO_HOME, SUMO_CMD, EDGES, EDGE_LEN_M, SAMPLE_INTERVAL,
    TL_ID, TL_PHASE_NS, TL_PHASE_EW,
    FLOW_HIGH_THRESHOLD, FLOW_LOW_THRESHOLD,
)

import traci
import traci.constants as tc


class SumoEnv:
    """
    Môi trường SUMO kết nối qua TraCI.
    Hỗ trợ:
      - Thu thập trạng thái (flow, speed, density, queue)
      - Điều khiển đèn tín hiệu thích nghi
    """

    def __init__(self, cmd=None, headless=True):
        self.cmd        = cmd or SUMO_CMD
        self.headless   = headless
        self._step      = 0
        self._connected = False

    def start(self):
        os.environ["SUMO_HOME"] = SUMO_HOME
        traci.start(self.cmd)
        self._connected = True
        self._step      = 0

    def close(self):
        if self._connected:
            try:
                traci.close()
            except Exception:
                pass
            self._connected = False

    def step(self):
        traci.simulationStep()
        self._step += 1
        return self._collect()

    def _collect(self):
        state = {}
        for eid in EDGES:
            try:
                n_veh   = traci.edge.getLastStepVehicleNumber(eid)
                speed   = traci.edge.getLastStepMeanSpeed(eid)
                halting = traci.edge.getLastStepHaltingNumber(eid)
            except Exception:
                n_veh, speed, halting = 0, 0.0, 0

            flow      = (n_veh / SAMPLE_INTERVAL) * 60.0
            density   = (n_veh / EDGE_LEN_M) * 1000.0
            speed_kmh = max(0.0, speed * 3.6)

            state[eid] = {
                "flow"    : round(flow,     3),
                "speed"   : round(speed_kmh, 3),
                "density" : round(density,  3),
                "queue"   : int(halting),
                "n_veh"   : int(n_veh),
            }
        return state

    def get_tl_phase(self, tl_id=TL_ID):
        try:
            return traci.trafficlight.getPhase(tl_id)
        except Exception:
            return -1

    def set_tl_phase(self, phase, tl_id=TL_ID):
        try:
            traci.trafficlight.setPhase(tl_id, phase)
        except Exception:
            pass

    def set_tl_duration(self, duration, tl_id=TL_ID):
        try:
            traci.trafficlight.setPhaseDuration(tl_id, duration)
        except Exception:
            pass

    def adaptive_control(self, flow_pred, tl_id=TL_ID):
        cur = self.get_tl_phase(tl_id)
        if flow_pred > FLOW_HIGH_THRESHOLD and cur != TL_PHASE_NS:
            self.set_tl_phase(TL_PHASE_NS, tl_id)
        elif flow_pred < FLOW_LOW_THRESHOLD and cur != TL_PHASE_EW:
            self.set_tl_phase(TL_PHASE_EW, tl_id)

    @property
    def sim_step(self):
        return self._step

    @property
    def connected(self):
        return self._connected
