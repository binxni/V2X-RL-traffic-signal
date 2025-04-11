import os
import csv
import numpy as np
import traci
from sumolib import checkBinary

# === 시뮬레이션 설정 ===
SUMO_CFG = "RL_0331.sumocfg"
USE_GUI = False
MAX_STEPS = 10000
CSV_PATH = "reward_component_stats.csv"


def run_sumo_and_collect():
    if traci.isLoaded():
        traci.close()

    sumo_binary = checkBinary("sumo-gui" if USE_GUI else "sumo")
    sumo_cmd = [sumo_binary, "-c", SUMO_CFG, "--start", "--quit-on-end"]
    traci.start(sumo_cmd)

    incoming_lanes = [
        "-E3_0", "-E3_1", "-E3_2",
        "-E1_0", "-E1_1", "-E1_2",
        "-E2_0", "-E2_1", "-E2_2",
        "E0_0", "E0_1", "E0_2"
    ]

    records = []

    for step in range(MAX_STEPS):
        traci.simulationStep()

        # 현재 phase 정보
        phase = traci.trafficlight.getPhase("J1")

        # 요소들 측정
        total_halted = sum(traci.lane.getLastStepHaltingNumber(l)
                           for l in incoming_lanes)
        avg_speed = np.mean([traci.lane.getLastStepMeanSpeed(l)
                            for l in incoming_lanes])
        long_waiting = 0
        for vid in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vid)
            wait = traci.vehicle.getWaitingTime(vid)
            if speed < 0.1 and wait >= 30:
                long_waiting += 1
        densities = []
        for l in incoming_lanes:
            num = traci.lane.getLastStepVehicleNumber(l)
            length = traci.lane.getLength(l)
            densities.append(num / length if length > 0 else 0)
        avg_density = np.mean(densities)

        records.append([step, phase, total_halted,
                       avg_speed, long_waiting, avg_density])

    traci.close()

    # CSV 저장
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "total_halted",
                        "avg_speed", "long_waiting", "avg_density"])
        writer.writerows(records)

    print(f"\n✅ 저장 완료: {CSV_PATH}")


if __name__ == "__main__":
    run_sumo_and_collect()
