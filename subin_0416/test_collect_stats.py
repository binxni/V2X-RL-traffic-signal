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
    total_waiting_time = 0  # 전체 차량의 대기 시간 합산
    total_vehicle_count = 0  # 전체 차량 수 합산

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

        step_waiting_time = 0  # 현재 스텝에서의 대기 시간 합산
        step_vehicle_count = len(traci.vehicle.getIDList())  # 현재 스텝에서 관측된 차량 수

        for vid in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vid)
            wait = traci.vehicle.getWaitingTime(vid)
            if speed < 0.1 and wait >= 30:
                long_waiting += 1

            # 현재 스텝에서의 대기 시간 합산
            step_waiting_time += wait

        densities = []
        for l in incoming_lanes:
            num = traci.lane.getLastStepVehicleNumber(l)
            length = traci.lane.getLength(l)
            densities.append(num / length if length > 0 else 0)
        avg_density = np.mean(densities)

        # 전체 대기 시간 및 차량 수 업데이트
        total_waiting_time += step_waiting_time
        total_vehicle_count += step_vehicle_count

        records.append([step, phase, total_halted,
                       avg_speed, long_waiting, avg_density])

    traci.close()

    # 전체 차량의 평균 대기 시간 계산
    if total_vehicle_count > 0:
        avg_waiting_time_per_vehicle = total_waiting_time / total_vehicle_count
    else:
        avg_waiting_time_per_vehicle = 0

    print(f"\n✅ 전체 차량의 평균 대기 시간: {avg_waiting_time_per_vehicle:.2f} 초")

    # CSV 저장
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "total_halted",
                        "avg_speed", "long_waiting", "avg_density"])
        writer.writerows(records)

    print(f"\n✅ 저장 완료: {CSV_PATH}")


if __name__ == "__main__":
    run_sumo_and_collect()
