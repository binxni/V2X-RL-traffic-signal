import os
import numpy as np
import traci
from sumolib import checkBinary
import csv
from env_sumo import SumoEnvironment
from config import SUMO_CONFIG_FILE

# 시뮬레이션 반복 횟수
NUM_EPISODES = 30
MAX_STEPS = 1024  # 1회 시뮬레이션당 step 수


def main():
    # 시뮬레이션 환경 설정 (GUI 안 켜도 됨)
    env = SumoEnvironment(sumo_cfg_path=SUMO_CONFIG_FILE,
                          max_steps=MAX_STEPS, gui=False)

    # 값 저장용 리스트
    total_halted_list = []
    avg_speed_list = []
    long_waiting_list = []
    avg_density_list = []

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False

        # ✅ 순차적으로 phase 변경: 0 → 1 → 2 → 3 → 0 ...
        phase = episode % 4
        duration = 20  # 고정 duration

        while not done:
            action = np.array([phase, duration])  # 각 에피소드에서 고정된 phase & duration
            _, reward, done, _ = env.step(action)


            # 보상 요소 직접 계산
            incoming_lanes = ["-E3_0", "-E3_1", "-E3_2",
                              "-E1_0", "-E1_1", "-E1_2",
                              "-E2_0", "-E2_1", "-E2_2",
                              "E0_0", "E0_1", "E0_2"]

            halts = [traci.lane.getLastStepHaltingNumber(
                lane) for lane in incoming_lanes]
            speeds = [traci.lane.getLastStepMeanSpeed(
                lane) for lane in incoming_lanes]
            total_halted = sum(halts)
            avg_speed = np.mean(speeds)

            # long_waiting
            long_waiting = 0
            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(veh_id) < 0.1 and traci.vehicle.getWaitingTime(veh_id) >= 30:
                    long_waiting += 1

            # avg_density
            densities = []
            for lane in incoming_lanes:
                veh_num = traci.lane.getLastStepVehicleNumber(lane)
                length = traci.lane.getLength(lane)
                if length > 0:
                    densities.append(veh_num / length)
            avg_density = np.mean(densities)

            total_halted_list.append(total_halted)
            avg_speed_list.append(avg_speed)
            long_waiting_list.append(long_waiting)
            avg_density_list.append(avg_density)

        print(f"[Episode {episode+1}] Total Halted: {total_halted:.2f} | Avg Speed: {avg_speed:.2f} | "
              f"Long Wait: {long_waiting:.2f} | Avg Density: {avg_density:.4f}")

        env.close()

    # 통계 계산 및 저장
    save_csv_summary(total_halted_list, avg_speed_list,
                     long_waiting_list, avg_density_list)


def save_csv_summary(halted, speed, wait, density, filename="reward_component_stats.csv"):
    data = {
        "Metric": ["Total Halted", "Average Speed", "Long Waiting", "Average Density"],
        "Mean": [np.mean(halted), np.mean(speed), np.mean(wait), np.mean(density)],
        "Min": [np.min(halted), np.min(speed), np.min(wait), np.min(density)],
        "Max": [np.max(halted), np.max(speed), np.max(wait), np.max(density)],
    }

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Min", "Max"])
        for i in range(4):
            writer.writerow(
                [data["Metric"][i], f"{data['Mean'][i]:.2f}", f"{data['Min'][i]:.2f}", f"{data['Max'][i]:.2f}"])

    print(f"\n📁 저장 완료: {filename}")


if __name__ == "__main__":
    main()
