import os
import sys
import numpy as np
import traci
import gym
from gym import spaces
from sumolib import checkBinary


def get_average_waiting_time():
    total_waiting_time = 0.0
    halted_vehicle_count = 0

    for veh_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(veh_id)
        if speed < 0.1:  # 정체된 차량 (속도 거의 0)
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time
            halted_vehicle_count += 1

    if halted_vehicle_count > 0:
        avg_waiting_time = total_waiting_time / halted_vehicle_count
    else:
        avg_waiting_time = 0.0

    return avg_waiting_time

class SumoEnvironment(gym.Env):
    def __init__(self, sumo_cfg_path, max_steps=1000, gui=True):
        super(SumoEnvironment, self).__init__()

        self.gui = gui
        self.gui = gui
        # SUMO 환경 설정 경로 저장
        self.sumo_cfg_path = sumo_cfg_path
        # 시뮬레이션 최대 스텝 수 저장
        self.max_steps = max_steps
        self.step_count = 0

        # Phase: 0~3 (4개 phase), Duration: 5~60초 → 총 action space
        self.action_space = spaces.Box(low=np.array(
            [0, 5]), high=np.array([3, 60]), dtype=np.float32)

        # Observation: [현재 페이즈 번호(1)] + [각 진입차로의 대기 차량 수 4개] + [각 진입차로의 평균 속도 4개]
        #             + [전체 정체 차량 수(1)] + [전체 평균 속도(1)] → 총 11차원
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(11,), dtype=np.float32)


    def reset(self):
        if traci.isLoaded():
            traci.close()
        sumo_binary = checkBinary('sumo') if not self.gui else checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path,
                    "--start",  "--no-warnings", "--quit-on-end"]
        traci.start(sumo_cmd)
        self.step_count = 0

        # ✅ 신호등 제어 초기 설정 (여기 추가)
        # traci.trafficlight.setProgram("J1", "0")  # tlLogic ID "J1", programID "0"
        traci.simulationStep()  # ✅ 반드시 첫 스텝을 실행해서 연결 완료

        return self._get_observation()

    def step(self, action):
        # Action 분해: 선택된 phase와 지속 시간
        phase = int(np.clip(action[0], 0, 3))
        duration = int(np.clip(action[1], 5, 60))

        print(f"[ENV] Phase: {phase}, Duration: {duration}")

        # 해당 phase 설정 및 duration 만큼 시뮬레이션 실행
        traci.trafficlight.setPhase("J1", phase)
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps
        info = {}
        print(
            f"[ENV] Phase: {phase}, Duration: {duration}, Reward: {reward:.2f}")
        return obs, reward, done, info

    def _get_observation(self):
        # 현재 phase
        current_phase = traci.trafficlight.getPhase("J1")

        # 각 진입 차로 ID 리스트
        incoming_lanes = ["-E3_0", "-E3_1", "-E3_2",
                        "-E1_0", "-E1_1", "-E1_2",
                        "-E2_0", "-E2_1", "-E2_2",
                        "E0_0", "E0_1", "E0_2"]

        # 각 진입 차로의 정체 차량 수 및 평균 속도 수집
        lane_queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes]

        # 각 차로의 속도를 정지선으로부터 100m 이내에서만 수집
        lane_speeds = []
        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            speeds_in_range = [
                traci.vehicle.getSpeed(vehicle_id)
                for vehicle_id in vehicle_ids
                if traci.vehicle.getLanePosition(vehicle_id) <= 100  # 정지선으로부터 150m 이내
            ]
            # 해당 구간 내 평균 속도 계산 (차량이 없으면 0으로 설정)
            if speeds_in_range:
                lane_speeds.append(np.mean(speeds_in_range))
            else:
                lane_speeds.append(0.0)

        average_waiting_time = get_average_waiting_time()

        # 전체 정체 차량 수 및 평균 속도
        total_halted = sum(lane_queues)
        mean_speed = np.mean(lane_speeds)

        obs = [current_phase] + lane_queues + lane_speeds + [total_halted, mean_speed] + [average_waiting_time]
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # 모든 진입 차선 (각 방향별 3개 차선씩)
        incoming_lanes = ["-E3_0", "-E3_1", "-E3_2",
                        "-E1_0", "-E1_1", "-E1_2",
                        "-E2_0", "-E2_1", "-E2_2",
                        "E0_0", "E0_1", "E0_2"]

        # 정지 차량 수 (속도 0)
        halts = [traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes]
        total_halted = sum(halts)

        # ✅ 정지선 기준 100m 이내 구간에서 평균 속도 계산
        speeds_in_range = []
        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicle_ids:
                position = traci.vehicle.getLanePosition(veh_id)
                if position <= 100:  # 정지선 기준 100m 이내
                    speeds_in_range.append(traci.vehicle.getSpeed(veh_id))
        avg_speed = np.mean(speeds_in_range) if speeds_in_range else 0.0

        # 장시간 정지 차량 수 (예: 30초 이상 정지)
        long_waiting = 0
        for veh_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(veh_id)
            wait_time = traci.vehicle.getWaitingTime(veh_id)
            if speed < 0.1 and wait_time >= 30:
                long_waiting += 1

        # ✅ 정지선 기준 100m 이내 구간에서 밀도 계산: 차량 수 / 해당 구간 길이
        densities = []
        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            veh_count_in_range = sum(
                1 for veh_id in vehicle_ids if traci.vehicle.getLanePosition(veh_id) <= 100
            )
            length_in_range = min(100, traci.lane.getLength(lane))  # 구간 길이는 최대 100m로 제한
            if length_in_range > 0:
                densities.append(veh_count_in_range / length_in_range)

        avg_density = np.mean(densities) if densities else 0.0

        # 보상 계산
        reward = (
            -1.0 * (total_halted / len(incoming_lanes)) +
            +2.0 * (avg_speed / 13.9) +
            -2.0 * (long_waiting / 1.0) +
            -1.0 * (avg_density / 0.0156) +
            -1.5 * (get_average_waiting_time()/60)
        )

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()

