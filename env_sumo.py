import os
import sys
import numpy as np
import traci
import gym
from gym import spaces
from sumolib import checkBinary


# Sumo 시뮬레이터와 PPO를 연동하는 Gym 환경 클래스
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
# SUMO GUI 사용 여부에 따라 실행 바이너리 선택
        sumo_binary = checkBinary(
            'sumo') if not self.gui else checkBinary('sumo-gui')
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
        lane_queues = [traci.lane.getLastStepHaltingNumber(
            lane) for lane in incoming_lanes]
        lane_speeds = [traci.lane.getLastStepMeanSpeed(
            lane) for lane in incoming_lanes]

        # 전체 정체 차량 수 및 평균 속도
        total_halted = sum(lane_queues)
        mean_speed = np.mean(lane_speeds)

        obs = [current_phase] + lane_queues + \
            lane_speeds + [total_halted, mean_speed]
        return np.array(obs, dtype=np.float32)

    '''def _compute_reward(self):
        # 보상 = 정체 차량 수의 음수 (작을수록 좋음)
        incoming_lanes = ["-E3_0", "-E1_0", "-E2_0", "E0_0"]
        total_halted = sum(traci.lane.getLastStepHaltingNumber(lane)
                           for lane in incoming_lanes)
        return -total_halted'''

    def _compute_reward(self):
        # 모든 진입 차선 (각 방향별 3개 차선씩)
        incoming_lanes = ["-E3_0", "-E3_1", "-E3_2",
                          "-E1_0", "-E1_1", "-E1_2",
                          "-E2_0", "-E2_1", "-E2_2",
                          "E0_0", "E0_1", "E0_2"]

        # 정지 차량 수 (속도 0)
        halts = [traci.lane.getLastStepHaltingNumber(
            lane) for lane in incoming_lanes]
        total_halted = sum(halts)

        # 평균 속도
        speeds = [traci.lane.getLastStepMeanSpeed(
            lane) for lane in incoming_lanes]
        avg_speed = np.mean(speeds)

        # 목적지에 도착한 차량 수 (비활성화 또는 보조 항목)
        # arrived_vehicles = traci.simulation.getArrivedNumber()

        # 장시간 정지 차량 수 (예: 30초 이상 정지)
        long_waiting = 0
        for veh_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(veh_id)
            wait_time = traci.vehicle.getWaitingTime(veh_id)
            if speed < 0.1 and wait_time >= 30:
                long_waiting += 1

        # ✅ 각 차선 밀도 계산: 차량 수 / 차선 길이
        densities = []
        for lane in incoming_lanes:
            veh_num = traci.lane.getLastStepVehicleNumber(lane)
            length = traci.lane.getLength(lane)
            if length > 0:
                densities.append(veh_num / length)

        avg_density = np.mean(densities)

        # ✅ 최종 보상 계산 (density는 감점 요인으로 추가)
        reward = (
            -1.0 * total_halted +
            +2.0 * avg_speed +
            -0.5 * long_waiting +
            -3.0 * avg_density  # ⬅️ 중요: 이 값은 실험하면서 튜닝
        )

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()

    def get_average_waiting_time(self):
        """
        현재 시점의 평균 대기 시간을 반환하는 함수
        """
        waiting_times = []
        for edge in traci.edge.getIDList():
            waiting_times.append(traci.edge.getWaitingTime(edge))
        return np.mean(waiting_times)
