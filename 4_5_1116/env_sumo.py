import os
import sys
import numpy as np
import traci
import gym
from gym import spaces
from sumolib import checkBinary


class SumoEnvironment(gym.Env):
    def __init__(self, sumo_cfg_path, max_steps=1000, gui = False):
        super(SumoEnvironment, self).__init__()

        self.gui = gui
        self.sumo_cfg_path = sumo_cfg_path
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
        sumo_binary = checkBinary(
            'sumo') if not self.gui else checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg]

        traci.start(sumo_cmd)
        self.step_count = 0

        traci.simulationStep()  # ✅ 반드시 첫 스텝을 실행해서 연결 완료

        return self._get_observation()

    def step(self, action):
        # Action 분해: 선택된 phase와 지속 시간
        phase = int(np.clip(action[0], 0, 3))
        duration = int(np.clip(action[1], 5, 60))

        # 해당 phase 설정 및 duration 만큼 시뮬레이션 실행
        traci.trafficlight.setPhase("J1", phase)
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        # 현재 phase
        current_phase = traci.trafficlight.getPhase("J1")

        # 각 진입 차로 ID 리스트
        incoming_lanes = ["-E3_0", "-E1_0", "-E2_0", "E0_0"]

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

    def _compute_reward(self):
        # 보상 = 정체 차량 수의 음수 (작을수록 좋음)
        incoming_lanes = ["-E3_0", "-E1_0", "-E2_0", "E0_0"]
        total_halted = sum(traci.lane.getLastStepHaltingNumber(lane)
                           for lane in incoming_lanes)
        return -total_halted

    def close(self):
        if traci.isLoaded():
            traci.close()
