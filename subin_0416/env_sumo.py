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
        self.sumo_cfg_path = sumo_cfg_path
        self.max_steps = max_steps
        self.step_count = 0
        self.prev_halted = 0

        # ✅ 행동 공간 재정의 (Phase: 이산, Duration: 연속)
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),        # Phase: 0~3
            spaces.Box(low=5, high=60, shape=(1,), dtype=np.float32)  # Duration: 5~60초
        ))

        # ✅ 상태 공간 재설계 (실제 관측 데이터와 일치하도록)
        self.observation_space = spaces.Dict({
            'phase': spaces.Box(0, 3, shape=(1,), dtype=np.int32),
            'queues': spaces.Box(0, 100, shape=(12,), dtype=np.float32),  # 12개 차선 대기열
            'speeds': spaces.Box(0, 50, shape=(12,), dtype=np.float32)   # 12개 차선 속도 (km/h)
        })


    def reset(self):
        if traci.isLoaded():
            traci.close()
        sumo_binary = checkBinary('sumo') if not self.gui else checkBinary('sumo-gui')
        sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path,
                    "--start",  "--no-warnings", "--quit-on-end"]
        traci.start(sumo_cmd)
        self.step_count = 0
        self.prev_halted = 0

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
        # 1. 신호 페이즈 정보 one-hot 인코딩
        current_phase = traci.trafficlight.getPhase("J1")
        num_phases = 4  # 실제 교차로 신호 단계 수로 변경 필요
        phase_one_hot = [0] * num_phases
        phase_one_hot[current_phase] = 1

        # 2. 차선별 정보 수집 (12개 차선)
        incoming_lanes = [
            "-E3_0", "-E3_1", "-E3_2",
            "-E1_0", "-E1_1", "-E1_2",
            "-E2_0", "-E2_1", "-E2_2",
            "E0_0", "E0_1", "E0_2"
        ]

        # 3. Q_MAX, N_MAX 계산 (net.xml 기반)
        vehicle_space = 7.5  # 차량 길이(5m) + 안전 거리(2.5m)
        lane_lengths = [160] * 12  # 모든 외부 차로 길이 160m (net.xml 확인)
        Q_MAX = int(min([length // vehicle_space for length in lane_lengths]))  # 21
        N_MAX = int(max([length // vehicle_space for length in lane_lengths]))  # 21
        V_MAX = 13.89

        lane_features = []
        for lane in incoming_lanes:
            # 4-1. 정체 차량 수 (0~1)
            halted = traci.lane.getLastStepHaltingNumber(lane) / Q_MAX
            
            # 4-2. 총 차량 수 (0~1)
            total_vehicles = traci.lane.getLastStepVehicleNumber(lane) / N_MAX
            
            # 4-3. 평균 속도 (차로별 최대 속도 기반 정규화)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            speeds = []
            for v_id in vehicle_ids:
                lane_pos = traci.vehicle.getLanePosition(v_id)
                if lane_pos <= 100:  # 100m 이내 차량만 포함
                    speed = traci.vehicle.getSpeed(v_id)
                    if speed < 0.1:  # 정지 상태 차량 처리
                        speed = 0.0
                    speeds.append(speed)
            
            # 차로별 최대 속도로 정규화        
            avg_speed = (np.mean(speeds) if speeds else 0.0) / V_MAX
            
            lane_features.extend([halted, total_vehicles, avg_speed])

        # 5. 상태 벡터 조합
        observation = phase_one_hot + lane_features
        
        return np.array(observation, dtype=np.float32)

    def _compute_reward(self):
        # 관측 차선 (외부 12개 차선)
        incoming_lanes = ["-E3_0", "-E3_1", "-E3_2",
                        "-E1_0", "-E1_1", "-E1_2",
                        "-E2_0", "-E2_1", "-E2_2",
                        "E0_0", "E0_1", "E0_2"]

        # 1. 대기열 감소 (Δ 정체 차량)
        current_halted = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)
        #delta_halted = getattr(self, 'prev_halted', current_halted) - current_halted
        #self.prev_halted = current_halted

        # 2. 속도 향상 (정지선 100m 이내)
        speeds = []
        for lane in incoming_lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getLanePosition(veh_id) <= 100:
                    speeds.append(traci.vehicle.getSpeed(veh_id))
        avg_speed = np.mean(speeds) if speeds else 0.0

        # 3. 차량 수 제어 (정지선 100m 이내)
        total_vehicles_in_range = 0
        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            total_vehicles_in_range += sum(
                1 for veh_id in vehicle_ids 
                if traci.vehicle.getLanePosition(veh_id) <= 100
            )

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

        # 정규화 상수 (실제 교차로 데이터 기반)
        MAX_HALTED = 150    # 최대 정체 차량 수 (차선당 15~20대)
        MAX_SPEED = 13.89   # 50 km/h → 13.89 m/s
        MAX_VEHICLES = 200  # 12차선 × 20대
        MAX_DENSITY = 0.5   # 차량/m (20대/100m)

        # 보상 계산 (가중치 실험적으로 조정)
        reward = (
            -2.0 * (current_halted / MAX_HALTED) +  # 대기열 
            +1.0* (avg_speed / MAX_SPEED) +      # 속도 향상 
            -1.0 * (total_vehicles_in_range / MAX_VEHICLES) +  # 차량 수 제어
            -1.0 * (avg_density / MAX_DENSITY) +   # 밀도 제어
            -1.5 * (get_average_waiting_time()/100)
        )

        return reward 
    
    def close(self):
        if traci.isLoaded():
            traci.close()

