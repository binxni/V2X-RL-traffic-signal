# env.py
import traci
import numpy as np
from typing import Tuple

class TrafficSignalEnv:
    def __init__(self, config_path: str, sumo_cmd: list = None):
        self.sumo_cmd = sumo_cmd or [
            "sumo", "-c", config_path,
            "--time-to-teleport", "300",
            "--waiting-time-memory", "1000"
        ]
        self.signal_phases = {
            0: "GGrrrrGGrrrr",  # NS 직진 허용
            1: "rrrGGGrrrGGG",  # EW 직진 허용
            2: "yyyyrrryyyyrr"  # 모든 방향 황색
        }
        self.episode_step = 0
        self.max_steps = 1000
        
    def reset(self) -> np.ndarray:
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        traci.simulationStep()
        self.episode_step = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """차량 밀도(0-1), 평균 속도(0-1), 대기 시간(0-1), 현재 페이즈(원핫)"""
        vehicles = traci.vehicle.getIDList()
        
        # 차량 밀도 계산
        density = [
            traci.lane.getLastStepVehicleNumber(lane) / 20.0  # 최대 20대 가정
            for lane in traci.lane.getIDList() if 'E' in lane
        ]
        
        # 속도 정규화 (0~50km/h 기준)
        speeds = [traci.vehicle.getSpeed(v)/13.9 for v in vehicles]  # 13.9m/s = 50km/h
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # 대기 시간
        waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        
        # 현재 신호 페이즈 (원핫 인코딩)
        current_phase = traci.trafficlight.getPhase('J1')  # 교차로 ID 가정
        phase_onehot = np.eye(3)[current_phase]
        
        return np.concatenate([density, [avg_speed, waiting], phase_onehot])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        traci.trafficlight.setPhase('J1', action)
        traci.simulationStep()  # 1 step 진행
        
        reward = self._calculate_reward()
        self.episode_step += 1
        done = self.episode_step >= self.max_steps
        
        return self._get_state(), reward, done, {}

    def _calculate_reward(self) -> float:
        """대기 시간, 처리량, 긴급 제동 정규화"""
        vehicles = traci.vehicle.getIDList()
        max_waiting_time = 1000  # 최대 대기 시간 가정
        max_throughput = 100     # 최대 처리량 가정
        max_emergency = 50       # 최대 긴급 제동 횟수 가정

        waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / max_waiting_time
        throughput = traci.simulation.getArrivedNumber() / max_throughput
        emergency = sum(traci.vehicle.getEmergencyDecel(v) for v in vehicles) / max_emergency

        """대기 시간(40%) + 처리량(40%) + 긴급 제동(20%)"""
        return -0.4 * waiting + 0.4 * throughput - 0.2 * emergency

