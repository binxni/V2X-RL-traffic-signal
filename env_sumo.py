import os
import sys
import random
import numpy as np
import traci
import sumolib


class SumoEnvironment:
    def __init__(self, config):
        # config: 설정 딕셔너리 (sumo path, step limit 등 포함)
        self.config = config
        self.sumo_binary = self._get_sumo_binary()
        self.step_limit = config.get("step_limit", 1000)
        self.use_gui = config.get("use_gui", False)
        self.sumo_cmd = self._build_sumo_command()
        self.reset()

    def _get_sumo_binary(self):
        # sumo or sumo-gui 경로 선택
        if self.config.get("use_gui", False):
            return sumolib.checkBinary("sumo-gui")
        else:
            return sumolib.checkBinary("sumo")


    def _build_sumo_command(self):
        # SUMO 실행 명령어 구성: config 파일 + 랜덤 시드 부여
        sumo_cfg = self.config["sumocfg_path"]
        seed = str(random.randint(0, 99999))
        # 기본 명령어
        cmd = [
            self.sumo_binary,
            "-c", sumo_cfg,
            "--random",
            "--seed", seed,
            "--start",
            "--quit-on-end",
            "--no-warnings"
        ]

        # GUI일 경우에만 창 크기 및 위치 지정
        if self.config.get("use_gui", False):
            cmd += [
                "--window-size", "800,600",     # ← 너비x높이
                "--window-pos", "100,100"       # ← 좌측 상단 위치 (x,y)
            ]

        return cmd

    def reset(self):
        # 환경 초기화: SUMO 시뮬레이션 시작
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        # 한 스텝 실행 + 보상 계산 + 다음 상태 반환
        self._apply_action(action)
        traci.simulationStep()
        self.step_count += 1

        state = self._get_state()
        reward = self._compute_reward()
        done = self.step_count >= self.step_limit

        return state, reward, done, {}

    def _apply_action(self, action):
        # 현재는 신호 그룹 ID 하나 사용한다고 가정
        tls_id = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tls_id, action)

    '''def _get_state(self):
        # 상태 추출: 신호 상태 + 각 차선의 차량 수, 평균 속도
        state = []
        tls_id = traci.trafficlight.getIDList()[0]
        state.append(traci.trafficlight.getPhase(tls_id))

        for lane_id in traci.lane.getIDList():
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            state.extend([veh_count, avg_speed])

        return np.array(state, dtype=np.float32)'''
    
    def _get_state(self):
        # 상태 추출: 신호 상태 + 주요 진입 차선 4개 + 전체 정지 차량 수 + 전체 평균 속도
        state = []

        # 1. 신호 상태 (Phase)
        tls_id = traci.trafficlight.getIDList()[0]
        state.append(traci.trafficlight.getPhase(tls_id))

        # 2. 주요 진입 차선 ID 리스트
        main_incoming_lanes = ['-E3_0', '-E2_0',
                               'E0_0', '-E1_0']  # 북, 남, 서, 동 방향 진입차선

        for lane_id in main_incoming_lanes:
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            state.extend([veh_count, avg_speed])

        # 3. 전체 정지 차량 수
        total_waiting = sum(traci.lane.getLastStepHaltingNumber(l)
                            for l in traci.lane.getIDList())
        state.append(total_waiting)

        # 4. 전체 평균 속도
        speeds = [traci.lane.getLastStepMeanSpeed(
            l) for l in traci.lane.getIDList()]
        avg_speed = np.mean(speeds) if speeds else 0.0
        state.append(avg_speed)

        return np.array(state, dtype=np.float32)

    def _compute_reward(self):
        lanes = traci.lane.getIDList()

        # 1. 정지 차량 수
        waiting_veh = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

        # 2. 통과한 차량 수
        passed_veh = traci.simulation.getArrivedNumber()

        # 3. 전체 평균 속도 (0~15 m/s 기준으로 정규화)
        speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes]
        avg_speed = np.mean(speeds) if speeds else 0
        normalized_speed = avg_speed / 15.0  # max speed 가정

        # 4. 오래 머문 차량 감점 (10초 이상 머무른 차량 수 체크)
        long_waiting_veh = 0
        # state에서는 4개 주요 진입 차선만 고려하기때문에 약간의 불일치 있을수있음
        for veh_id in traci.vehicle.getIDList():
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            if waiting_time > 10:  # 10초 이상 대기
                long_waiting_veh += 1

        # 보상 계산
        reward = (
            -1.0 * waiting_veh +
            1.0 * passed_veh +
            2.0 * normalized_speed -
            0.5 * long_waiting_veh
        )

        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()


    def get_average_waiting_time(self):
        # 전체 차량들의 평균 대기 시간 계산
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        total_wait = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        return total_wait / len(veh_ids)
