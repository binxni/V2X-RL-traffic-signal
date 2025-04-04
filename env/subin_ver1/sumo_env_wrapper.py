import time
import traci
import numpy as np

# 교차로 중심 위치
INTERSECTION_CENTER = (5.32, -67.01)
EGO_ID = "ego"

def get_state(tls_id, lane_ids):
    state = []

    # 1. 현재 신호 phase (정수 → 정규화)
    phase = traci.trafficlight.getPhase(tls_id) / 10.0
    state.append(phase)

    # 2. 방향별 차량 수 및 정체 차량 수
    for lane in lane_ids:
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        num = len(vehicles)
        stopped = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)
        state.append(min(num / 50, 1.0))
        state.append(min(stopped / 50, 1.0))

    # 3. 자차 정보
    try:
        pos = traci.vehicle.getPosition(EGO_ID)
        speed = traci.vehicle.getSpeed(EGO_ID)
        angle = traci.vehicle.getAngle(EGO_ID)
        dx = INTERSECTION_CENTER[0] - pos[0]
        dy = INTERSECTION_CENTER[1] - pos[1]
        dist = (dx**2 + dy**2)**0.5

        state.extend([
            pos[0] / 100.0, pos[1] / 100.0,
            speed / 20.0,
            angle / 360.0,
            dist / 100.0
        ])
    except traci.TraCIException:
        state.extend([0.0, 0.0, 0.0, 0.0, 1.0])

    # 4. 주변 차량 평균 속도 (V2V 시뮬레이션)
    vehicle_ids = traci.vehicle.getIDList()
    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids if v != EGO_ID]
    avg_speed = np.mean(speeds) if speeds else 0.0
    state.append(avg_speed / 20.0)  # 정규화

    # 5. 현재 신호 phase의 남은 시간 (I2V 시뮬레이션)
    try:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        current_phase_index = traci.trafficlight.getPhase(tls_id)
        phase_duration = logic.phases[current_phase_index].duration
        phase_elapsed = traci.trafficlight.getPhaseDuration(tls_id)
        phase_remaining = max(phase_duration - phase_elapsed, 0)
        state.append(phase_remaining / 60.0)  # 정규화 (최대 60초 가정)
    except:
        state.append(0.0)

    return np.array(state, dtype=np.float32)

def apply_action(tls_id, action):
    # 현재 신호 상태 가져오기
    try:
        current_phase = traci.trafficlight.getRedYellowGreenState(tls_id)
    except Exception:
        current_phase = "r" * 20  # 초기 상태 대비용

    # 다음 phase 기본값: 모든 빨간불
    next_phase = list("r" * 20)

    # 행동에 따른 초록불 위치 설정
    if action == 0:
        next_phase[2] = "G"   # -E3→E2
        next_phase[12] = "G"  # -E2→E3
    elif action == 1:
        next_phase[7] = "G"   # -E1→-E0
        next_phase[17] = "G"  # E0→E1
    elif action == 2:
        next_phase[4] = "G"   # -E3→E1
        next_phase[14] = "G"  # -E2→-E0
    elif action == 3:
        next_phase[9] = "G"   # -E1→E2
        next_phase[19] = "G"  # E0→E3

    next_phase_str = "".join(next_phase)

    # 노란불 단계 계산: 초록 → 빨강으로 바뀌는 곳만 'y'
    yellow_phase = list(current_phase)
    for i in range(len(current_phase)):
        if current_phase[i] == "G" and next_phase[i] == "r":
            yellow_phase[i] = "y"
        else:
            yellow_phase[i] = current_phase[i]

    yellow_phase_str = "".join(yellow_phase)

    # 현재와 다르면 노란불 먼저 삽입
    if yellow_phase_str != current_phase:
        traci.trafficlight.setRedYellowGreenState(tls_id, yellow_phase_str)
        time.sleep(1.0)  # 노란불 유지 시간

    # 최종적으로 다음 phase 설정
    traci.trafficlight.setRedYellowGreenState(tls_id, next_phase_str)

def calculate_reward(prev_state, curr_state):
    reward = 0.0

    # 방향별 정체 차량 수 감소 보상 (lane 0~3)
    for i in range(1, 8, 2):  # index 1, 3, 5, 7 (정체 차량 비율 위치)
        prev_cong = prev_state[i]
        curr_cong = curr_state[i]
        reward += 1.0 * (prev_cong - curr_cong)

    # 전체 평균 속도 증가 보상 (index 14)
    reward += 0.5 * (curr_state[14] - prev_state[14])

    # ego 차량이 교차로 중심에 가까워질수록 보상 (index 13: dist to center)
    reward += 0.3 *(prev_state[13] - curr_state[13])

    #clipping
    reward = np.clip(reward, -1, 1)

    return reward