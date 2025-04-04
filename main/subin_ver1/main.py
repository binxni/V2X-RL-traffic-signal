import traci
import time
import numpy as np
from env.sumo_env_wrapper import get_state, apply_action, calculate_reward
from models.ppo_model import PPOActorCritic
from agents.ppo_utils import PPOAgent

sumo_binary = "sumo-gui"  # 또는 "sumo-gui"
sumocfg_file = "sumo/Wi_ver2/RL_0331.sumocfg"
tls_id = "J1"  # 신호등 ID (net.xml에서 확인 필요)
lane_ids = ['E0_0', 'E1_0', 'E2_0', 'E3_0']

# 하이퍼파라미터
input_dim = 16 # 상태 구성 요소 개수 (직접 조정 가능)
action_dim = 4
agent = PPOAgent(PPOActorCritic(input_dim, action_dim))

# 학습 루프 시작
episodes = 100
for episode in range(episodes):
    print(f"\n▶ Episode {episode + 1}")
    traci.start([sumo_binary, "-c", sumocfg_file])

    step = 0
    done = False
    prev_state = get_state(tls_id, lane_ids)

    states, actions, rewards, next_states, log_probs = [], [], [], [], []

    while traci.simulation.getMinExpectedNumber() > 0:

        action, log_prob = agent.select_action(prev_state)
        apply_action(tls_id, action)

        traci.simulationStep()
        time.sleep(0.05)  # 빠르게 돌리고 싶다면 제거 가능

        curr_state = get_state(tls_id, lane_ids)
        reward = calculate_reward(prev_state, curr_state)  # 임시: 실제 대기시간 기반 보상으로 수정 필요

        print(f"[Step {step}] Action: {action}")
        # print(f"State: {state}")
        print(f"Reward: {reward:.4f}")

        # 메모리에 저장
        states.append(prev_state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(curr_state)
        log_probs.append(log_prob)

        prev_state = curr_state
        step += 1

    loss = agent.update(states, actions, rewards, next_states, log_probs)
    print(f"Episode {episode + 1} finished. Loss: {loss:.4f}")
    traci.close()
