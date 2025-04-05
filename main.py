import csv
import os
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env_sumo import SumoEnvironment
from ppo_agent import PPOAgent
from config import *


def main():
    # --- 환경 및 에이전트 초기화 ---
    env = SumoEnvironment({
        "use_gui": USE_GUI,
        "sumocfg_path": SUMO_CONFIG_FILE,
        "step_limit": T_HORIZON
    })

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        gamma=GAMMA,
        clip_eps=EPS_CLIP,
        epochs=K_EPOCH
    )

    # --- 로그 디렉토리 및 파일 준비 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir=f"runs/ppo_sumo_{timestamp}")

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/train_log_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Episode", "Reward", "PolicyLoss", "ValueLoss"])

    os.makedirs("models_backup", exist_ok=True)

    # --- 학습 루프 ---
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        score = 0 # 이번 episode에서 받은 총 reward를 누적하기 위해 사용하는 변수

        for t in range(T_HORIZON):
        # 하나의 episode 안에서 최대 T_HORIZON만큼 step 진행
            action, log_prob = agent.select_action(state)
            # 현재 state에서 agent가 policy network(actor)를 통해 action을 선택
            next_state, reward, done, _ = env.step(action)
            # 선택한 action을 적용하고 한 step 시뮬레이션을 진행 => 다음 상태, 보상, 종료여부를 반환받음
            
            agent.store_transition(
                # 하나의 transition을 agent의 메모리에 저장 -> 나중에 PPO 업데이트에 사용됨
                state, action, log_prob, reward, next_state, done)
            

            state = next_state
            score += reward

            if done:
                break

        # --- 정책 업데이트 ---
        policy_loss, value_loss = agent.update(next_state, done)
        # Actor & Critic 업데이트 호출

        # --- 로그 출력 ---
        print(
            f"[Episode {episode:03d}] Reward: {score:.2f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")

        # --- TensorBoard 로그 기록 ---
        writer.add_scalar("Reward/Total", score, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)

        # --- CSV 로그 저장 ---
        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([episode, score, policy_loss, value_loss])

        # --- 모델 백업 저장 ---
        backup_path = f"models_backup/ppo_sumo_ep{episode:03d}_{timestamp}.pth"
        torch.save(agent.policy_net.state_dict(), backup_path)

    # --- 최종 모델 저장 ---
    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    writer.close()
    print("\n\U00002705 Training Complete. Model saved to:", MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()

# 끝나고 아래 명령어로 tensorboard 실행
# tensorboard --logdir=runs
# 해당 명령어 치고나서 브라우저에 http://localhost:6006 해서 들어가면 실시간 로그 확인 가능

