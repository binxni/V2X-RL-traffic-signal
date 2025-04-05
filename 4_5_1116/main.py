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
    env = SumoEnvironment(
        sumo_cfg_path=SUMO_CONFIG_FILE,  # SUMO 시뮬레이션 설정 파일 경로
        max_steps=T_HORIZON,             # 한 에피소드에서 최대 시뮬레이션 스텝 수
        gui=USE_GUI                      # GUI 실행 여부
    )

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        gamma=GAMMA,
        clip_eps=EPS_CLIP,
        epochs=K_EPOCH
    )

    # --- TensorBoard 및 CSV 로그 파일 설정 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir=f"runs/ppo_sumo_{timestamp}")

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/train_log_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            ["Episode", "Reward", "PolicyLoss", "ValueLoss", "AvgWaitingTime"])

    os.makedirs("models_backup", exist_ok=True)

    # --- 학습 루프 시작 ---
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        score = 0  # 누적 리워드

        for t in range(T_HORIZON):
            # 현재 상태에서 행동 선택
            action, log_prob = agent.select_action(state)

            # 환경에 액션 적용
            next_state, reward, done, _ = env.step(action)
            score += reward

            # Transition 저장
            agent.store_transition(
                state, action, log_prob, reward, next_state, done)

            state = next_state

            if done:
                break

        # --- PPO 에이전트 업데이트 ---
        policy_loss, value_loss = agent.update(next_state, done)

        # --- 대기 시간 계산 ---
        avg_waiting_time = env.get_average_waiting_time()

        # --- 콘솔 출력 ---
        print(f"[Episode {episode:03d}] Reward: {score:.2f} | "
              f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | "
              f"Avg Waiting Time: {avg_waiting_time:.2f}s")

        # --- TensorBoard 기록 ---
        writer.add_scalar("Reward/Total", score, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        writer.add_scalar("Traffic/Average_Waiting_Time",
                          avg_waiting_time, episode)

        # --- CSV 기록 ---
        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(
                [episode, score, policy_loss, value_loss, avg_waiting_time])

        # --- 에피소드별 모델 저장 ---
        backup_path = f"models_backup/ppo_sumo_ep{episode:03d}_{timestamp}.pth"
        torch.save(agent.policy_net.state_dict(), backup_path)

    # --- 최종 모델 저장 ---
    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    writer.close()
    print("\n✅ Training Complete. Model saved to:", MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()

# 끝나고 아래 명령어로 tensorboard 실행
# tensorboard --logdir=runs
# 해당 명령어 치고나서 브라우저에 http://localhost:6006 해서 들어가면 실시간 로그 확인 가능
