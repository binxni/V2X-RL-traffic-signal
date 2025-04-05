# main.py
import os
os.environ["SUMO_HOME"] = "C:/Program Files (x86)/Eclipse/Sumo"

from som_env import TrafficSignalEnv
from som_ppo_model import PPO
import numpy as np
import torch
import traci
import matplotlib.pyplot as plt

def train():
    env = TrafficSignalEnv("som_v2x.sumocfg")
    state_dim = len(env.reset())
    action_dim = 3  # 신호 페이즈 개수
    agent = PPO(state_dim, action_dim)
    episode_rewards = []
    
    # 학습 로그 파일 초기화
    log_file = open("training_log.txt", "w")
    log_file.write("Episode, Reward\n")  # 헤더 추가
    
    try:
        for ep in range(1000):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action_probs, _ = agent.policy(torch.FloatTensor(state).cpu())
                action = np.random.choice(action_dim, p=action_probs.cpu().detach().numpy())
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(agent.memory) >= 10:
                    agent.update(
                        states=np.array([m[0] for m in agent.memory]),
                        actions=np.array([m[1] for m in agent.memory]),
                        rewards=np.array([m[2] for m in agent.memory]),
                        dones=np.array([m[4] for m in agent.memory])
                    )
                    agent.memory.clear()

            episode_rewards.append(total_reward)
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}")
            
            # 로그 파일에 기록 (CSV 형식)
            log_file.write(f"{ep+1}, {total_reward:.2f}\n")
            log_file.flush()  # 실시간 저장 보장

            # 10회마다 모델 및 시각화 결과 저장
            if (ep+1) % 10 == 0:
                torch.save(agent.policy.state_dict(), f"ppo_traffic_{ep+1}.pth")
                visualize_results(episode_rewards, save_path=f"progress_{ep+1}.png")

    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다. 현재까지 결과를 저장합니다.")
        visualize_results(episode_rewards, save_path="interrupted_progress.png")
        
    finally:
        if traci.isLoaded():
            traci.close()
        log_file.close()  # 반드시 파일 닫기

def visualize_results(episode_rewards, save_path="training_progress.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train()
