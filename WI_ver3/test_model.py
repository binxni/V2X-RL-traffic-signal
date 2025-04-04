import torch
from env_sumo import SumoEnvironment
from ppo_agent import PPOAgent
from config import *


def test():
    env = SumoEnvironment({
        "use_gui": USE_GUI,
        "sumocfg_path": SUMO_CONFIG_FILE,
        "step_limit": T_HORIZON
    })

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        lr=LR,
        clip_eps=EPS_CLIP,
        epochs=K_EPOCH
    )

    agent.policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH))
    agent.policy_net.eval()

    try:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"\n✅ 테스트 에피소드 종료 - 총 보상: {total_reward:.2f}")
    finally:
        env.close()

if __name__ == '__main__':
    test()
