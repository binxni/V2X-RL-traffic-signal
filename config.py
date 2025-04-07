# config.py

import os

# === SUMO 환경 설정 ===
#USE_GUI = False  # True면 sumo-gui, False면 CLI(sumod)
USE_GUI = True
SUMO_BINARY = "sumo-gui" if USE_GUI else "sumo"
#SUMO_CONFIG_FILE = os.path.join("env", "RL_0331.sumocfg")  # sumo config 파일 경로
SUMO_CONFIG_FILE = "RL_0331.sumocfg"

# === PPO 하이퍼파라미터 ===
GAMMA = 0.99               # Discount factor
LR = 1e-4                  # Learning rate
EPS_CLIP = 0.1             # PPO clip ratio
K_EPOCH = 3                # PPO 학습 반복 횟수
T_HORIZON = 1024           # 한 에피소드에서 수집할 step 수


# === 에이전트 설정 ===
STATE_DIM = 27             # 상태 벡터 차원 (state 정의에 따라 수정 가능)
ACTION_DIM = 4             # 행동 차원 (ex. 4개의 신호 조합 등)
HIDDEN_DIM = 128            # NN 히든 레이어 차원

# === 학습 설정 ===
MAX_EPISODES = 1000         # 총 학습할 에피소드 수
LOG_INTERVAL = 2          # 몇 에피소드마다 로그 출력
MODEL_SAVE_PATH = os.path.join("models", "ppo_sumo.pth")
SEED = None                # None이면 매번 랜덤, 아니면 시드 고정

# config 딕셔너리를 따로 만들어주기
config = {
    "use_gui": USE_GUI,
    "sumocfg_path": SUMO_CONFIG_FILE,
    "step_limit": T_HORIZON,
    "gamma": GAMMA,
    "lr": LR,
    "eps_clip": EPS_CLIP,
    "k_epoch": K_EPOCH,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "hidden_dim": HIDDEN_DIM,
    "max_episodes": MAX_EPISODES,
    "log_interval": LOG_INTERVAL,
    "model_save_path": MODEL_SAVE_PATH,
    "seed": SEED,
}
