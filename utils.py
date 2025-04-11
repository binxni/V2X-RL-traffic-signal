from datetime import datetime
import csv
import numpy as np
import torch

# 일반적인 값을 tensor로 바꾸는 함수


def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)

# 학습 과정에서 advantage 계산 (GAE 사용 안 하는 기본 방식)


def compute_advantages(rewards, values, gamma=0.99):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * \
            (values[step + 1] if step + 1 < len(values) else 0) - values[step]
        gae = delta + gamma * gae
        advantages.insert(0, gae)
    return advantages

# 상태 normalization (선택사항)


def normalize(x):
    x = np.array(x)
    return (x - x.mean()) / (x.std() + 1e-8)

# 학습률 스케줄링 함수 (선택사항)


def linear_schedule(initial_value, final_value, current_step, total_steps):
    return final_value + (initial_value - final_value) * (1 - current_step / total_steps)

# 하이퍼파라미터 기록
def save_hyperparams_to_csv(config, entropy_coeff, filename="hyperparams_log.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "timestamp": now,
        "episode": config.get("episode", 0),
        "learning_rate": config.get("lr", None),
        "gamma": config.get("gamma", None),
        "lambda": config.get("lam", None),
        "entropy_coeff": entropy_coeff,
        "clip_ratio": config.get("clip_ratio", None),
        "batch_size": config.get("batch_size", None),
        "num_epochs": config.get("num_epochs", None),
        "hidden_sizes": str(config.get("hidden_sizes", None)),
        # 필요 시 추가 가능
    }

    fieldnames = list(data.keys())

    try:
        with open(filename, "x", newline='') as f:  # 파일 없으면 새로 생성
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
    except FileExistsError:
        with open(filename, "a", newline='') as f:  # 있으면 이어쓰기
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(data)
