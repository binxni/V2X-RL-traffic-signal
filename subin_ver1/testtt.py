# test_default_waiting_time.py
import pandas as pd
from env_sumo import SumoEnvironment

env = SumoEnvironment(...)  # config 설정
avg_wait_times = []

for episode in range(100):  # 테스트용 에피소드 수
    state = env.reset()
    done = False

    while not done:
        # 무작위 혹은 고정된 액션
        action = [0, 20]
        state, reward, done, _ = env.step(action)

    stats = env.get_stats()  # 기존 학습에서도 사용했을 함수
    avg_wait_times.append(stats["avg_waiting_time"])

# 결과 저장
pd.DataFrame({"avg_waiting_time": avg_wait_times}).to_csv(
    "default_waiting.csv", index=False)
