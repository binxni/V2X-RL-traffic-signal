# main.py (테스트용 코드)
from env_sumo import SumoEnvironment
from config import config
import time  # 맨 위에 추가
import traci

env = SumoEnvironment(config)

for _ in range(100):  # 100 스텝 정도만
    action = 0  # 임의로 신호 phase 0번으로 고정
    state, reward, done, _ = env.step(action)
    print("State:", state)
    print("Reward:", reward)
    if done:
        break

env.close()


def step(self, action):
    self._apply_action(action)
    traci.simulationStep()
    time.sleep(0.5)  # 0.2초 멈추기 (조절 가능)
