import torch
from ppo_agent import PolicyNetwork, ValueNetwork  # 당신이 정의한 모델들
from ppo_agent import PPOAgent
import torch
from main import agent


# 모델 정보
state_dim = 27
hidden_dim = 128

# 모델 인스턴스 생성
policy_net = PolicyNetwork(state_dim, hidden_dim)
value_net = ValueNetwork(state_dim, hidden_dim)

# 저장된 checkpoint 로드
checkpoint = torch.load("/home/subin/Downloads/V2X-RL-traffic-signal-master/models_backup/ppo_sumo_ep001_20250414_1943.pth", map_location=torch.device('cpu'))

agent.policy_net.load_state_dict(checkpoint['policy'])
agent.value_net.load_state_dict(checkpoint['value'])
agent.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
agent.optimizer_value.load_state_dict(checkpoint['optimizer_value'])

episode = checkpoint['episode']
avg_waiting_time = checkpoint['avg_waiting_time']
# 모델에 파라미터 주입
policy_net.load_state_dict(checkpoint['policy'])
value_net.load_state_dict(checkpoint['value'])

# 확인
policy_net.eval()
value_net.eval()

# 예시 입력값 (state_dim=27일 때)
dummy_input = torch.rand(1, state_dim)
with torch.no_grad():
    policy_output = policy_net(dummy_input)
    value_output = value_net(dummy_input)

print("Policy output:", policy_output)
print("Value output:", value_output)