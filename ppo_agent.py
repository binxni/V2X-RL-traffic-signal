import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

# 경험 저장용 튜플
Transition = namedtuple(
    "Transition", ["state", "action", "log_prob", "reward", "next_state", "done"])

# 신경망: 정책 (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# 신경망: 가치 함수 (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma=0.99, lr=3e-4, clip_eps=0.2, epochs=10):
        # 네트워크 초화
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

        # 하이퍼라미터 설정
        self.gamma = gamma  # discount factor
        self.clip_eps = clip_eps  # 클리핑 범위
        self.epochs = epochs  # 업데이트 반복 횟수

        self.memory = []  # Transition(경험) 저장 리스트

    def select_action(self, state):
        # 주어진 상태에서 확률적으로 행동 선택
        state = torch.FloatTensor(state).unsqueeze(0)  # 배치 차원 추가
        probs = self.policy_net(state)                 # 행동 확률 분포
        # state를 넣어 policy 확률 얻음
        dist = torch.distributions.Categorical(probs)  # 범주형 분포
        action = dist.sample()                         # action 샘플링
        return action.item(), dist.log_prob(action)    # 행동과 그 로그 확률 반환

    def store_transition(self, *args):
        # 경험 저장 함수 -> update()에 사용됨
        self.memory.append(Transition(*args))

# GAE 안쓴 버전
    '''def compute_returns(self, next_state, done):
        # episode 종료 후 GAE 없이 단순한 cumulative reward 계산(discounted reward)
        returns = []
        R = 0 if done else self.value_net(torch.FloatTensor(next_state)).item()
        for transition in reversed(self.memory):
            R = transition.reward + self.gamma * R
            returns.insert(0, R)
        return returns'''


    
    def compute_gae(self, next_state, done):
        
        values = self.value_net(torch.FloatTensor(
            [t.state for t in self.memory])).detach().squeeze()
        next_value = self.value_net(torch.FloatTensor(
            next_state)).detach() if not done else torch.tensor(0.0)
        rewards = [t.reward for t in self.memory]
        masks = [0.0 if t.done else 1.0 for t in self.memory]

        values = torch.cat([values, next_value.unsqueeze(0)])
        gae = 0
        returns = []

        LAMBDA = 0.95 
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * \
                values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * LAMBDA * masks[step] * gae
            returns.insert(0, gae + values[step])  # Advantage + baseline

        return returns, [r - v.item() for r, v in zip(returns, values[:-1])]
    

    def update(self, next_state, done):
        # PPO 핵심 학습 함수 : policy와 value 네트워크 모두 업데이트
        transitions = self.memory
        returns, advantages = self.compute_gae(next_state, done)
        # 마지막 state부터 시작해 backward로 누적 return 계산
        # return = r + v*next_return

        states = torch.FloatTensor([t.state for t in transitions])
        actions = torch.LongTensor(
            [t.action for t in transitions]).unsqueeze(1)
        old_log_probs = torch.stack([t.log_prob for t in transitions]).detach()
        returns = torch.FloatTensor(returns).unsqueeze(1)

        for _ in range(self.epochs):
            # 현재 정책으로 다시 log_prob 계산
            probs = self.policy_net(states)                     # 현재 정책 확률
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions.squeeze())        # log π_θ(a|s)
            entropy = dist.entropy().mean()                     

            # 비율 계산 (r_t(theta))
            ratio = torch.exp(log_probs - old_log_probs)        # 정책 비율

            # Advantage 계산
            values = self.value_net(states)                     # V(s)
            #advantage = returns - values.detach()               # A_t = R_t - V(s_t) (GAE 미적용)
            advantage = torch.FloatTensor(advantages).unsqueeze(1) #GAE 적용

            # PPO Loss 계산 (Clipped Surrogate Objective)
            surrogate1 = ratio * advantage      
            surrogate2 = torch.clamp(
                ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
            policy_loss = -(torch.min(surrogate1, surrogate2) +
                            0.01 * entropy).mean()

            value_loss = nn.MSELoss()(values, returns)

            # 네트워크 업데이트
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        # 메모리 초기화 후 손실 반환 (로깅용)
        self.memory = []  # 메모리 비움
        return policy_loss.item(), value_loss.item()
