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
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

        self.gamma = gamma  # discount factor
        self.clip_eps = clip_eps  # 클리핑 범위
        self.epochs = epochs  # 업데이트 반복 횟수

        self.memory = []  # Transition 저장용

    def select_action(self, state):
        # 주어진 상태에서 확률적으로 행동 선택
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, *args):
        self.memory.append(Transition(*args))

    def compute_returns(self, next_state, done):
        # GAE 없이 단순한 cumulative reward 계산
        returns = []
        R = 0 if done else self.value_net(torch.FloatTensor(next_state)).item()
        for transition in reversed(self.memory):
            R = transition.reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, next_state, done):
        transitions = self.memory
        returns = self.compute_returns(next_state, done)

        states = torch.FloatTensor([t.state for t in transitions])
        actions = torch.LongTensor(
            [t.action for t in transitions]).unsqueeze(1)
        old_log_probs = torch.stack([t.log_prob for t in transitions]).detach()
        returns = torch.FloatTensor(returns).unsqueeze(1)

        for _ in range(self.epochs):
            # 현재 정책으로 다시 log_prob 계산
            probs = self.policy_net(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy().mean()

            # 비율 계산 (r_t(theta))
            ratio = torch.exp(log_probs - old_log_probs)

            # Advantage 계산
            values = self.value_net(states)
            advantage = returns - values.detach()

            # PPO Loss 계산 (Clipped Surrogate Objective)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(
                ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(values, returns)

            # 네트워크 업데이트
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()


        self.memory = []  # 메모리 비움
        return policy_loss.item(), value_loss.item()
