# ppo_model.py
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        features = self.shared(state)
        return self.actor(features), self.critic(features)

from collections import deque
class PPO:
    
    def __init__(self, state_dim, action_dim):
        self.memory = deque(maxlen=10)
        self.gamma = 0.98
        self.eps_clip = 0.2
        self.lr = 2e-4
        
        self.policy = PPOActorCritic(state_dim, action_dim).cpu()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.old_policy = PPOActorCritic(state_dim, action_dim).cpu()
        self.old_policy.load_state_dict(self.policy.state_dict())
        
    def update(self, states, actions, rewards, dones):
        # Tensor 변환
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Advantage 계산
        with torch.no_grad():
            _, values = self.old_policy(states)  # values: [batch_size, 1]
            values = values.squeeze()  # [batch_size]
            advantages = rewards + self.gamma * (1 - dones) * values - values  # [batch_size]

        # PPO 손실 계산
        probs, _ = self.policy(states)  # probs: [batch_size, action_dim]
        old_probs, _ = self.old_policy(states)
        
        # 선택한 action에 대한 확률만 추출
        probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        ratio = probs / (old_probs.clamp(min=1e-8))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        loss = -torch.min(surr1, surr2).mean()

        # 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
