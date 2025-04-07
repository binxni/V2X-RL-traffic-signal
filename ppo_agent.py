import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

# 경험 저장용 튜플
Transition = namedtuple(
    "Transition", ["state", "action", "log_prob", "reward", "next_state", "done"])

# 신경망: 정책 (Actor)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 5)  # phase4개와 duration 1개
        )

    def forward(self, x):
        return self.fc(x)

# 신경망: 가치 함수 (Critic)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma=0.99, lr=3e-4, clip_eps=0.2, epochs=10):
        # 네트워크 초기화
        self.policy_net = PolicyNetwork(state_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

        # 하이퍼파라미터 설정
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs

        # phase 분포 기록용 로그 설정 추가
        self.phase_counter = Counter()
        self.writer = SummaryWriter(log_dir="runs/phase_distribution")
        self.memory = []  # Transition 저장 리스트
        self.total_steps = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 정책 네트워크의 출력 (4개 phase에 대한 logit, duration 하나)
        output = self.policy_net(state_tensor)
        phase_logits = output[:, :4]      # shape: [1, 4]
        duration_raw = output[:, 4]       # shape: [1]

        # Phase: Categorical 분포에서 샘플링
        phase_probs = torch.softmax(phase_logits, dim=-1)
        phase_dist = torch.distributions.Categorical(phase_probs)
        phase = phase_dist.sample().item()
        log_prob = phase_dist.log_prob(torch.tensor(phase))

        # ✅ phase 카운트 추가
        self.phase_counter[phase] += 1

        # Duration: 정규 분포 기반 샘플링 (혹은 tanh 등 활용)
        # duration = int(np.clip(duration_raw.item(), 5, 60))
        duration = int(5 + 55 * torch.sigmoid(duration_raw).item())

        return np.array([phase, duration]), log_prob

    def store_transition(self, *args):
        self.memory.append(Transition(*args))

    def compute_gae(self, next_state, done):
        # GAE 계산 함수
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
            returns.insert(0, gae + values[step])

        return returns, [r - v.item() for r, v in zip(returns, values[:-1])]

    def update(self, next_state, done):
        # 정책 및 가치 네트워크 업데이트 (PPO 핵심)
        transitions = self.memory
        returns, advantages = self.compute_gae(next_state, done)

        states = torch.FloatTensor([t.state for t in transitions])
        actions = torch.FloatTensor([t.action for t in transitions])
        old_log_probs = torch.stack([t.log_prob for t in transitions]).detach()
        returns = torch.FloatTensor(returns).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)

        for _ in range(self.epochs):
            # ✅ 정책 네트워크에서 [phase_logits, duration_pred] 분리
            policy_output = self.policy_net(states)  # shape: [batch, 2]
            phase_logits = policy_output[:, 0:1]     # Phase에 대한 logits
            duration_pred = policy_output[:, 1]      # Duration 예측값 (현재 사용 안함)

            # ✅ Categorical 분포에서 새 log_prob 계산 (phase)
            phase_probs = torch.softmax(phase_logits.squeeze(1), dim=0)
            dist = torch.distributions.Categorical(probs=phase_probs)
            new_log_probs = dist.log_prob(actions[:, 0].long()).unsqueeze(1)

            # ✅ PPO ratio 계산
            ratio = torch.exp(new_log_probs - old_log_probs)

            # ✅ Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # ✅ 엔트로피 보너스 추가
            entropy = dist.entropy().mean()  # 평균 엔트로피
            entropy_coeff = 0.05  # 탐험 비중 조절 하이퍼파라미터 (값이 크면 탐험 증가)
            policy_loss = policy_loss - entropy_coeff * entropy  # 엔트로피를 보너스로 추가

            # ✅ Value loss는 그대로
            values = self.value_net(states)
            value_loss = nn.MSELoss()(values, returns)

            # 🔄 최적화
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # ✅ phase 선택 분포 기록
            total = sum(self.phase_counter.values())
            if total > 0:
                for p in range(4):  # phase 0~3
                    ratio = self.phase_counter[p] / total
                    # or self.episode if 있음
                    self.writer.add_scalar("Phase/Selection_Ratio",
                                           ratio, self.total_steps)
            self.phase_counter.clear()  # 에피소드별 초기화

        self.memory = []
        return policy_loss.item(), value_loss.item()
