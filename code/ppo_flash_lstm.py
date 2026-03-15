import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

#------------------ 시뮬레이션 환경 ------------------
class FlashClickEnv(gym.Env):
    def __init__(self, T_true=1.0, dt=0.02, tol=0.15, max_steps=3000):
        super().__init__()
        self.T_true = T_true #플래시 간격
        self.dt = dt #타임스텝
        self.tol = tol
        self.max_steps = max_steps

        #관찰: [flash_on, time_since_last_flash, predicted_interval, vigilant_remaining]
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        #액션: click or no-click
        self.action_space = spaces.Discrete(2)

        #초기화
        self.reset()

    def reset(self):
        self.now = 0.0
        self.last_flash_time = 0.0
        self.next_flash_time = self.T_true
        self.vigilant_remaining = 0.0
        self.step_count = 0
        self.click_times = []
        self.flash_times = []
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        #normalized inputs
        flash_on = 1.0 if abs(self.now - self.next_flash_time) < self.dt/2 else 0.0
        time_since_last_flash = self.now - self.last_flash_time
        predicted_interval = self.T_true #placeholder, 에이전트 predictor가 필요시 변경
        vigilant = self.vigilant_remaining
        return np.array([flash_on, time_since_last_flash, predicted_interval, vigilant], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = 0.0

        #플래시 체크
        flash_on = False
        if self.now >= self.next_flash_time:
            flash_on = True
            self.last_flash_time = self.next_flash_time
            self.flash_times.append(self.next_flash_time)
            self.next_flash_time += self.T_true  #간단하게 일정 간격

        #motor delay 시뮬레이션
        motor_delay = np.random.normal(0.12, 0.03)
        if self.vigilant_remaining > 0:
            motor_delay *= 0.4  #vigilant 동안 오차 감소

        #클릭 처리
        if action == 1:  # click
            actual_click_time = self.now + motor_delay
            self.click_times.append(actual_click_time)

            #가장 가까운 플래시에 매칭
            if len(self.flash_times) > 0:
                closest_flash = min(self.flash_times, key=lambda t: abs(t - actual_click_time))
                delta = abs(closest_flash - actual_click_time)
                if delta <= self.tol:
                    reward = max(0.0, 1 - delta/self.tol) #정확성 보상
                else:
                    reward = -0.2  #false alarm
            else:
                reward = -0.2  #no flash yet

            reward -= 0.01  #click cost

        #vigilant 상태 갱신
        if (self.next_flash_time - self.now) <= 0.0: #predicted interval 도달
            self.vigilant_remaining = 1.0
        if self.vigilant_remaining > 0:
            self.vigilant_remaining -= self.dt

        self.now += self.dt
        obs = self._get_obs()
        return obs, reward, done, {}

#------------------ LSTM 정책 네트워크 ------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_layers=1, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x.unsqueeze(0), hidden) #seq_len=1
        x = x.squeeze(0)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value, hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, 256)
        c = torch.zeros(1, batch_size, 256)
        return (h, c)

#------------------ PPO 학습 루프(미완) ------------------
env = FlashClickEnv()
policy = LSTMPolicy()
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
gamma = 0.99
eps_clip = 0.2
K_epochs = 4

# 간단 single episode 학습 예시
for episode in range(100):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    hidden = policy.init_hidden()
    log_probs = []
    values = []
    rewards = []

    done = False
    while not done:
        logits, value, hidden = policy(obs.unsqueeze(0), hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        obs_next, reward, done, _ = env.step(action.item())
        obs_next = torch.tensor(obs_next, dtype=torch.float32)

        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards.append(reward)

        obs = obs_next

    # Compute discounted returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)

    # Advantage
    advantage = returns - values.detach()

    # PPO loss
    for _ in range(K_epochs):
        ratios = torch.exp(log_probs - log_probs.detach())
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-values)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}, total_reward: {sum(rewards):.2f}")

