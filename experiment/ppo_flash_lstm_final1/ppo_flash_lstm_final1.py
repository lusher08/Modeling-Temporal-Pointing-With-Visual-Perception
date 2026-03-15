import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import csv

# ------------------ 시뮬레이션 환경 ------------------
class FlashClickEnv(gym.Env):
    def __init__(self, T_true=1.0, dt=0.02, tol=0.15, max_steps=3000):
        super().__init__()
        self.T_true = T_true
        self.dt = dt
        self.tol = tol
        self.max_steps = max_steps

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # 더 자주 매칭되도록 윈도우 확대
        self.match_window = self.T_true * 1.0

        # right-skewed 형태의 그래프가 되도록 조정중..
        self.motor_mu = 0.15
        self.motor_sigma = 0.05

        # 탐색 보장 (무조건 클릭을 해야하도록)
        self.min_click_prob = 0.05

        self.reset()

    def reset(self):
        self.now = 0.0
        self.last_flash_time = 0.0
        self.next_flash_time = self.T_true
        self.vigilant_remaining = 0.0
        self.step_count = 0
        self.click_times = []
        self.flash_times = []
        self.click_to_flash_map = []
        self.internal_clock_remaining = None
        self.predicted_interval = None
        return self._get_obs()

    def _get_obs(self):
        flash_on = 1.0 if abs(self.now - self.next_flash_time) < self.dt/2 else 0.0
        time_since_last_flash = self.now - self.last_flash_time
        vigilant = self.vigilant_remaining
        return np.array([flash_on, time_since_last_flash, vigilant], dtype=np.float32)

    def set_predicted_interval(self, pred_interval):
        pred_interval = float(np.clip(pred_interval, 0.05, 10.0))
        self.predicted_interval = pred_interval
        time_since_last = self.now - self.last_flash_time
        remaining = pred_interval - time_since_last
        self.internal_clock_remaining = max(0.0, remaining)

    def _match_click(self, actual_click_time):
        candidates = list(self.flash_times)
        candidates.append(self.next_flash_time)
        if len(candidates) == 0:
            return None
        diffs = [abs(actual_click_time - ft) for ft in candidates]
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= self.match_window:
            return candidates[min_idx]
        return None

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = 0.0

        # Flash 발생
        if self.now >= self.next_flash_time:
            self.last_flash_time = self.next_flash_time
            self.flash_times.append(self.next_flash_time)
            self.next_flash_time += self.T_true

        # motor delay (right-skewed 그래프)
        motor_delay = np.random.normal(self.motor_mu, self.motor_sigma)
        motor_delay = max(motor_delay, 0.0)
        if self.vigilant_remaining > 0:
            motor_delay *= 0.6

        # 최소 클릭 확률 보장
        if np.random.rand() < self.min_click_prob:
            action = 1

        if int(action) == 1:
            actual_click_time = self.now + motor_delay
            self.click_times.append(actual_click_time)
            matched_flash = self._match_click(actual_click_time)

            if matched_flash is not None:
                delta = actual_click_time - matched_flash
                if delta < 0:
                    reward = -0.6 - 0.4 * abs(delta)
                else:
                    tau = 0.25
                    reward = np.exp(-delta / tau)
                self.click_to_flash_map.append((matched_flash, actual_click_time))
            else:
                reward = -0.2
                self.click_to_flash_map.append((None, actual_click_time))

            reward -= 0.001

        # vigilant timer
        if self.internal_clock_remaining is not None:
            self.internal_clock_remaining -= self.dt
            if self.internal_clock_remaining <= 0 and self.vigilant_remaining <= 0:
                self.vigilant_remaining = 1.0
        if self.vigilant_remaining > 0:
            self.vigilant_remaining -= self.dt
            if self.vigilant_remaining < 0:
                self.vigilant_remaining = 0.0

        self.now += self.dt
        return self._get_obs(), reward, done, {}

# ------------------ 정책 ------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.interval_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x, hidden):
        x = torch.relu(self.fc1(x))
        x_seq = x.unsqueeze(0)
        out, hidden = self.lstm(x_seq, hidden)
        out = out.squeeze(0)
        logits = self.actor(out)
        value = self.critic(out)
        pred_raw = self.interval_head(out)
        pred_interval = self.softplus(pred_raw)
        return logits, value.squeeze(-1), pred_interval.squeeze(-1), hidden

    def init_hidden(self, batch_size=1, device=None):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if device is not None:
            h, c = h.to(device), c.to(device)
        return (h, c)

# ------------------ PPO 학습 ------------------
def compute_returns(rewards, gamma=0.99):
    returns, R = [], 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlashClickEnv(T_true=1.0, max_steps=3000)
    policy = LSTMPolicy(hidden_dim=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    gamma, eps_clip, K_epochs, max_episodes = 0.99, 0.2, 3, 50

    for episode in range(max_episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        hidden = policy.init_hidden(device=device)

        obs_buf, actions_buf, old_log_probs_buf, values_buf, rewards_buf = [], [], [], [], []

        done, total_reward = False, 0.0
        while not done:
            logits, value, pred_interval, hidden = policy(obs_t.unsqueeze(0), hidden)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device)).detach().cpu().numpy().item()
            env.set_predicted_interval(pred_interval.detach().cpu().item())
            obs_next, reward, done, _ = env.step(action)
            total_reward += reward

            obs_buf.append(obs_t.cpu().numpy())
            actions_buf.append(action)
            old_log_probs_buf.append(log_prob)
            values_buf.append(value.detach().cpu().numpy().item())
            rewards_buf.append(reward)

            obs_t = torch.tensor(obs_next, dtype=torch.float32).to(device)

        returns = compute_returns(rewards_buf, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.tensor(values_buf, dtype=torch.float32).to(device)
        advantages = returns_t - values_t

        obs_tensor = torch.tensor(np.stack(obs_buf), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions_buf, dtype=torch.long).to(device)
        old_log_probs_tensor = torch.tensor(old_log_probs_buf, dtype=torch.float32).to(device)

        for _ in range(K_epochs):
            hidden_re = policy.init_hidden(device=device)
            new_log_probs, new_values = [], []
            for t in range(obs_tensor.shape[0]):
                ot = obs_tensor[t].unsqueeze(0)
                logits, value, _, hidden_re = policy(ot, hidden_re)
                dist = Categorical(logits=logits.squeeze(0))
                lp = dist.log_prob(actions_tensor[t])
                new_log_probs.append(lp)
                new_values.append(value)
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze(-1)
            ratios = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_t - new_values).pow(2).mean()
            total_loss = actor_loss + critic_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        # 에피소드마다 데이터 저장
        flash_to_click = {ft: "" for ft in env.flash_times}
        for (ft, ct) in env.click_to_flash_map:
            if ft is None: continue
            if flash_to_click.get(ft, "") == "":
                flash_to_click[ft] = ct
        with open("model_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for ft in sorted(env.flash_times):
                writer.writerow([ft, flash_to_click.get(ft, "")])

        print(f"Episode {episode:03d} finished, reward={total_reward:.2f}, flashes={len(env.flash_times)}")

    print("All episodes done. model_data.csv generated with aggregated data.")

if __name__ == "__main__":
    train()
