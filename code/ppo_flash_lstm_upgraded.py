# internal clock을 학습 기반으로 바꾸고 ppo 정책을 갈아엎었음..

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import csv

# 환경
class FlashClickEnv(gym.Env):
    def __init__(self, T_true=3.0, dt=0.02, tol=0.15, max_steps=3000):
        super().__init__()
        self.T_true = T_true
        self.dt = dt
        self.tol = tol
        self.max_steps = max_steps

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.reset()

        # 매칭 허용 윈도우 (초). 클릭이 이 윈도우 밖이면 매칭 안 함 (false alarm)
        self.match_window = self.match_window = self.T_true / 2.0

    def reset(self):
        self.now = 0.0
        self.last_flash_time = 0.0
        self.next_flash_time = self.T_true
        self.vigilant_remaining = 0.0
        self.step_count = 0
        self.click_times = []
        self.flash_times = []
        self.click_to_flash_map = []  # 리스트 of tuples (flash_time, click_time)
        self.internal_clock_remaining = None
        self.predicted_interval = None
        obs = self._get_obs()
        return obs

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
        """
        클릭이 발생한 순간에 가능한 플래시 후보들(past flash_times + next_flash_time)을 보고
        match_window 내에서 가장 가까운 플래시를 찾아 매핑한다.
        매핑이 없으면 None 반환 (false alarm)
        """
        candidates = list(self.flash_times)  # 과거 플래시들
        # 아직 기록되지 않았지만 곧 올 다음 플래시도 후보로 포함
        candidates.append(self.next_flash_time)
        if len(candidates) == 0:
            return None
        # compute absolute diffs
        diffs = [abs(actual_click_time - ft) for ft in candidates]
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= self.match_window:
            return candidates[min_idx]
        else:
            return None

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = 0.0

        # Flash 발생 체크
        flash_on = False
        if self.now >= self.next_flash_time:
            flash_on = True
            self.last_flash_time = self.next_flash_time
            self.flash_times.append(self.next_flash_time)
            self.next_flash_time += self.T_true

        motor_delay = np.random.normal(0.12, 0.03)
        if self.vigilant_remaining > 0:
            motor_delay *= 0.4

        if int(action) == 1:
            actual_click_time = self.now + motor_delay
            self.click_times.append(actual_click_time)

            # 클릭이 어떤 플래시를 겨냥했는지 즉시 매칭해서 저장
            matched_flash = self._match_click(actual_click_time)
            if matched_flash is not None:
                # 매칭된 플래시에 대해 reward 계산
                delta = abs(matched_flash - actual_click_time)
                if delta <= self.tol:
                    reward = max(0.0, 1.0 - delta / self.tol)
                else:
                    reward = -0.5
                # 기록: (flash_time, click_time)
                self.click_to_flash_map.append((matched_flash, actual_click_time))
            else:
                reward = -0.2  # false alarm
                # 기록: (None, click_time) - optional
                self.click_to_flash_map.append((None, actual_click_time))

            reward -= 0.01

        # internal clock countdown 및 vigilant 트리거
        if self.internal_clock_remaining is not None:
            self.internal_clock_remaining -= self.dt
            if self.internal_clock_remaining <= 0 and self.vigilant_remaining <= 0.0:
                self.vigilant_remaining = 1.0

        if self.vigilant_remaining > 0:
            self.vigilant_remaining -= self.dt
            if self.vigilant_remaining < 0:
                self.vigilant_remaining = 0.0

        self.now += self.dt
        obs = self._get_obs()
        info = {"true_interval": self.T_true, "predicted_interval": self.predicted_interval}
        return obs, reward, done, info

# ------------------ LSTM 정책 네트워크 ------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=1, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        # 추가: interval predictor (양의 값 예측)
        self.interval_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x, hidden):
        """
        x: [batch, input_dim] float tensor
        hidden: (h, c) with shapes (num_layers, batch, hidden_dim)
        returns: logits [batch, action_dim], value [batch,1], pred_interval [batch,1], hidden
        """
        x = torch.relu(self.fc1(x))
        x_seq = x.unsqueeze(0)  # seq_len=1, LSTM expects (seq, batch, feature)
        out, hidden = self.lstm(x_seq, hidden)
        out = out.squeeze(0)  # [batch, hidden_dim]
        logits = self.actor(out)
        value = self.critic(out)
        pred_raw = self.interval_head(out)
        pred_interval = self.softplus(pred_raw)  # 양수 보장
        return logits, value.squeeze(-1), pred_interval.squeeze(-1), hidden

    def init_hidden(self, batch_size=1, device=None):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if device is not None:
            h = h.to(device); c = c.to(device)
        return (h, c)

# ------------------ PPO 학습 루프 ------------------
def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlashClickEnv(T_true=1.0, dt=0.02, tol=0.15, max_steps=1000)
    policy = LSTMPolicy(input_dim=3, hidden_dim=256, num_layers=1, action_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    gamma = 0.99
    eps_clip = 0.2
    K_epochs = 4
    max_episodes = 50

    for episode in range(max_episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        hidden = policy.init_hidden(batch_size=1, device=device)

        # rollout buffers
        obs_buf = []
        actions_buf = []
        old_log_probs_buf = []
        values_buf = []
        rewards_buf = []
        pred_intervals_buf = []
        hidden_states_buf = []  # store hidden (h,c) BEFORE stepping (for re-eval)

        done = False
        total_reward = 0.0
        while not done:
            # store hidden state (detach)
            h_detach = tuple([h.detach().cpu() for h in hidden])
            hidden_states_buf.append(h_detach)

            logits, value, pred_interval, hidden = policy(obs_t.unsqueeze(0), hidden)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device)).detach().cpu().numpy().item()

            # agent의 예측 간격을 env에 알려줌
            env.set_predicted_interval(pred_interval.detach().cpu().numpy().item())

            obs_next, reward, done, info = env.step(action)
            total_reward += reward

            # store rollout data
            obs_buf.append(obs_t.cpu().numpy())
            actions_buf.append(action)
            old_log_probs_buf.append(log_prob)
            values_buf.append(value.detach().cpu().numpy().item())
            rewards_buf.append(reward)
            pred_intervals_buf.append(pred_interval.detach().cpu().numpy().item())

            # next step
            obs_t = torch.tensor(obs_next, dtype=torch.float32).to(device)

            # safety: if episode too long break (env.done handles)
        # end rollout

        # compute returns & advantages
        returns = compute_returns(rewards_buf, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.tensor(values_buf, dtype=torch.float32).to(device)
        advantages = returns_t - values_t

        # convert buffers to tensors
        obs_tensor = torch.tensor(np.stack(obs_buf, axis=0), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions_buf, dtype=torch.long).to(device)
        old_log_probs_tensor = torch.tensor(old_log_probs_buf, dtype=torch.float32).to(device)
        pred_intervals_tensor = torch.tensor(pred_intervals_buf, dtype=torch.float32).to(device)


        # PPO update: K_epochs passes over the whole trajectory
        for epoch in range(K_epochs):

            hidden_re = policy.init_hidden(batch_size=1, device=device)
            new_log_probs = []
            new_values = []
            new_pred_intervals = []

            for t in range(obs_tensor.shape[0]):
                ot = obs_tensor[t].unsqueeze(0)  # [1, input_dim]
                logits, value, pred_interval, hidden_re = policy(ot, hidden_re)
                dist = Categorical(logits=logits.squeeze(0))
                lp = dist.log_prob(actions_tensor[t]).unsqueeze(0)  # shape [1]
                new_log_probs.append(lp.squeeze(0))
                new_values.append(value)
                new_pred_intervals.append(pred_interval)

            new_log_probs = torch.stack(new_log_probs)  # [T]
            new_values = torch.stack(new_values).squeeze(-1)  # [T]
            new_pred_intervals = torch.stack(new_pred_intervals).squeeze(-1)  # [T]

            # ratios
            ratios = torch.exp(new_log_probs - old_log_probs_tensor)

            # surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_t - new_values).pow(2).mean()

            target_intervals = torch.tensor([env.T_true] * new_pred_intervals.shape[0], dtype=torch.float32).to(device)
            aux_loss = nn.MSELoss()(new_pred_intervals, target_intervals)

            total_loss = actor_loss + critic_loss + 1.0 * aux_loss  # weight of aux_loss can be tuned

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, total_reward: {total_reward:.3f}, avg_return: {returns_t.mean().item():.3f}")
            
    # 학습 끝난 뒤 예시 rollout에서 데이터 저장
    obs = env.reset()
    hidden = policy.init_hidden(batch_size=1, device=device)
    done = False
    while not done:
        logits, value, pred_interval, hidden = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device), hidden)
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample().item()
        env.set_predicted_interval(pred_interval.detach().cpu().item())
        obs, reward, done, info = env.step(action)

    # 저장: flash 당 매칭된 클릭을 기록
    flash_to_click = {}
    for ft in env.flash_times:
        flash_to_click[ft] = ""  # default empty

    # assign matches from recorded map
    for (ft, ct) in env.click_to_flash_map:
        if ft is None:
            continue
        # if multiple clicks match same flash, keep the earliest click (or choose policy)
        if flash_to_click.get(ft, "") == "":
            flash_to_click[ft] = ct

    # write CSV
    with open("model_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for ft in sorted(env.flash_times):
            writer.writerow([ft, flash_to_click.get(ft, "")])

    print("Model data saved to model_data.csv (fixed matching logic)")

if __name__ == "__main__":
    train()
