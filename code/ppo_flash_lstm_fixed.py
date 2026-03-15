import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import csv
import os

#======================시뮬레이션 환경===========================
class FlashClickEnv(gym.Env):
    """
    플래시가 일정간격으로 뜨고, 에이전트가 click/no-click을 선택하면
    motor delay를 적용해 에이전트의 실제 클릭을 기록,
    이후 클릭과 플래시를 매칭하여(윈도우의 1/2기준) 보상을 계산한다
    """
    def __init__(self, T_true=3.0, dt=0.02, tol=0.15, max_steps=3000):
        super().__init__()
        self.T_true = T_true #실제 플래시 간격(초)
        self.dt = dt #시뮬레이션 타임스텝(시각적으로 자연스럽다의 최소 기준인 50Hz가 기본)
        self.tol = tol #클릭이 플래시에 얼마나 가까워야 정확한지 허용 반경(초)
        self.max_steps = max_steps #에피소드당 최대 타임스텝 수

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 1.0], dtype=np.float32)
        """
        <관찰 공간>
        플래시가 켜졌는지, 직전 플래시로부터 얼마나 시간이 지났는지, 바짝대기 상태인지 체크한다
        """
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        """
        <액션 공간>
        0: no_click / 1: click
        """
        self.action_space = spaces.Discrete(2) 
        self.match_window = self.T_true / 2.0 #클릭-플래시 매칭 범위(간격의 절반 정도)
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
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """
        3차원 벡터로 이루어진 에이전트의 관찰(입력) 시점
        """
        #현재 타임스텝에서 플래시가 켜져있으면 1, 아니면 0
        flash_on = 1.0 if abs(self.now - self.next_flash_time) < self.dt/2 else 0.0
        #직전 플래시 이후 경과한 시간(초)
        time_since_last_flash = self.now - self.last_flash_time
        #바짝대기 잔여시간(0이면 비활성화된다)
        vigilant = self.vigilant_remaining
        return np.array([flash_on, time_since_last_flash, vigilant], dtype=np.float32)

    def set_predicted_interval(self, pred_interval):
        """
        에이전트가 예측한 다음 플래시까지의 간격(internal clock)을 환경에 전달,
        환경은 이를 카운트다운하며 바짝대기 상태 트리거에 사용한다
        """
        pred_interval = float(np.clip(pred_interval, 0.05, 10.0)) #범위 제한
        #예측값 기록
        self.predicted_interval = pred_interval
        #직전 플래시 이후 경과된 시간 측정
        time_since_last = self.now - self.last_flash_time
        #예측값에 도달하기까지 얼마나 시간이 남았는지
        remaining = pred_interval - time_since_last
        self.internal_clock_remaining = max(0.0, remaining) #범위 보정

    def _match_click(self, actual_click_time):
        """
        에이전트가 실제로 클릭했을 때(현재 시각+motor delay)
        클릭을 플래시와 매핑하기 위해 플래시 중 제일 가까운 것을 찾아 반환하고,
        매칭 거리가 허용된 윈도우보다 클 경우 클릭이 없었다고 간주한다
        """
        #과거 플래시 리스트
        candidates = list(self.flash_times)
        #아직 기록되지 않았지만 곧 나타날 플래시도 후보에 포함시킨다
        candidates.append(self.next_flash_time)
        #만약 후보가 없다면 매핑하지 않는다
        if len(candidates) == 0:
            return None
        #윈도우 범위 안의 후보 중 제일 적은 시간차를 보이는 후보를 반환한다
        diffs = [abs(actual_click_time - ft) for ft in candidates]
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= self.match_window:
            return candidates[min_idx]
        else:
            return None

    def step(self, action):
        """
        환경의 한 타임스텝을 진행한다 (플래시 생성, motor delay,
        클릭-플래시 매칭,보상 계산, vigilant/clock 로직, 시간이동)
        """
        #스텝 수가 증가
        self.step_count += 1
        #에피소드 종료 여부 검사
        done = self.step_count >= self.max_steps
        #보상 초기화
        reward = 0.0

        #플래시 발생 체크
        flash_on = False
        if self.now >= self.next_flash_time - 1e-12:
            flash_on = True
            self.last_flash_time = self.next_flash_time
            self.flash_times.append(self.next_flash_time)
            self.next_flash_time += self.T_true

        #motor delay 샘플링
        motor_delay = float(np.random.normal(0.12, 0.03))
        #바짝대기 상태일 경우 정확도 향상(motor delay 분산 감소)
        if self.vigilant_remaining > 0:
            motor_delay *= 0.4

        #클릭 여부
        clicked = False
        if int(action) == 1:
            clicked = True
            #클릭 시간 = 의사결정 시간 + motor delay
            actual_click_time = self.now + motor_delay
            self.click_times.append(actual_click_time)

            #클릭 시간과 플래시 매칭
            matched_flash = self._match_click(actual_click_time)
            if matched_flash is not None:
                delta = abs(matched_flash - actual_click_time)
                if delta <= self.tol:
                    reward = max(0.0, 1.0 - delta / self.tol)
                else:
                    reward = -0.5
                self.click_to_flash_map.append((matched_flash, actual_click_time))
            else:
                reward = -0.2
                self.click_to_flash_map.append((None, actual_click_time))

            reward -= 0.01  # click cost

        #클릭하지 않으면 약간의 페널티를 준다
        if not clicked:
            reward -= 0.001

        #internal clock 카운트다운 및 viligant 트리거
        if self.internal_clock_remaining is not None:
            self.internal_clock_remaining -= self.dt
            #내부 시계가 닳으면 바짝대기 상태를 1초 동안 유지한다
            if self.internal_clock_remaining <= 0 and self.vigilant_remaining <= 0.0:
                self.vigilant_remaining = 1.0

        if self.vigilant_remaining > 0:
            self.vigilant_remaining -= self.dt
            if self.vigilant_remaining < 0:
                self.vigilant_remaining = 0.0

        #시간 이동
        self.now += self.dt
        #이동한 시간대의 관찰을 생성한다
        obs = self._get_obs()
        info = {"true_interval": self.T_true, "predicted_interval": self.predicted_interval}
        return obs, reward, done, info
    
#======================LSTM 네트워크===========================
class LSTMPolicy(nn.Module):
    """
    관찰을 받아서 '1.행동 2.상태 가치 3.다음 플래시 간격 예측'을 반환한다
    """
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=1, action_dim=2):
        super().__init__() #입력 선형변환
        self.hidden_dim = hidden_dim #시퀀스(메모리)를 위한 LSTM
        self.num_layers = num_layers #정책 logits 생성(2차원)

        self.fc1 = nn.Linear(input_dim, 128) #상태가치 추정
        self.lstm = nn.LSTM(128, hidden_dim, num_layers) 
        self.actor = nn.Linear(hidden_dim, action_dim) 
        self.critic = nn.Linear(hidden_dim, 1)
        self.interval_head = nn.Linear(hidden_dim, 1) #interval 예측을 위한 헤드
        self.softplus = nn.Softplus() #interval을 양수에 제한

    def forward(self, x, hidden):
        """
        한 스텝의 관찰 x와 LSTM hidden을 받아
        logits, value, pred_interval, new_hidden을 반환한다
        """
        x = torch.relu(self.fc1(x))
        #LSTM 입력은 [seq_len, batch, features] 형태이므로 seq_len=1로 확장한다
        x_seq = x.unsqueeze(0)
        #LSTM 통과(다음 hidden 반환)
        out, hidden = self.lstm(x_seq, hidden)
        #seq_len 차원 제거 -> [batch, hidden_dim]
        out = out.squeeze(0)
        logits = self.actor(out) #행동 logits
        value = self.critic(out) #상태 가치(tensor)
        pred_raw = self.interval_head(out) #interval 값
        pred_interval = self.softplus(pred_raw) #interval 양수로 변환
        return logits, value.squeeze(-1), pred_interval.squeeze(-1), hidden

    def init_hidden(self, batch_size=1, device=None):
        """
        LSTM의 초기 hidden(h,c)를 0으로 생성하여 반환한다
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if device is not None: #device가 주어지는 경우 .to(device)로 이동한다
            h = h.to(device); c = c.to(device)
        return (h, c)

#======================PPO 학습 루프===========================
def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    """
    환경 생성, 정책 초기화, 에피소드별 롤아웃 수집, ppo 업데이트, 학습 종료 후 최종 롤아웃 저장
    (classic PPO에 interval predictor MSE와 entropy를 추가함)
    """
    #초기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlashClickEnv()
    policy = LSTMPolicy(input_dim=3, hidden_dim=256, num_layers=1, action_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    #하이퍼파라미터
    gamma = 0.99
    eps_clip = 0.2
    K_epochs = 4
    max_episodes = 20

    entropy_coef = 0.05 #정책 엔트로피 보너스로 exploration을 유도한다
    aux_loss_weight = 1.0 #interval 예측을 보조하는 손실 가중치

    #에피소드 루프
    for episode in range(max_episodes):
        obs = env.reset()
        #현재 관찰 텐서화
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        #LSTM hidden 초기화
        hidden = policy.init_hidden(batch_size=1, device=device)

        #버퍼 생성
        obs_buf = []
        actions_buf = []
        old_log_probs_buf = []
        values_buf = []
        rewards_buf = []
        pred_intervals_buf = []

        done = False
        total_reward = 0.0

        #타임스텝 내부
        while not done:
            #obs_t.unsqueeze(0)로 batch dimension을 맞춘다([batch=1,input_dim])
            logits, value, pred_interval, hidden = policy(obs_t.unsqueeze(0), hidden)
            dist = Categorical(logits=logits.squeeze(0)) #logits에서 분포 생성
            action = dist.sample().item() #클릭 여부(0 또는 1)
            #로그확률을 계산한다
            log_prob = dist.log_prob(torch.tensor(action, device=device))

            #정책이 내린 간격 예측 결과를 환경에 전달한다(internal clock 형성)
            env.set_predicted_interval(pred_interval.detach().cpu().numpy().item())
            #환경에서 한 스텝 진행
            obs_next, reward, done, info = env.step(action)
            total_reward += reward

            #버퍼에 각종 값 저장
            obs_buf.append(obs_t.cpu().numpy())
            actions_buf.append(action)
            #IMPORTANT# .item()을 사용하여 float로 버퍼에 저장해야 오류 안남
            old_log_probs_buf.append(log_prob.item())
            values_buf.append(value.detach().cpu().numpy().item())
            rewards_buf.append(reward)
            pred_intervals_buf.append(pred_interval.detach().cpu().numpy().item())

            #다음 스텝
            obs_t = torch.tensor(obs_next, dtype=torch.float32).to(device)

        #return/advantage 계산
        returns = compute_returns(rewards_buf, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.tensor(values_buf, dtype=torch.float32).to(device)
        advantages = returns_t - values_t

        #텐서 변환(버퍼)
        obs_tensor = torch.tensor(np.stack(obs_buf, axis=0), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions_buf, dtype=torch.long).to(device)
        old_log_probs_tensor = torch.tensor(old_log_probs_buf, dtype=torch.float32).to(device)
        pred_intervals_tensor = torch.tensor(pred_intervals_buf, dtype=torch.float32).to(device)

        #PPO 업데이트
        """
        현재 정책으로 재평가한 new_log_probs, new_values, new_pred_intervals, entropies를
        얻어 surrogate loss를 계산 및 최적화한다
        """
        for epoch in range(K_epochs):
            #평가할 때 LSTM hidden을 초기화한다
            hidden_re = policy.init_hidden(batch_size=1, device=device)
            #빈 리스트 생성
            new_log_probs = []
            new_values = []
            new_pred_intervals = []
            entropies = []

            #각 타임의 obs를 재투입한다
            for t in range(obs_tensor.shape[0]):
                ot = obs_tensor[t].unsqueeze(0)
                #재평가 (hidden_re를 연속적으로 계속 전달한다)
                logits, value, pred_interval, hidden_re = policy(ot, hidden_re)
                dist = Categorical(logits=logits.squeeze(0))
                lp = dist.log_prob(actions_tensor[t])
                new_log_probs.append(lp)
                new_values.append(value)
                new_pred_intervals.append(pred_interval)
                entropies.append(dist.entropy())

            #스택화를 통한 텐서 변환
            new_log_probs = torch.stack(new_log_probs).squeeze(-1)
            new_values = torch.stack(new_values).squeeze(-1)
            new_pred_intervals = torch.stack(new_pred_intervals).squeeze(-1)
            entropies = torch.stack(entropies).squeeze(-1)

            #PPO 비율
            ratios = torch.exp(new_log_probs - old_log_probs_tensor)

            #clipped surrpogate (ppo)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_t - new_values).pow(2).mean()

            #정책이 interval을 환경에 맞게 예측하도록 보조학습
            target_intervals = torch.tensor([env.T_true] * new_pred_intervals.shape[0], dtype=torch.float32).to(device)
            aux_loss = nn.MSELoss()(new_pred_intervals, target_intervals)

            entropy_loss = -entropies.mean() * entropy_coef

            total_loss = actor_loss + critic_loss + aux_loss_weight * aux_loss + entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, total_reward: {total_reward:.3f}, avg_return: {returns_t.mean().item():.3f}")

    #학습 종료 후 최종 롤아웃 CSV 저장
    obs = env.reset()
    hidden = policy.init_hidden(batch_size=1, device=device)
    done = False
    while not done:
        logits, value, pred_interval, hidden = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device), hidden)
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample().item()
        env.set_predicted_interval(pred_interval.detach().cpu().item())
        obs, reward, done, info = env.step(action)

    #매핑 빌드
    flash_to_click = {ft: None for ft in env.flash_times}

    for (ft, ct) in env.click_to_flash_map:
        if ft is None:
            continue
        if flash_to_click.get(ft) is None:
            flash_to_click[ft] = ct

    if len(env.click_times) > 0:
        for ft in list(flash_to_click.keys()):
            if flash_to_click[ft] is None:
                closest_click = min(env.click_times, key=lambda ct: abs(ct - ft))
                flash_to_click[ft] = closest_click

    #CSV 파일로 저장
    out_path = "model_data.csv"
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        for ft in sorted(flash_to_click.keys()):
            ct = flash_to_click.get(ft)
            if ct is None:
                writer.writerow([ft, -1.0])
            else:
                writer.writerow([ft, float(ct)])

    print("model data saved at model_data.csv")

if __name__ == "__main__":
    train()
