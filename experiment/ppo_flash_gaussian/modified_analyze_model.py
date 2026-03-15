import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. 데이터 불러오기 ===
CSV_PATH = "model_data.csv"
df = pd.read_csv(CSV_PATH, header=None, names=["flash_time", "click_time"])
df = df.dropna()

# === 2. delta 계산 ===
df["delta"] = df["click_time"] - df["flash_time"]   # 단위: 초

# === 초 → ms로 변환 ===
df["delta_ms"] = df["delta"] * 1000

# === 3. 통계량 계산 ===
mean_ms = df["delta_ms"].mean()
std_ms = df["delta_ms"].std()
min_ms = df["delta_ms"].min()
max_ms = df["delta_ms"].max()

print(f"총 개수: {len(df)}")
print(f"평균: {mean_ms:.2f} ms")
print(f"표준편차: {std_ms:.2f} ms")
print(f"최소값: {min_ms:.2f} ms, 최대값: {max_ms:.2f} ms")

# === 4. 시간창(Wt) 및 Dt 설정 ===
# 모델에서는 flash가 순간적으로 발생하므로
# Wt=0 으로 간주하고 Dt를 플래시 간격(T_true)에 맞춰 너가 조절 가능하게 설계
# 플래시 간격이 1초 → Dt=1000ms
WT = 0          # 플래시가 순간 발생
DT = 1000       # 1초 간격 → 1000ms

half = DT / 2
x_min, x_max = -half, WT + half   # -500ms ~ +500ms 범위

vals = df["delta_ms"].values

# === bin 설정 ===
bin_width = 1000 / 60
range_width = (x_max - x_min)
n_bins = int(range_width // bin_width) + 1

# === 5. 히스토그램 ===
plt.figure(figsize=(11, 6))
plt.hist(vals, bins=n_bins, range=(x_min, x_max), color="cornflowerblue",
         alpha=0.8, edgecolor="black", label="Model outputs")

# 평균선
plt.axvline(mean_ms, color="red", linestyle="-", linewidth=2,
            label=f"μ={mean_ms:+.1f} ms")

# 기준선 (플래시 순간)
plt.axvline(0, color="green", linestyle="--", linewidth=1.5, label="Flash time (0ms)")

# 음영 범위
plt.axvspan(x_min, x_max, color="skyblue", alpha=0.12)

# 텍스트 박스 (mean, std)
plt.text(
    mean_ms, plt.ylim()[1] * 0.9,
    f"μ={mean_ms:+.1f} ms\nσ={std_ms:.1f}",
    color="black", fontsize=10, fontweight="bold", ha="center",
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
              boxstyle="round,pad=0.3")
)

plt.title("Model Reaction Time Distribution (Click - Flash)")
plt.xlabel("Δt (ms, relative to flash time)")
plt.ylabel("Count")
plt.xlim(x_min, x_max)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
