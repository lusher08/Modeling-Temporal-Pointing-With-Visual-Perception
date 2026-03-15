import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# === 1. 데이터 불러오기 ===
df = pd.read_csv("model_data.csv", header=None, names=["flash_time", "click_time"])
df = df.dropna()  # 비어있는 행 제거

# === 2. delta 계산 ===
df["delta"] = df["click_time"] - df["flash_time"]

# === 3. 통계 요약 ===
mean_delta = df["delta"].mean()
std_delta = df["delta"].std()
median_delta = df["delta"].median()
min_delta = df["delta"].min()
max_delta = df["delta"].max()
q1, q3 = df["delta"].quantile([0.25, 0.75])

print(f"통계 요약 (Click - Flash)")
print(f"총 데이터 개수: {len(df)}")
print(f"평균: {mean_delta:.4f} 초")
print(f"표준편차: {std_delta:.4f} 초")
print(f"중앙값: {median_delta:.4f} 초")
print(f"1사분위: {q1:.4f} 초, 3사분위: {q3:.4f} 초")
print(f"최소값: {min_delta:.4f} 초, 최대값: {max_delta:.4f} 초")

# === 4. 히스토그램 시각화 ===
plt.figure(figsize=(8,5))
plt.hist(df["delta"], bins=40, color="#4682B4", edgecolor="black", alpha=0.75)

plt.axvline(mean_delta, color="red", linestyle="--", label=f"평균: {mean_delta:.3f}s")
plt.axvline(median_delta, color="orange", linestyle="-.", label=f"중앙값: {median_delta:.3f}s")

plt.title("모델 클릭 시간 분포 (Click - Flash)", fontsize=14)
plt.xlabel("Click - Flash (초)")
plt.ylabel("빈도")
plt.legend()
plt.grid(alpha=0.3)

# === 5. 이미지 저장 ===
plt.tight_layout()
plt.savefig("click_distribution.png", dpi=300)
plt.show()
