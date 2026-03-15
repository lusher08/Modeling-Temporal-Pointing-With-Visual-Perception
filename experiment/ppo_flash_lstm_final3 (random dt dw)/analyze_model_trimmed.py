import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# === 1. 데이터 불러오기 ===
df = pd.read_csv("model_data.csv", header=None, names=["flash_time", "click_time"])
df = df.dropna()  # 비어있는 행 제거

# === 2. delta 계산 ===
df["delta"] = df["click_time"] - df["flash_time"]

# === 3. 초반 데이터 트리밍 ===
# 방법 ①: 전체 데이터 중 앞부분 일정 비율 제거 (예: 15%)
trim_ratio = 0.15
trim_index = int(len(df) * trim_ratio)
df_trimmed = df.iloc[trim_index:].reset_index(drop=True)

# 방법 ②: flash_time 기준으로 초반 시간 제외 (예: 처음 5초 이내 데이터 제거)
# df_trimmed = df[df["flash_time"] > 5.0].reset_index(drop=True)

print(f"총 {len(df)}개 중 {len(df_trimmed)}개 데이터 사용 ({100*(1-trim_ratio):.1f}% 유지)")

# === 4. 통계 요약 ===
mean_delta = df_trimmed["delta"].mean()
std_delta = df_trimmed["delta"].std()
median_delta = df_trimmed["delta"].median()
min_delta = df_trimmed["delta"].min()
max_delta = df_trimmed["delta"].max()
q1, q3 = df_trimmed["delta"].quantile([0.25, 0.75])

print(f"\n통계 요약 (Click - Flash)")
print(f"평균: {mean_delta:.4f} 초")
print(f"표준편차: {std_delta:.4f} 초")
print(f"중앙값: {median_delta:.4f} 초")
print(f"1사분위: {q1:.4f} 초, 3사분위: {q3:.4f} 초")
print(f"최소값: {min_delta:.4f} 초, 최대값: {max_delta:.4f} 초")

# === 5. 히스토그램 시각화 ===
plt.figure(figsize=(8,5))
plt.hist(df_trimmed["delta"], bins=40, color="#4682B4", edgecolor="black", alpha=0.75)

plt.axvline(mean_delta, color="red", linestyle="--", label=f"평균: {mean_delta:.3f}s")
plt.axvline(median_delta, color="orange", linestyle="-.", label=f"중앙값: {median_delta:.3f}s")

plt.title("모델 클릭 시간 분포 (Click - Flash, 초반 데이터 제외)", fontsize=14)
plt.xlabel("Click - Flash (초)")
plt.ylabel("빈도")
plt.legend()
plt.grid(alpha=0.3)

# === 6. 이미지 저장 ===
plt.tight_layout()
plt.savefig("click_distribution_trimmed.png", dpi=300)
plt.show()
