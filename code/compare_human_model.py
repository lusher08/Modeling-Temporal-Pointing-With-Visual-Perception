import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    flash_times, click_times = [], []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flash_times.append(float(row["flash_time"]))
            if row["click_time"] != "":
                click_times.append(float(row["click_time"]))
    return np.array(flash_times), np.array(click_times)

def compute_reaction_times(flash_times, click_times, tol=0.5):
    rts = []
    for f in flash_times:
        candidates = [c for c in click_times if abs(c - f) < tol]
        if candidates:
            closest = min(candidates, key=lambda c: abs(c-f))
            rts.append(closest - f)
    return np.array(rts)

def compute_normalized_rts(flash_times, click_times):
    """
    플래시-클릭을 순서대로 매칭해서 normalized RT 계산
    """
    n = min(len(flash_times), len(click_times))
    norm_rts = []
    abs_rts = []
    for i in range(n):
        delta = click_times[i] - flash_times[i]

        if i < n-1:
            interval = flash_times[i+1] - flash_times[i]
        else:
            interval = np.median(np.diff(flash_times))
        norm_rts.append(delta / interval)
        abs_rts.append(delta)
    return np.array(norm_rts), np.array(abs_rts)


human_flash, human_click = load_data("human_data.csv")
model_flash, model_click = load_data("model_data.csv")

human_norm_rts, human_abs_rts = compute_normalized_rts(human_flash, human_click)
model_norm_rts, model_abs_rts = compute_normalized_rts(model_flash, model_click)


plt.hist(human_norm_rts, bins=20, alpha=0.5, label="Human (normalized)")
plt.hist(model_norm_rts, bins=20, alpha=0.5, label="Model (normalized)")
plt.xlabel("Normalized Reaction Time (fraction of interval)")
plt.ylabel("Count")
plt.legend()
plt.show()

print("Human normalized mean:", np.mean(human_norm_rts))
print("Model normalized mean:", np.mean(model_norm_rts))
print("Human normalized std:", np.std(human_norm_rts))
print("Model normalized std:", np.std(model_norm_rts))

