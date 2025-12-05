#!/usr/bin/env python3
"""
Two-group analysis script:
- Define two groups of float data
- Run Welch's t-test
- Make dot + violin plot

Usage:
1. Install deps:  pip install numpy scipy matplotlib
2. Put your numbers into group1 and group2 below.
3. Run:          python analysis_two_groups.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.xkcd()

# ============================
# 1. PUT YOUR DATA HERE
# ============================
# Replace the example numbers with your own.
# You can copy-paste from Excel as comma-separated values, e.g. 1.2, 3.4, 2.9

group1 = [
    1.2, 1.5, 1.7, 1.9, 2.0  # <-- your data here
]

group2 = [
    2.1, 2.3, 2.5, 2.7, 3.0  # <-- your data here
]

# Optional: change labels + y-axis label
group1_label = "Group 1"
group2_label = "Group 2"
y_label = "Measurement"


# ============================
# 2. PREP & STATS
# ============================
g1 = np.asarray(group1, dtype=float)
g2 = np.asarray(group2, dtype=float)

if g1.size == 0 or g2.size == 0:
    raise ValueError("One of the groups is empty. Please add data to group1 and group2.")

# Welch's t-test (does NOT assume equal variances)
ttest_res = stats.ttest_ind(g1, g2, equal_var=False)

mean1, mean2 = np.mean(g1), np.mean(g2)
sd1, sd2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

print("====================================")
print("Descriptive statistics")
print("====================================")
print(f"{group1_label}: n = {g1.size}, mean = {mean1:.3f}, SD = {sd1:.3f}")
print(f"{group2_label}: n = {g2.size}, mean = {mean2:.3f}, SD = {sd2:.3f}")
print("")
print("====================================")
print("Welch's independent samples t-test")
print("====================================")
print(f"t = {ttest_res.statistic:.3f}")
print(f"p = {ttest_res.pvalue:.6f}")
print("")


# ============================
# 3. PLOTTING
# ============================

def jitter(x, n_points, spread=0.08):
    """Small random jitter around x for dot scatter."""
    return x + (np.random.rand(n_points) - 0.5) * 2 * spread


fig, ax = plt.subplots(figsize=(5, 6))

# Violin plot
data = [g1, g2]
positions = [1, 2]

violins = ax.violinplot(
    data,
    positions=positions,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

# Make violins semi-transparent
for body in violins['bodies']:
    body.set_alpha(0.4)

# Dot (scatter) plots on top
x1 = jitter(positions[0], g1.size)
x2 = jitter(positions[1], g2.size)

ax.scatter(x1, g1, alpha=0.9, edgecolor="black", linewidth=0.5)
ax.scatter(x2, g2, alpha=0.9, edgecolor="black", linewidth=0.5)

# Plot group means as horizontal bars
ax.hlines(mean1, positions[0] - 0.2, positions[0] + 0.2, linewidth=2)
ax.hlines(mean2, positions[1] - 0.2, positions[1] + 0.2, linewidth=2)

# Formatting
ax.set_xticks(positions)
ax.set_xticklabels([group1_label, group2_label])
ax.set_ylabel(y_label)

# Simple p-value annotation above the groups
y_max = max(g1.max(), g2.max())
y_min = min(g1.min(), g2.min())
y_range = y_max - y_min if y_max > y_min else 1.0
line_height = y_max + 0.15 * y_range
text_height = y_max + 0.22 * y_range

ax.plot([positions[0], positions[0], positions[1], positions[1]],
        [line_height, line_height + 0.02 * y_range,
         line_height + 0.02 * y_range, line_height],
        color="black", linewidth=1)

ax.text((positions[0] + positions[1]) / 2,
        text_height,
        f"p = {ttest_res.pvalue:.3g}",
        ha="center", va="bottom")

ax.set_xlim(0.5, 2.5)
ax.set_title("Dot + Violin Plot with Welch's t-test", pad=15)

fig.tight_layout()
plt.show()
