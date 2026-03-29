import matplotlib.pyplot as plt
import numpy as np

# Per-class accuracy values from the validation tables
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

clean_per_class = [
    0.9610000252723694,  # airplane
    0.9679999947547913,  # automobile
    0.9430000185966492,  # bird
    0.9279999732971191,  # cat
    0.9520000219345093,  # deer
    0.9079999923706055,  # dog
    0.9810000061988831,  # frog
    0.9869999885559082,  # horse
    0.9850000143051147,  # ship
    0.9869999885559082,  # truck
]

spur_per_class = [
    0.9769999980926514,  # airplane
    0.9810000061988831,  # automobile
    0.9449999928474426,  # bird
    0.9240000247955322,  # cat
    0.9539999961853027,  # deer
    0.9330000281333923,  # dog
    0.9800000190734863,  # frog
    0.9789999723434448,  # horse
    0.984000027179718,   # ship
    0.9760000109672546,  # truck
]

# Compute degradation: spur - clean (positive = spurious is better)
degradation = [s - c for c, s in zip(clean_per_class, spur_per_class)]

x = np.arange(len(classes))
bar_width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# ── Left: grouped bar chart of clean vs spur per class ────────────────────────
ax1 = axes[0]
bars_clean = ax1.bar(x - bar_width / 2, clean_per_class, bar_width, label="Clean", color="#1f77b4", alpha=0.85)
bars_spur  = ax1.bar(x + bar_width / 2, spur_per_class,  bar_width, label="Spurious", color="#ff7f0e", alpha=0.85)

ax1.set_xlabel("Class", fontsize=13)
ax1.set_ylabel("Top-1 Accuracy", fontsize=13)
ax1.set_title("Per-Class Accuracy: Clean vs Spurious", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
ax1.legend(fontsize=12)
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.set_ylim(0.90, 1.0)

# ── Right: bar chart of accuracy degradation per class ────────────────────────
ax2 = axes[1]
colors = ["#d62728" if d > 0 else "#2ca02c" for d in degradation]
bars = ax2.bar(classes, degradation, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

ax2.axhline(0, color="black", linewidth=1.0, linestyle="-")
ax2.set_xlabel("Class", fontsize=13)
ax2.set_ylabel("Accuracy Degradation (Spurious − Clean)", fontsize=13)
ax2.set_title("Per-Class Accuracy Degradation", fontsize=14)
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.6)

# Annotate bars with degradation values
for bar, val in zip(bars, degradation):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (0.0002 if val >= 0 else -0.0005),
        f"{val:.4f}",
        ha="center", va="bottom" if val >= 0 else "top",
        fontsize=8.5,
    )

plt.tight_layout()
plt.savefig("accuracy_degradation_histogram.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: accuracy_degradation_histogram.png")