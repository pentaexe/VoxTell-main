"""
Generate result figures for VoxTell optimization report.
Saves PNGs to figures/ directory.

Usage:
    python plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

VERSIONS = ["v0_gpu\n(baseline)", "v1\ntile=0.75", "v2\n+Cache", "v3\n+Numba"]
PRE   = [0.13, 0.13, 0.13, 0.09]
EMBED = [0.51, 0.51, 0.02, 0.02]
SLIDE = [2.44, 2.22, 2.22, 2.22]
POST  = [0.03, 0.03, 0.03, 0.03]
TOTALS = [p+e+s+po for p,e,s,po in zip(PRE, EMBED, SLIDE, POST)]

COLORS = {
    "Preprocessing": "#4C9BE8",
    "Text Embedding": "#E85C4C",
    "Sliding Window": "#F0A500",
    "Postprocessing": "#6ABF69",
}

# ── Figure 1: Stacked bar — per-phase breakdown across versions ───────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(VERSIONS))
w = 0.55

bars_pre   = ax.bar(x, PRE,   w, label="Preprocessing",  color=COLORS["Preprocessing"])
bars_embed = ax.bar(x, EMBED, w, bottom=PRE, label="Text Embedding", color=COLORS["Text Embedding"])
bars_slide = ax.bar(x, SLIDE, w, bottom=[p+e for p,e in zip(PRE,EMBED)], label="Sliding Window", color=COLORS["Sliding Window"])
bars_post  = ax.bar(x, POST,  w, bottom=[p+e+s for p,e,s in zip(PRE,EMBED,SLIDE)], label="Postprocessing", color=COLORS["Postprocessing"])

# Annotate totals
for i, total in enumerate(TOTALS):
    ax.text(i, total + 0.05, f"{total:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(VERSIONS, fontsize=10)
ax.set_ylabel("Inference Time (seconds)", fontsize=12)
ax.set_title("VoxTell Inference Time — Per-Phase Breakdown by Optimization Version", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, 4.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate speedup vs v0
for i in range(1, len(VERSIONS)):
    speedup = TOTALS[0] / TOTALS[i]
    ax.text(i, TOTALS[i] + 0.08, f"{speedup:.1f}×", ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))

plt.tight_layout()
plt.savefig("figures/fig1_stacked_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig1_stacked_breakdown.png")

# ── Figure 2: Log-scale total time comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#D9534F", "#5B9BD5", "#70AD47", "#ED7D31"]
bars = ax.bar(x, TOTALS, w, color=bar_colors, edgecolor="white", linewidth=0.8)

for bar, total in zip(bars, TOTALS):
    ax.text(bar.get_x() + bar.get_width()/2, total * 1.15,
            f"{total:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(VERSIONS, fontsize=10)
ax.set_ylabel("Total Inference Time (seconds)", fontsize=12)
ax.set_title("Total Inference Time — GPU Baseline to Optimized", fontsize=13, fontweight="bold")
ax.set_ylim(0, 4.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Speedup labels
for i in range(1, len(VERSIONS)):
    speedup = TOTALS[0] / TOTALS[i]
    ax.text(i, TOTALS[i] + 0.08, f"{speedup:.2f}×\nfaster", ha="center",
            fontsize=8.5, color="navy")

plt.tight_layout()
plt.savefig("figures/fig2_total_logscale.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig2_total_logscale.png")

# ── Figure 3: H100 fair algorithmic comparison ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

phases = ["Preprocessing", "Text Embedding", "Sliding Window", "Postprocessing"]
v0_h100 = [0.14, 6.76, 0.51, 0.12]   # H100 unoptimized (cold embed, tile_step=0.5)
v3_h100 = [0.20, 0.06, 0.50, 0.17]   # H100 optimized   (warm cache, tile_step=0.75)
x2 = np.arange(len(phases))
w2 = 0.35

ax = axes[0]
ax.bar(x2 - w2/2, v0_h100, w2, label="v0_gpu (no optimizations, cold)", color="#5B9BD5", alpha=0.9)
ax.bar(x2 + w2/2, v3_h100, w2, label="v3 (all optimizations, warm cache)", color="#70AD47", alpha=0.9)
ax.set_xticks(x2)
ax.set_xticklabels(phases, fontsize=9, rotation=10)
ax.set_ylabel("Time (seconds)", fontsize=11)
ax.set_title("H100 Algorithmic Comparison: Per-Phase", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
totals_fair = [sum(v0_h100), sum(v3_h100)]
labels_fair = ["v0_gpu\n(no optimizations\ncold embed)", "v3\n(all optimizations\nwarm cache)"]
bar_colors_fair = ["#5B9BD5", "#70AD47"]
bars2 = ax.bar([0, 1], totals_fair, 0.5, color=bar_colors_fair, alpha=0.9)
for bar, total in zip(bars2, totals_fair):
    ax.text(bar.get_x() + bar.get_width()/2, total + 0.15,
            f"{total:.2f}s", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_fair, fontsize=10)
ax.set_ylabel("Total Inference Time (seconds)", fontsize=11)
ax.set_title(f"H100 Algorithmic Comparison: Total\n(Speedup: {totals_fair[0]/totals_fair[1]:.1f}×)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 9.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle("Algorithmic Speedup on H100 MIG 3g.40gb (same hardware, both INT4)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig3_fair_gpu_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig3_fair_gpu_comparison.png")

# ── Figure 4: CT Accuracy — DSC per organ ────────────────────────────────────
organs = [
    "Liver", "Right\nKidney", "Spleen", "Pancreas", "Aorta",
    "IVC", "RAG", "LAG", "Gallbladder", "Esophagus",
    "Stomach", "Duodenum", "Left\nKidney"
]
# Per-organ DSC values from accuracy_eval.py (AMOS, 5 cases, seed=42)
dsc_v0 = [0.9667, 0.9393, 0.9356, 0.7763, 0.9207, 0.7174, 0.5866, 0.6670, 0.8264, 0.6207, 0.8934, 0.7240, 0.9431]
dsc_v3 = [0.9671, 0.9397, 0.9355, 0.7699, 0.9225, 0.7190, 0.5948, 0.6646, 0.8264, 0.6208, 0.8924, 0.7254, 0.9431]

x3 = np.arange(len(organs))
w3 = 0.38
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x3 - w3/2, dsc_v0, w3, label=f"v0 (tile=0.5)  mean={np.mean(dsc_v0):.4f}", color="#5B9BD5", alpha=0.9)
ax.bar(x3 + w3/2, dsc_v3, w3, label=f"v3 (tile=0.75) mean={np.mean(dsc_v3):.4f}", color="#70AD47", alpha=0.9)
ax.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Minimum gate (0.800)")
ax.set_xticks(x3)
ax.set_xticklabels(organs, fontsize=8.5)
ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=11)
ax.set_ylim(0.5, 1.02)
ax.set_title("CT Segmentation Accuracy — Per-Organ DSC (AMOS, 5 cases, seed=42)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig4_ct_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig4_ct_accuracy.png")

print("\nAll figures saved to figures/")
print("Add to report with: ![](figures/fig1_stacked_breakdown.png)")
