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

VERSIONS = ["v0\n(CPU bug)", "v1\nFP16+GPU\ntile=0.75", "v2\n+Cache", "v3\n+Numba"]
PRE   = [0.38, 0.10, 0.10, 0.09]
EMBED = [126.02, 2.70, 0.02, 0.02]
SLIDE = [18.66, 5.22, 5.58, 2.22]
POST  = [0.19, 0.04, 0.03, 0.03]
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
    ax.text(i, total + 1.5, f"{total:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(VERSIONS, fontsize=10)
ax.set_ylabel("Inference Time (seconds)", fontsize=12)
ax.set_title("VoxTell Inference Time — Per-Phase Breakdown by Optimization Version", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(0, 165)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate speedup vs v0
for i in range(1, len(VERSIONS)):
    speedup = TOTALS[0] / TOTALS[i]
    ax.text(i, TOTALS[i] + 6, f"{speedup:.1f}×", ha="center", va="bottom",
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

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(VERSIONS, fontsize=10)
ax.set_ylabel("Total Inference Time (seconds, log scale)", fontsize=12)
ax.set_title("Total Inference Time — Log Scale Comparison", fontsize=13, fontweight="bold")
ax.set_ylim(0.5, 500)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Speedup labels
for i in range(1, len(VERSIONS)):
    speedup = TOTALS[0] / TOTALS[i]
    ax.annotate(f"{speedup:.1f}× faster\nvs v0",
                xy=(i, TOTALS[i]), xytext=(i + 0.35, TOTALS[i] * 2.5),
                fontsize=8.5, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy", lw=1.2))

plt.tight_layout()
plt.savefig("figures/fig2_total_logscale.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig2_total_logscale.png")

# ── Figure 3: Fair GPU-vs-GPU comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

phases = ["Preprocessing", "Text Embedding", "Sliding Window", "Postprocessing"]
v0_gpu = [0.13, 0.51, 2.44, 0.03]
v3_gpu = [0.09, 0.02, 2.22, 0.03]
x2 = np.arange(len(phases))
w2 = 0.35

ax = axes[0]
ax.bar(x2 - w2/2, v0_gpu, w2, label="v0_gpu (baseline, FP16, no cache)", color="#5B9BD5", alpha=0.9)
ax.bar(x2 + w2/2, v3_gpu, w2, label="v3 (all optimizations)", color="#70AD47", alpha=0.9)
ax.set_xticks(x2)
ax.set_xticklabels(phases, fontsize=9, rotation=10)
ax.set_ylabel("Time (seconds)", fontsize=11)
ax.set_title("Fair GPU-vs-GPU: Per-Phase", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
totals_fair = [sum(v0_gpu), sum(v3_gpu)]
labels_fair = ["v0_gpu\n(FP16, no cache\ntile=0.5)", "v3\n(all optimizations\ntile=0.75)"]
bar_colors_fair = ["#5B9BD5", "#70AD47"]
bars2 = ax.bar([0, 1], totals_fair, 0.5, color=bar_colors_fair, alpha=0.9)
for bar, total in zip(bars2, totals_fair):
    ax.text(bar.get_x() + bar.get_width()/2, total + 0.05,
            f"{total:.2f}s", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_fair, fontsize=10)
ax.set_ylabel("Total Inference Time (seconds)", fontsize=11)
ax.set_title(f"Fair GPU-vs-GPU: Total\n(Algorithmic speedup: {totals_fair[0]/totals_fair[1]:.2f}×)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 4.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle("Fair GPU-vs-GPU Comparison (RTX 4070 SUPER, both FP16)", fontsize=13, fontweight="bold", y=1.02)
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
# Per-organ DSC values from accuracy_eval.py (v3, FLARE 2022, 5 cases)
dsc_v0 = [0.963, 0.921, 0.958, 0.780, 0.952, 0.878, 0.782, 0.771, 0.652, 0.742, 0.861, 0.693, 0.935]
dsc_v3 = [0.964, 0.922, 0.959, 0.783, 0.953, 0.879, 0.784, 0.773, 0.655, 0.744, 0.863, 0.695, 0.937]

x3 = np.arange(len(organs))
w3 = 0.38
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x3 - w3/2, dsc_v0, w3, label=f"v0 (tile=0.5)  mean={np.mean(dsc_v0):.4f}", color="#5B9BD5", alpha=0.9)
ax.bar(x3 + w3/2, dsc_v3, w3, label=f"v3 (tile=0.75) mean={np.mean(dsc_v3):.4f}", color="#70AD47", alpha=0.9)
ax.axhline(0.88, color="red", linestyle="--", linewidth=1.2, label="Minimum gate (0.880)")
ax.set_xticks(x3)
ax.set_xticklabels(organs, fontsize=8.5)
ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=11)
ax.set_ylim(0.5, 1.02)
ax.set_title("CT Segmentation Accuracy — Per-Organ DSC (FLARE 2022, 5 cases)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig4_ct_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig4_ct_accuracy.png")

print("\nAll figures saved to figures/")
print("Add to report with: ![](figures/fig1_stacked_breakdown.png)")
