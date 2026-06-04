#!/usr/bin/env python3
"""
Generate all figures for the AgentCell KDD paper from the real experimental
numbers. Figures are written to Figures/ as high-resolution PDFs (vector) and
PNGs. Fonts are deliberately large and bold for readability in two-column print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("Figures")
OUT.mkdir(exist_ok=True)

# ----- Global style: large, bold, legible -----
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# Palette
C_AGENT = "#1b6ca8"
C_FIXED = "#a8a8a8"
C_BASE = "#d98c00"
C_HL = "#c0392b"
C_GREEN = "#2e7d32"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# =====================================================================
# FIGURE 1 (saved as fig_agent_vs_fixed): Agent vs fixed pipelines, NSC + NSCB
# =====================================================================
def fig_agent_vs_fixed():
    labels = ["F5\nRN50\nDAPI", "F4\nViT\nDAPI", "F3\nRN50\n3-ch",
              "F2\nViT\n3-ch L2", "F1\nViT\n3-ch", "Agent\n(ours)"]
    nsc = [0.7573, 0.7379, 0.8835, 0.9029, 0.9126, 0.9709]
    nscb = [0.5437, 0.5825, 0.6699, 0.7087, 0.6990, 0.7670]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [C_FIXED] * 5 + [C_AGENT]
    b1 = ax.bar(x - w/2, nsc, w, label="NSC", color=colors,
                edgecolor="black", linewidth=1.5)
    b2 = ax.bar(x + w/2, nscb, w, label="NSCB", color=colors, alpha=0.55,
                edgecolor="black", linewidth=1.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Autonomous Agent vs. Fixed Pipelines (BBBC021)")
    ax.axhline(0.9126, ls="--", color=C_BASE, lw=2.2)
    ax.text(0.05, 0.918, "best fixed NSC = 0.913", color=C_BASE,
            fontsize=13, fontweight="bold")
    for xi, v in zip(x - w/2, nsc):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")
    # annotate the +5.8 gain
    ax.annotate("", xy=(5 - w/2, 0.9709), xytext=(5 - w/2, 0.9126),
                arrowprops=dict(arrowstyle="<->", color=C_HL, lw=2.5))
    ax.text(5.0, 0.945, "+5.8", color=C_HL, fontsize=14, fontweight="bold",
            ha="left")
    legA = mpatches.Patch(facecolor=C_AGENT, edgecolor="black", label="Agent")
    legF = mpatches.Patch(facecolor=C_FIXED, edgecolor="black", label="Fixed")
    legN = mpatches.Patch(facecolor="white", edgecolor="black", label="NSC (solid)")
    legB = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                          label="NSCB (hatched)")
    ax.legend(handles=[legA, legF, legN, legB], loc="lower right", ncol=1,
              frameon=True)
    save(fig, "fig_agent_vs_fixed")


# =====================================================================
# FIGURE 2 (fig_trajectory): the agent's decision trajectory (main run)
# =====================================================================
def fig_trajectory():
    # main run (run_3c_clean) step -> NSC where scored
    steps = [
        ("embed\nViT-IN", None, "baseline\nembed"),
        ("score\nplain", 0.9126, "baseline\n0.913"),
        ("train\nDAPI", None, "SSL\n100ep"),
        ("train\nTubulin", None, "SSL\n100ep"),
        ("train\nActin", None, "SSL\n100ep"),
        ("embed\nViT-tr", None, "trained\nembed"),
        ("score\nplain", 0.9320, "0.932"),
        ("score\nL2", 0.9709, "0.971\n(best)"),
        ("DONE", None, "converge"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n = len(steps)
    xs = np.linspace(0.5, n - 0.5, n)
    y = 0.55
    for i, (lab, nsc, note) in enumerate(steps):
        if "train" in lab:
            fc = C_GREEN
        elif "score" in lab:
            fc = C_AGENT
        elif "DONE" in lab:
            fc = C_HL
        else:
            fc = C_BASE
        box = FancyBboxPatch((xs[i] - 0.36, y - 0.13), 0.72, 0.26,
                             boxstyle="round,pad=0.02", linewidth=2,
                             edgecolor="black", facecolor=fc, alpha=0.85)
        ax.add_patch(box)
        ax.text(xs[i], y, lab, ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold")
        ax.text(xs[i], y - 0.26, note, ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=(C_HL if (nsc == 0.9709) else "black"))
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((xs[i] + 0.36, y), (xs[i+1] - 0.36, y),
                         arrowstyle="-|>", mutation_scale=22, lw=2.2,
                         color="black"))
    ax.text(xs[0], y + 0.30, "baseline", ha="center", fontsize=12,
            color=C_BASE, fontweight="bold")
    ax.text((xs[2]+xs[4])/2, y + 0.30, "self-supervised training (per channel)",
            ha="center", fontsize=12, color=C_GREEN, fontweight="bold")
    ax.text((xs[6]+xs[7])/2, y + 0.30, "normalization choice",
            ha="center", fontsize=12, color=C_AGENT, fontweight="bold")
    ax.set_xlim(0, n)
    ax.set_ylim(0.1, 1.0)
    ax.axis("off")
    ax.set_title("Autonomous Decision Trajectory: Goal \u2192 SOTA (NSC 0.971)",
                 fontsize=18)
    save(fig, "fig_trajectory")


# =====================================================================
# FIGURE 3 (fig_robustness): three independent runs converge to 0.971
# =====================================================================
def fig_robustness():
    runs = ["Main\n(temp 0)", "Self-corr off\n(temp 0)", "Robust\n(temp 0.4)"]
    final = [0.9709, 0.9709, 0.9709]
    base = 0.9126
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(x, final, 0.55, color=C_AGENT, edgecolor="black", linewidth=2,
           label="Agent final NSC")
    ax.axhline(base, ls="--", color=C_BASE, lw=2.2, label="ImageNet baseline")
    for xi, v in zip(x, final):
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel("Final NSC")
    ax.set_ylim(0.88, 0.99)
    ax.set_title("Reproducibility: Three Independent Runs")
    ax.legend(loc="lower center", frameon=True)
    save(fig, "fig_robustness")


# =====================================================================
# FIGURE 4 (fig_planner): planner comparison - compute vs outcome
# =====================================================================
def fig_planner():
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    # DeepSeek: 100 epochs/channel x3 = 300 epoch-units -> 0.9709
    # GPT-5.5 : 400 epochs/channel x3 = 1200 epoch-units -> 0.9417 (compound set)
    comp = [300, 1200]
    nsc = [0.9709, 0.9417]
    names = ["DeepSeek-V3\n(100 ep/ch)", "GPT-5.5\n(400 ep/ch)"]
    cols = [C_AGENT, C_HL]
    ax.scatter(comp, nsc, s=[420, 420], c=cols, edgecolor="black",
               linewidth=2.5, zorder=3)
    for c, n, nm in zip(comp, nsc, names):
        ax.annotate(nm, (c, n), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=14,
                    fontweight="bold")
    ax.axhline(0.9709, ls=":", color=C_AGENT, lw=1.8, alpha=0.7)
    ax.set_xlabel("Training compute (total SSL epochs)")
    ax.set_ylabel("Final NSC")
    ax.set_xlim(0, 1400)
    ax.set_ylim(0.92, 0.985)
    ax.set_title("Planner Choice: 4\u00d7 Compute, No Gain")
    # arrow highlighting more compute lower result
    ax.annotate("", xy=(1200, 0.9417), xytext=(300, 0.9709),
                arrowprops=dict(arrowstyle="->", color="black", lw=2,
                                ls="--", alpha=0.6))
    ax.text(720, 0.957, "4\u00d7 compute\n\u22122.9 NSC", fontsize=13,
            fontweight="bold", ha="center", color=C_HL)
    save(fig, "fig_planner")


# =====================================================================
# FIGURE 5 (fig_loss_vs_metric): loss saturates, downstream metric varies
# =====================================================================
def fig_loss_vs_metric():
    # epochs vs downstream NSC (from runs): 0 (ImageNet)=0.9126,
    # 100=0.9709 (DeepSeek), 200=0.9417 (first run_3c), 400=0.9709 (orig) / 0.9417 (gpt55)
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ep = [0, 100, 200, 400, 400]
    nscv = [0.9126, 0.9709, 0.9417, 0.9709, 0.9417]
    tags = ["ImageNet", "100ep", "200ep", "400ep (run A)", "400ep (run B)"]
    ax.scatter(ep, nscv, s=300, c=[C_BASE, C_AGENT, C_GREEN, C_AGENT, C_HL],
               edgecolor="black", linewidth=2, zorder=3)
    for e, v, t in zip(ep, nscv, tags):
        dy = 12 if v > 0.95 else -22
        ax.annotate(t, (e, v), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("SSL training epochs")
    ax.set_ylabel("Downstream NSC")
    ax.set_xlim(-30, 460)
    ax.set_ylim(0.90, 0.985)
    ax.set_title("Training Loss Saturates; Retrieval Quality Varies")
    save(fig, "fig_loss_vs_metric")


if __name__ == "__main__":
    fig_agent_vs_fixed()
    fig_trajectory()
    fig_robustness()
    fig_planner()
    fig_loss_vs_metric()
    print("ALL FIGURES DONE")
