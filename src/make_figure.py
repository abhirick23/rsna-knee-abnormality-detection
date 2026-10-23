"""Generates results/measurement.png — the one figure for the README.

Panel A: the resolution cliff. A single true +0.014 macro-AUC improvement,
         measured by paired bootstrap at three sample sizes.
Panel B: the decision that nearly got made on noise. Lateral Meniscus AUC
         per epoch on the 58 gold studies (~10 positives, SE ~= 0.09).

Writes a light and a dark version so the README can serve both themes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import paired_bootstrap

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series="#2a78d6", critical="#d03b3b"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", series="#3987e5", critical="#d03b3b"),
}

# ---- Panel A data: simulate one true effect at three sample sizes ----------
def resolution_cliff(seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for N in (58, 870, 4349):
        T = (rng.rand(N, 12) < 0.25).astype(float)
        signal = T * 1.0
        base = signal + rng.randn(N, 12) * 1.00
        better = signal + rng.randn(N, 12) * 0.92
        out.append((N,) + paired_bootstrap(better, base, T, n=400, seed=seed))
    return out

# ---- Panel B data: measured, from the 5-plane 5-fold run ------------------
LATMEN = [0.748, 0.805, 0.817, 0.706, 0.752, 0.740, 0.725]
SE = 0.09  # ~10 positives in 58 studies


def build(theme_name):
    c = THEMES[theme_name]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": c["ink"], "axes.labelcolor": c["ink2"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "axes.edgecolor": c["axis"],
    })
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(12.0, 5.4), facecolor=c["surface"],
        gridspec_kw=dict(wspace=0.30, left=0.095, right=0.97, top=0.745, bottom=0.135))

    fig.text(0.095, 0.945, "The instrument decided, not the model",
             fontsize=16, fontweight="bold", color=c["ink"], ha="left", va="top")
    fig.text(0.095, 0.885,
             "Both panels show the same problem: 58 labelled studies cannot "
             "measure the effects worth chasing.",
             fontsize=10.5, color=c["ink2"], ha="left", va="top")

    # ---------------- Panel A ----------------
    rows = resolution_cliff()
    ys = np.arange(len(rows))[::-1]
    axA.set_facecolor(c["surface"])
    axA.axvline(0, color=c["axis"], lw=1.2, zorder=1)
    for y, (N, m, lo, hi) in zip(ys, rows):
        resolvable = lo > 0
        col = c["series"] if resolvable else c["critical"]
        axA.plot([lo, hi], [y, y], lw=2, color=col, solid_capstyle="round", zorder=3)
        axA.plot([m], [y], "o", ms=9, color=col, mec=c["surface"], mew=2, zorder=4)
        label = ("resolves the effect" if resolvable else "cannot resolve the effect")
        axA.text(hi + 0.005, y + 0.22, label, va="bottom", fontsize=10,
                 color=c["ink"] if resolvable else c["critical"],
                 fontweight="bold" if resolvable else "normal")
        axA.text(hi + 0.005, y - 0.22, f"CI width {hi - lo:.3f}", va="top",
                 fontsize=9, color=c["muted"])
    axA.set_yticks(ys)
    axA.set_yticklabels([f"n = {N:,}\n{lab}" for (N, *_), lab in
                         zip(rows, ["gold set", "one fold", "out-of-fold"])],
                        fontsize=10.5, color=c["ink"])
    axA.set_xlabel("measured difference in macro AUC  (95% CI)", fontsize=10)
    axA.set_xlim(-0.075, 0.125)
    axA.set_ylim(-0.6, 2.6)
    axA.set_xticks([-0.05, 0.0, 0.05, 0.10])
    axA.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:+.02f}"))
    axA.grid(axis="x", color=c["grid"], lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.set_title("A.  One true +0.014 improvement, three sample sizes",
                  fontsize=11.5, color=c["ink"], loc="left", pad=12)

    # ---------------- Panel B ----------------
    x = np.arange(len(LATMEN))
    axB.set_facecolor(c["surface"])
    axB.fill_between(x, np.array(LATMEN) - SE, np.array(LATMEN) + SE,
                     color=c["series"], alpha=0.13, lw=0, zorder=1)
    axB.plot(x, LATMEN, lw=2, color=c["series"], solid_capstyle="round", zorder=3)
    axB.plot(x, LATMEN, "o", ms=7, color=c["series"], mec=c["surface"], mew=2, zorder=4)
    axB.plot([2], [LATMEN[2]], "o", ms=11, color=c["critical"],
             mec=c["surface"], mew=2, zorder=5)
    axB.annotate("0.817 — read at the time\nas the hypothesis confirmed",
                 xy=(2, LATMEN[2]), xytext=(2.45, 0.878),
                 fontsize=9.5, color=c["critical"], va="top",
                 arrowprops=dict(arrowstyle="-", color=c["critical"], lw=1.2))
    axB.annotate("0.706, one epoch later", xy=(3, LATMEN[3]), xytext=(3.3, 0.655),
                 fontsize=9.5, color=c["ink2"], va="top",
                 arrowprops=dict(arrowstyle="-", color=c["muted"], lw=1.2))
    axB.set_xticks(x)
    axB.set_xticklabels([f"ep{i}" for i in x], fontsize=10)
    axB.set_ylim(0.60, 0.94)
    axB.set_ylabel("Lateral Meniscus AUC", fontsize=10)
    axB.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.set_title("B.  One model, scored each epoch on 58 gold studies",
                  fontsize=11.5, color=c["ink"], loc="left", pad=12)
    axB.text(0.04, 0.617, "shaded band = ±1 SE for ~10 positives",
             fontsize=9, color=c["muted"])

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "results", f"measurement_{theme_name}.png")
    fig.savefig(out, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    for t in ("light", "dark"):
        print("wrote", build(t))
