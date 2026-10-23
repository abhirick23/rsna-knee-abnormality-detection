"""Generates a single-panel cover image (Panel A only) for the Kaggle dataset
thumbnail -- the full two-panel figure reads cluttered at small size.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from stats import paired_bootstrap

c = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", series="#2a78d6", critical="#d03b3b")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": c["ink"], "axes.labelcolor": c["ink2"],
    "xtick.color": c["muted"], "ytick.color": c["muted"],
    "axes.edgecolor": c["axis"],
})


def resolution_cliff(seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for N in (58, 870, 4349):
        T = (rng.rand(N, 12) < 0.25).astype(float)
        signal = T * 1.0
        base = signal + rng.randn(N, 12) * 1.00
        better = signal + rng.randn(N, 12) * 0.92
        out.append((N,) + paired_bootstrap(better, base, T, n=500, seed=seed))
    return out


fig, ax = plt.subplots(figsize=(8.0, 6.0), facecolor=c["surface"],
                        gridspec_kw=dict(left=0.14, right=0.97, top=0.80, bottom=0.14))

fig.text(0.07, 0.945, "The instrument decided, not the model",
         fontsize=17, fontweight="bold", color=c["ink"], ha="left", va="top")
fig.text(0.07, 0.885,
         "One true +0.014 macro-AUC improvement, measured at three sample sizes",
         fontsize=11, color=c["ink2"], ha="left", va="top")

rows = resolution_cliff()
ys = np.arange(len(rows))[::-1]
ax.set_facecolor(c["surface"])
ax.axvline(0, color=c["axis"], lw=1.2, zorder=1)
for y, (N, m, lo, hi) in zip(ys, rows):
    resolvable = lo > 0
    col = c["series"] if resolvable else c["critical"]
    ax.plot([lo, hi], [y, y], lw=3, color=col, solid_capstyle="round", zorder=3)
    ax.plot([m], [y], "o", ms=12, color=col, mec=c["surface"], mew=2.5, zorder=4)
    label = "resolves the effect" if resolvable else "cannot resolve the effect"
    ax.text(hi + 0.006, y + 0.24, label, va="bottom", fontsize=12,
            color=c["ink"] if resolvable else c["critical"],
            fontweight="bold" if resolvable else "normal")
    ax.text(hi + 0.006, y - 0.24, f"CI width {hi - lo:.3f}", va="top",
            fontsize=10.5, color=c["muted"])
ax.set_yticks(ys)
ax.set_yticklabels([f"n = {N:,}\n{lab}" for (N, *_), lab in
                    zip(rows, ["gold set", "one fold", "out-of-fold"])],
                   fontsize=12.5, color=c["ink"])
ax.set_xlabel("measured difference in macro AUC  (95% CI)", fontsize=11.5)
ax.set_xlim(-0.075, 0.135)
ax.set_ylim(-0.6, 2.6)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:+.02f}"))
ax.grid(axis="x", color=c["grid"], lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "kaggle_dataset", "cover.png")
fig.savefig(out, dpi=200, facecolor=c["surface"])
print("wrote", out)
