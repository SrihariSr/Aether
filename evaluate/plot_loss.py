"""
Parse training logs and produce a loss curve in loss_curve.png
"""

import re
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_PATH = "logs/medium_run.log"
OUTPUT_PATH = "loss_curve.png"

VAL_LINE = re.compile(
    r"\[step\s+(\d+)\]\s+val_loss\s*=\s*([\d.]+)\s+lr\s*=\s*([\d.e+-]+)\s+elapsed\s*=\s*([\d.]+)m"
)

steps, val_losses, elapsed = [], [], []

with open(LOG_PATH) as f:
    for line in f:
        m = VAL_LINE.search(line)
        if m:
            step, loss, lr, elap = m.groups()
            steps.append(int(step))
            val_losses.append(float(loss))
            elapsed.append(float(elap))

print(f"Found {len(steps)} validation points")
print(f"Final: step {steps[-1]}, val_loss {val_losses[-1]:.4f}, "
      f"perplexity {math.exp(val_losses[-1]):.1f}")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#666666",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
})

fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left panel: validation loss
ax_loss.plot(steps, val_losses, color="#2563eb", linewidth=2.0)
ax_loss.scatter(steps, val_losses, color="#2563eb", s=18, zorder=5)
ax_loss.set_xlabel("Training step")
ax_loss.set_ylabel("Validation loss (cross-entropy)")
ax_loss.set_title("Validation loss", fontweight="500", pad=12)
ax_loss.grid(True, axis="y", alpha=0.3, linestyle="--")
ax_loss.set_xlim(0, max(steps) * 1.02)

final_step, final_loss = steps[-1], val_losses[-1]
ax_loss.annotate(
    f"  {final_loss:.2f}",
    xy=(final_step, final_loss),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=10,
    color="#2563eb",
    fontweight="500",
    va="center",
)

perplexities = [math.exp(l) for l in val_losses]
ax_ppl.plot(steps, perplexities, color="#dc2626", linewidth=2.0)
ax_ppl.scatter(steps, perplexities, color="#dc2626", s=18, zorder=5)
ax_ppl.set_yscale("log")
ax_ppl.set_xlabel("Training step")
ax_ppl.set_ylabel("Validation perplexity (log scale)")
ax_ppl.set_title("Validation perplexity", fontweight="500", pad=12)
ax_ppl.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")
ax_ppl.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_ppl.set_xlim(0, max(steps) * 1.02)

final_ppl = perplexities[-1]
ax_ppl.annotate(
    f"  {final_ppl:.0f}",
    xy=(final_step, final_ppl),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=10,
    color="#dc2626",
    fontweight="500",
    va="center",
)

total_hours = elapsed[-1] / 60
fig.suptitle(
    f"Andromeda-Medium training run | 30K steps | {total_hours:.1f}h on Apple M4 Max",
    fontsize=13,
    fontweight="500",
    y=1.02,
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved plot to {OUTPUT_PATH}")