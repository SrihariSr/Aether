"""
Parse training logs and produce a publication-quality loss curve.

Reads both the original training log and the extended-training log,
combines them, and produces:
  - loss_curve.png: val loss + perplexity over the full 0-40K training
"""

import re
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Config
LOG_PATHS = [
    "logs/medium_run.log",           # original 30K steps
    "logs/medium_run_extended.log",  # extension 30K-40K
]
OUTPUT_PATH = "loss_curve.png"

# Match lines like: [step  5000] val_loss = 5.2756  lr = 2.96e-04  elapsed = 38.9m
VAL_LINE = re.compile(
    r"\[step\s+(\d+)\]\s+val_loss\s*=\s*([\d.]+)\s+lr\s*=\s*([\d.e+-]+)\s+elapsed\s*=\s*([\d.]+)m"
)

# Parse the logs
steps, val_losses = [], []
seen_steps = set()  # avoid duplicating step 30000 if it's in both logs

for log_path in LOG_PATHS:
    try:
        with open(log_path) as f:
            for line in f:
                m = VAL_LINE.search(line)
                if m:
                    step = int(m.group(1))
                    loss = float(m.group(2))
                    if step not in seen_steps:
                        steps.append(step)
                        val_losses.append(loss)
                        seen_steps.add(step)
    except FileNotFoundError:
        print(f"Warning: {log_path} not found, skipping")

# Sort by step (in case logs aren't in order)
sorted_pairs = sorted(zip(steps, val_losses))
steps = [p[0] for p in sorted_pairs]
val_losses = [p[1] for p in sorted_pairs]

print(f"Found {len(steps)} validation points spanning steps {steps[0]} to {steps[-1]}")
print(f"Final: step {steps[-1]}, val_loss {val_losses[-1]:.4f}")

# Find the actual best (lowest) val loss
best_idx = val_losses.index(min(val_losses))
best_step = steps[best_idx]
best_loss = val_losses[best_idx]
print(f"Best:  step {best_step}, val_loss {best_loss:.4f}, perplexity {math.exp(best_loss):.1f}")

# Plot
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

# Mark the boundary between original and extension runs
ax_loss.axvline(x=30000, color="#94a3b8", linestyle=":", linewidth=1.2, alpha=0.7)
ax_loss.text(30000, ax_loss.get_ylim()[1], "extension begins", 
             fontsize=9, color="#64748b", ha="left", va="top")

# Annotate the best point
ax_loss.annotate(
    f"  best: {best_loss:.2f}",
    xy=(best_step, best_loss),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=10,
    color="#2563eb",
    fontweight="500",
)

# Right panel: perplexity (log-scale)
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

# Mark the boundary
ax_ppl.axvline(x=30000, color="#94a3b8", linestyle=":", linewidth=1.2, alpha=0.7)

# Annotate the best perplexity
best_ppl = math.exp(best_loss)
ax_ppl.annotate(
    f"  best: {best_ppl:.0f}",
    xy=(best_step, best_ppl),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=10,
    color="#dc2626",
    fontweight="500",
)

# Overall title
fig.suptitle(
    "Aether-Medium training run | 40K steps | 8.5h on Apple M4 Max",
    fontsize=13,
    fontweight="500",
    y=1.02,
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved plot to {OUTPUT_PATH}")