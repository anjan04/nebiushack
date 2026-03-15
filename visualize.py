#!/usr/bin/env python3
"""Parse experiment logs and generate score evolution chart + summary report."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR, RESULTS_DIR = os.path.join(_DIR, "logs"), os.path.join(_DIR, "results")
CHART_PATH, SUMMARY_PATH = os.path.join(RESULTS_DIR, "score_evolution.png"), os.path.join(RESULTS_DIR, "summary.txt")


def load_experiments():
    """Load and parse all experiment log files, skipping corrupted ones."""
    experiments = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "experiment_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            exp_num = data.get("experiment") or data.get("iteration")
            score = data.get("primary_score") or data.get("metrics", {}).get("primary_score")
            if exp_num is None or score is None:
                continue
            experiments.append({
                "experiment": int(exp_num), "primary_score": float(score),
                "kept": bool(data.get("kept", data.get("improved", data.get("decision") == "improvement"))),
                "description": data.get("description", data.get("change", "")),
            })
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            print(f"Warning: skipping corrupted log file: {path}")
    experiments.sort(key=lambda e: e["experiment"])
    return experiments


def compute_running_best(experiments):
    """Return a list of running-best scores aligned with experiments."""
    best, running = float("-inf"), []
    for exp in experiments:
        best = max(best, exp["primary_score"])
        running.append(best)
    return running


def find_top_jumps(experiments, n=3):
    """Find the n biggest score improvements (kept experiments only)."""
    jumps, prev_best = [], 0.0
    for exp in experiments:
        if exp["kept"]:
            delta = exp["primary_score"] - prev_best
            if delta > 0:
                jumps.append((delta, exp))
            prev_best = exp["primary_score"]
    jumps.sort(key=lambda x: x[0], reverse=True)
    return jumps[:n]


def generate_chart(experiments, running_best):
    """Create and save the score evolution chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = [e["experiment"] for e in experiments]
    for kept_flag, color, sz, alpha, label in [
        (False, "red", 40, 0.6, "Reverted"),
        (True, "green", 60, 0.8, "Kept (improvement)"),
    ]:
        ex = [e["experiment"] for e in experiments if e["kept"] == kept_flag]
        ey = [e["primary_score"] for e in experiments if e["kept"] == kept_flag]
        ax.scatter(ex, ey, c=color, s=sz, alpha=alpha, label=label, zorder=3 + kept_flag)
    ax.plot(xs, running_best, c="blue", linewidth=2, label="Running best", zorder=2)
    ax.set_xlabel("Experiment Number", fontsize=12)
    ax.set_ylabel("Primary Score — Forward Velocity (m/s)", fontsize=12)
    ax.set_title("AutoRobot: AI-Designed Reward Evolution", fontsize=15, fontweight="bold")
    n_kept = sum(1 for e in experiments if e["kept"])
    best_score = max(running_best)
    ax.text(0.5, 1.02,
            f"Total: {len(experiments)} experiments | Improvements: {n_kept} | Best: {best_score:.3f} m/s",
            transform=ax.transAxes, ha="center", fontsize=10, color="gray")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {CHART_PATH}")


def _save_text(text):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(text)


def generate_summary(experiments, running_best):
    """Build a text summary, print it, and save to file."""
    if not experiments:
        msg = "No experiment logs found. Run agent.py to generate data.\n"
        print(msg)
        _save_text(msg)
        return
    total, n_kept = len(experiments), sum(1 for e in experiments if e["kept"])
    baseline, best = experiments[0]["primary_score"], max(running_best)
    pct = ((best / baseline - 1) * 100) if baseline else 0
    lines = [
        "=" * 50, "  AutoRobot — Experiment Summary", "=" * 50,
        f"  Total experiments:   {total}",
        f"  Improvements found:  {n_kept}",
        f"  Baseline score:      {baseline:.4f} m/s",
        f"  Best score:          {best:.4f} m/s",
        f"  Improvement:         {best - baseline:+.4f} m/s ({pct:.1f}%)",
        "", "  Top impactful changes:",
    ]
    top_jumps = find_top_jumps(experiments)
    for i, (delta, exp) in enumerate(top_jumps, 1):
        desc = exp["description"] or f"experiment {exp['experiment']}"
        lines.append(f"    {i}. +{delta:.4f} m/s — {desc} (exp #{exp['experiment']})")
    if not top_jumps:
        lines.append("    (none)")
    lines.append("=" * 50)
    text = "\n".join(lines) + "\n"
    print(text)
    _save_text(text)
    print(f"Summary saved to {SUMMARY_PATH}")


def main():
    experiments = load_experiments()
    if not experiments:
        print("No experiment logs found in logs/experiment_*.json")
        generate_summary([], [])
        return
    running_best = compute_running_best(experiments)
    generate_chart(experiments, running_best)
    generate_summary(experiments, running_best)


if __name__ == "__main__":
    main()
