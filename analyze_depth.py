"""Hallucination rate vs. trajectory depth for a HalluWorld terminal run.

Usage:
    python3 analyze_depth.py results/u_4omini/results.jsonl
    python3 analyze_depth.py results/u_4omini/results.jsonl results/u_haiku/results.jsonl

Prints a table per run and saves depth_plot.png (needs matplotlib; table works without it).
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

# The terminal track leaves steps_before_probe empty, but the v0.1 question bank encodes the
# trigger step in each raw filename, e.g. 000288_make-doom-for-mips_step-0214_....json
BANK_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--DegenAI-Labs--HalluWorld/snapshots/*/v0.1/raw/terminal/*_step-*.json")


def terminal_steps():
    steps = {}
    for f in glob.glob(BANK_GLOB):
        m = re.match(r"(\d{6})_.*_step-(\d+)_", os.path.basename(f))
        if m:
            steps[f"terminal/{m.group(1)}"] = int(m.group(2))
    return steps


STEPS = terminal_steps()


def load(path):
    # A --resume run appends new records, so keep the latest scored record per question.
    latest = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("error") or r.get("is_correct") is None:
            continue
        if r.get("steps_before_probe") is None and r["question_id"] in STEPS:
            r["steps_before_probe"] = STEPS[r["question_id"]]
        latest[r["question_id"]] = r
    return list(latest.values())


def quartile_bins(rows, key, n_bins=4):
    rows = sorted(rows, key=lambda r: r[key])
    size = len(rows) / n_bins
    return [rows[round(i * size): round((i + 1) * size)] for i in range(n_bins)]


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def report(path):
    rows = load(path)
    model = rows[0].get("model", Path(path).parent.name)
    halluc = lambda rs: 1 - sum(bool(r["is_correct"]) for r in rs) / len(rs)
    print(f"\n=== {model}  ({len(rows)} scored probes, {halluc(rows):.1%} hallucination) ===")

    out = {}
    for key, label in [("steps_before_probe", "depth (commands before probe)"),
                       ("prompt_tokens", "prompt tokens")]:
        keyed = [r for r in rows if isinstance(r.get(key), (int, float))]
        if len(keyed) < 8:
            print(f"\nBy {label}: field '{key}' is empty for this run, skipping.")
            continue
        if len(keyed) < len(rows):
            print(f"\n(note: {len(rows) - len(keyed)} probes have no '{key}' and are left out below)")
        rows_all, rows = rows, keyed
        print(f"\nBy {label}:")
        print(f"  {'range':>18}  {'n':>3}  {'halluc':>7}  {'±95%':>6}")
        pts = []
        for b in quartile_bins(rows, key):
            h = halluc(b)
            moe = 1.96 * (h * (1 - h) / len(b)) ** 0.5
            lo, hi = b[0][key], b[-1][key]
            print(f"  {f'{lo}–{hi}':>18}  {len(b):>3}  {h:>7.1%}  {moe:>6.1%}")
            pts.append((sum(r[key] for r in b) / len(b), h))
        r = corr([x[key] for x in rows], [0 if x["is_correct"] else 1 for x in rows])
        print(f"  correlation with hallucinating: r = {r:+.2f}")
        out[key] = pts
        rows = rows_all
    return model, out


def main(paths):
    if not STEPS:
        print("(note: couldn't find the v0.1 question bank in your Hugging Face cache, so depth is unavailable)")
    results = [report(p) for p in paths]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot. pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, out in results:
        for ax, key in zip(axes, ["steps_before_probe", "prompt_tokens"]):
            if key not in out:
                continue
            xs, ys = zip(*out[key])
            ax.plot(xs, ys, marker="o", label=model)
    axes[0].set_xlabel("commands before probe (quartile mean)")
    axes[1].set_xlabel("prompt tokens (quartile mean)")
    for ax in axes:
        ax.set_ylabel("hallucination rate")
        ax.set_ylim(0, 1)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle("HalluWorld terminal: uncertainty probes")
    fig.tight_layout()
    fig.savefig("depth_plot.png", dpi=150)
    print("\nSaved depth_plot.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/u_4omini/results.jsonl"])