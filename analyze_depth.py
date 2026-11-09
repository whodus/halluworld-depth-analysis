"""Hallucination rate vs. trajectory depth for HalluWorld terminal runs.

Usage:
    python3 analyze_depth.py results/u_4omini/results.jsonl
    python3 analyze_depth.py results/*/results.jsonl --plot depth_plot.png

For each run this prints:
  * every record the harness wrote, split into scored and excluded (with the reason), plus how
    many responses hit the output-token cap (stop_reason max_tokens / length);
  * the hallucination rate as wrong/scored;
  * hallucination by depth quartile and by prompt-token quartile, with the point-biserial
    correlation (Pearson r between the depth value and a 0/1 "hallucinated" flag) and its n.

Depth ("steps_before_probe") is empty in terminal records, but the v0.1 raw question bank encodes
the trigger step in each filename, e.g. 000288_make-doom-for-mips_step-0214_....json. Download it
once with:
    huggingface-cli download DegenAI-Labs/HalluWorld --repo-type dataset --include "v0.1/raw/terminal/*"
or point --bank-glob at your own copy. The plot needs matplotlib; the tables do not.
"""
import argparse
import ast
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

DEFAULT_BANK_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--DegenAI-Labs--HalluWorld/snapshots/*/v0.1/raw/terminal/*_step-*.json")
TRUNCATION_STOP_REASONS = {"max_tokens", "length"}  # Anthropic, OpenAI


def load_steps(bank_glob):
    steps = {}
    for f in glob.glob(bank_glob):
        m = re.match(r"(\d{6})_.*_step-(\d+)_", os.path.basename(f))
        if m:
            steps[f"terminal/{m.group(1)}"] = int(m.group(2))
    return steps


def extra(record):
    e = record.get("extra") or {}
    if isinstance(e, str):  # older records stored extra as a repr string
        try:
            e = ast.literal_eval(e)
        except (ValueError, SyntaxError):
            e = {}
    return e


def load(path, steps):
    """All records, keeping the latest one per question (a --resume run appends records)."""
    latest = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("steps_before_probe") is None:
            r["steps_before_probe"] = steps.get(r["question_id"])
        r["stop_reason"] = extra(r).get("stop_reason")
        latest[r["question_id"]] = r
    return list(latest.values())


def is_scored(r):
    return r.get("score") is not None and r.get("is_correct") is not None


def wrong(rows):
    return sum(not r["is_correct"] for r in rows)


def quartile_bins(rows, key, n_bins=4):
    rows = sorted(rows, key=lambda r: r[key])
    size = len(rows) / n_bins
    return [rows[round(i * size): round((i + 1) * size)] for i in range(n_bins)]


def point_biserial(xs, flags):
    """Pearson correlation between a numeric x and a 0/1 flag (= point-biserial r)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(flags) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, flags))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in flags)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def report(path, steps):
    records = load(path, steps)
    scored = [r for r in records if is_scored(r)]
    excluded = Counter(r.get("error") or "unscored" for r in records if not is_scored(r))
    truncated = [r for r in records if r["stop_reason"] in TRUNCATION_STOP_REASONS]
    model = records[0].get("model") if records else Path(path).parent.name
    name = f"{model} [{Path(path).parent.name}]"

    print(f"\n=== {name} ===")
    print(f"records: {len(records)}   scored: {len(scored)}   excluded: {sum(excluded.values())}"
          + (f" ({', '.join(f'{k}: {v}' for k, v in excluded.items())})" if excluded else ""))
    print(f"hit output-token cap: {len(truncated)} "
          f"({sum(not is_scored(r) for r in truncated)} excluded, "
          f"{sum(is_scored(r) for r in truncated)} graded on a cut-off answer)")
    if not scored:
        print("no scored probes; nothing to analyse")
        return name, {}
    print(f"hallucination: {wrong(scored)}/{len(scored)} = {wrong(scored) / len(scored):.1%}")

    out = {}
    for key, label in [("steps_before_probe", "depth (commands before probe)"),
                       ("prompt_tokens", "prompt tokens")]:
        keyed = [r for r in scored if isinstance(r.get(key), (int, float))]
        if len(keyed) < 8:
            print(f"\nBy {label}: only {len(keyed)} scored probes have '{key}', skipping.")
            continue
        print(f"\nBy {label}:")
        if len(keyed) < len(scored):
            print(f"  (note: {len(scored) - len(keyed)} scored probes have no '{key}' and are left out)")
        print(f"  {'range':>18}  {'n':>3}  {'halluc':>7}  {'±95%':>6}")
        pts = []
        for b in quartile_bins(keyed, key):
            h = wrong(b) / len(b)
            moe = 1.96 * (h * (1 - h) / len(b)) ** 0.5
            print(f"  {f'{b[0][key]}–{b[-1][key]}':>18}  {len(b):>3}  {h:>7.1%}  {moe:>6.1%}")
            pts.append((sum(r[key] for r in b) / len(b), h))
        r = point_biserial([x[key] for x in keyed], [0 if x["is_correct"] else 1 for x in keyed])
        print(f"  point-biserial r with hallucinating: {r:+.2f} (n={len(keyed)})")
        out[key] = pts
    return name, out


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot. pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for name, out in results:
        for ax, key in zip(axes, ["steps_before_probe", "prompt_tokens"]):
            if key in out:
                xs, ys = zip(*out[key])
                ax.plot(xs, ys, marker="o", label=name)
    axes[0].set_xlabel("commands before probe (quartile mean)")
    axes[1].set_xlabel("prompt tokens (quartile mean)")
    axes[0].set_ylabel("hallucination rate")
    for ax in axes:
        ax.set_ylim(0, 1)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("HalluWorld v0.1 terminal: uncertainty probes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["results/u_4omini/results.jsonl"])
    ap.add_argument("--bank-glob", default=DEFAULT_BANK_GLOB, help="glob for the v0.1 raw terminal probes")
    ap.add_argument("--plot", default="depth_plot.png", help="where to save the plot ('' to skip)")
    args = ap.parse_args()

    steps = load_steps(args.bank_glob)
    if not steps:
        print("(note: v0.1 raw question bank not found, so depth is unavailable; see the module docstring)")
    results = [report(p, steps) for p in args.paths]
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
