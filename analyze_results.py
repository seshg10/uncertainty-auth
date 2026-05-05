"""
analyze_results.py — Compare predicted vs actual labels from one or more
inference JSON files produced by run_inference.py.

Single file:
  - Detailed per-example top_logprobs bars (N ≤ 20)
  - Aggregate view: confusion matrix + per-example correctness strip (N > 20)

Multiple files:
  - Accuracy bar chart per model
  - Confusion matrix per model
  - Per-example correctness heatmap (rows = models, cols = examples sorted by
    ground truth), colour-coded by correct/incorrect and confidence

Usage:
    python analyze_results.py results.json
    python analyze_results.py results_gpt3.5.json results_gpt4o.json
    python analyze_results.py results*.json --output my_plot.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ANSWER_LABELS = {"yes", "no", "maybe"}
TOKEN_PALETTE  = {"yes": "#4CAF50", "no": "#F44336", "maybe": "#FF9800"}
DETAIL_THRESHOLD = 20   # switch to aggregate view above this many examples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    return token.strip().lower().rstrip(".")


def find_answer_token(token_logprobs: list[dict]) -> dict | None:
    for entry in token_logprobs:
        if normalize_token(entry["token"]) in ANSWER_LABELS:
            return entry
    return None


def predict_from_text(generated_text: str) -> str | None:
    """Parse 'Answer: yes/no/maybe' for models that return no logprobs."""
    import re
    m = re.search(r"\bAnswer\s*:\s*(yes|no|maybe)\b", generated_text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b(yes|no|maybe)\b", generated_text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def parse_records(records: list[dict]) -> dict:
    """
    Returns dict keyed by pubid with fields:
      ground_truth, predicted, correct, prob (or None), answer_entry (or None)
    """
    out = {}
    for r in records:
        pubid = str(r["pubid"])
        truth = r["ground_truth"].lower()

        token_lp = r.get("token_logprobs") or []
        entry    = find_answer_token(token_lp) if token_lp else None

        if entry is not None:
            predicted = normalize_token(entry["token"])
            prob      = entry["prob"]
        else:
            predicted = predict_from_text(r.get("generated_text", "") or "")
            prob      = None

        if predicted is None:
            continue

        out[pubid] = {
            "ground_truth": truth,
            "predicted":    predicted,
            "correct":      predicted == truth,
            "prob":         prob,
            "answer_entry": entry,
        }
    return out


# ---------------------------------------------------------------------------
# Single-model plots
# ---------------------------------------------------------------------------

def plot_single_detail(parsed: dict, model_name: str, out_path: Path) -> None:
    """Per-example top_logprobs bars for small N."""
    items  = list(parsed.values())
    pubids = list(parsed.keys())
    n      = len(items)
    correct_count = sum(it["correct"] for it in items)
    accuracy = correct_count / n

    fig, axes = plt.subplots(2, n, figsize=(max(4 * n, 8), 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    fig.suptitle(
        f"{model_name}  |  Accuracy: {accuracy:.0%} ({correct_count}/{n})",
        fontsize=13, fontweight="bold",
    )

    for j, (pubid, it) in enumerate(parsed.items()):
        entry = it["answer_entry"]
        top   = entry["top_logprobs"] if entry else []

        tokens     = [t["token"] for t in top]
        probs      = [t["prob"]  for t in top]
        bar_colors = [TOKEN_PALETTE.get(normalize_token(tok), "#9E9E9E") for tok in tokens]

        ax = axes[0][j]
        bars = ax.barh(range(len(tokens)), probs, color=bar_colors, edgecolor="white")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([repr(t) for t in tokens], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=8)
        if entry:
            ax.axvline(entry["prob"], color="black", linestyle="--", linewidth=1,
                       label=f"chosen={entry['prob']:.3f}")
            ax.legend(fontsize=7)

        match = it["correct"]
        color = "#2E7D32" if match else "#C62828"
        ax.set_title(
            f"{'✓' if match else '✗'} pubid {pubid}\n"
            f"Pred: {it['predicted']}  Actual: {it['ground_truth']}",
            fontsize=9, color=color, fontweight="bold",
        )
        for bar, prob in zip(bars, probs):
            ax.text(min(prob + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{prob:.4f}", va="center", fontsize=7)

        ax_strip = axes[1][j]
        ax_strip.bar([0], [1], color="#4CAF50" if match else "#F44336", width=0.6, alpha=0.85)
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        ax_strip.set_title("Correct" if match else "Incorrect", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def plot_single_aggregate(parsed: dict, model_name: str, out_path: Path) -> None:
    """Aggregate view for larger N: confusion matrix + correctness strip + prob scatter."""
    pubids = list(parsed.keys())
    items  = list(parsed.values())
    n      = len(items)
    correct_count = sum(it["correct"] for it in items)
    accuracy = correct_count / n

    truths = [it["ground_truth"] for it in items]
    preds  = [it["predicted"]    for it in items]
    probs  = [it["prob"]         for it in items]
    has_probs = any(p is not None for p in probs)

    active_labels = sorted(set(truths) | set(preds))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{model_name}  |  Accuracy: {accuracy:.0%} ({correct_count}/{n} examples)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # --- Confusion matrix ---
    ax_cm = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(truths, preds, labels=active_labels)
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    ax_cm.set_xticks(range(len(active_labels)))
    ax_cm.set_yticks(range(len(active_labels)))
    ax_cm.set_xticklabels(active_labels, fontsize=9)
    ax_cm.set_yticklabels(active_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted", fontsize=9)
    ax_cm.set_ylabel("Actual", fontsize=9)
    ax_cm.set_title("Confusion matrix")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # --- Per-class accuracy bar ---
    ax_pc = fig.add_subplot(gs[0, 1])
    per_class = {}
    for lbl in active_labels:
        idxs = [i for i, t in enumerate(truths) if t == lbl]
        per_class[lbl] = sum(1 for i in idxs if items[i]["correct"]) / len(idxs) if idxs else 0
    ax_pc.bar(active_labels, [per_class[l] for l in active_labels],
              color=[TOKEN_PALETTE.get(l, "#9E9E9E") for l in active_labels],
              edgecolor="white", width=0.5)
    ax_pc.set_ylim(0, 1.2)
    ax_pc.set_ylabel("Accuracy")
    ax_pc.set_title("Per-class accuracy")
    ax_pc.axhline(accuracy, color="black", linestyle="--", linewidth=1, label=f"overall {accuracy:.0%}")
    ax_pc.legend(fontsize=8)
    for i, lbl in enumerate(active_labels):
        ax_pc.text(i, per_class[lbl] + 0.03, f"{per_class[lbl]:.0%}", ha="center", fontsize=9)

    # --- Probability scatter (if logprobs available) ---
    ax_sc = fig.add_subplot(gs[0, 2])
    if has_probs:
        # Sort by ground truth then by prob
        order = sorted(range(n), key=lambda i: (truths[i], -(probs[i] or 0)))
        sc_probs  = [probs[i] or 0 for i in order]
        sc_colors = ["#4CAF50" if items[i]["correct"] else "#F44336" for i in order]
        ax_sc.scatter(range(n), sc_probs, c=sc_colors, s=8, alpha=0.7)
        ax_sc.set_xlabel("Example (sorted by ground truth)", fontsize=8)
        ax_sc.set_ylabel("Prob of chosen token", fontsize=8)
        ax_sc.set_title("Confidence — green=correct, red=incorrect")
        ax_sc.set_ylim(0, 1.05)
        ax_sc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    else:
        ax_sc.text(0.5, 0.5, "No logprobs available\n(text prediction only)",
                   ha="center", va="center", transform=ax_sc.transAxes, fontsize=10)
        ax_sc.set_title("Confidence")

    # --- Correctness heatmap (bottom row, full width) ---
    ax_hm = fig.add_subplot(gs[1, :])
    order = sorted(range(n), key=lambda i: (truths[i], -(probs[i] or 0) if has_probs else 0))
    heat = np.zeros((1, n))
    for col, i in enumerate(order):
        heat[0, col] = (probs[i] or 0.5) if items[i]["correct"] else -(probs[i] or 0.5)

    ax_hm.imshow(heat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_yticks([0])
    ax_hm.set_yticklabels([model_name[:20]], fontsize=8)
    ax_hm.set_xlabel("Examples sorted by ground truth", fontsize=8)
    ax_hm.set_title("Correctness heatmap — green=correct, red=incorrect, intensity=confidence")

    # Boundary lines between ground-truth classes
    truth_sorted = [truths[i] for i in order]
    for x in range(1, n):
        if truth_sorted[x] != truth_sorted[x - 1]:
            ax_hm.axvline(x - 0.5, color="white", linewidth=1.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Multi-model plot
# ---------------------------------------------------------------------------

def plot_multi(all_parsed: dict[str, dict], common_pubids: list[str], out_path: Path) -> None:
    model_names = list(all_parsed.keys())
    n_models    = len(model_names)
    n           = len(common_pubids)

    # Build aligned arrays
    truths  = [all_parsed[model_names[0]][pid]["ground_truth"] for pid in common_pubids]
    active_labels = sorted(set(truths))

    accuracies = {}
    for model in model_names:
        parsed = all_parsed[model]
        accuracies[model] = sum(parsed[pid]["correct"] for pid in common_pubids) / n

    fig = plt.figure(figsize=(18, 5 + 2 * n_models))
    fig.suptitle(
        f"Multi-model comparison  |  {n} common examples",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2 + n_models, n_models + 1,
                           figure=fig, hspace=0.55, wspace=0.4)

    # --- Row 0: Accuracy bars ---
    ax_acc = fig.add_subplot(gs[0, :])
    bar_colors = ["#5C85D6"] * n_models
    best_idx = int(np.argmax(list(accuracies.values())))
    bar_colors[best_idx] = "#E07B39"
    bars = ax_acc.bar(model_names, list(accuracies.values()),
                      color=bar_colors, edgecolor="white", width=0.5)
    ax_acc.set_ylim(0, 1.2)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Test accuracy (orange = best)")
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, acc in zip(bars, accuracies.values()):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.0%}", ha="center", fontsize=9, fontweight="bold")

    # --- Row 1: Confusion matrix per model ---
    for k, model in enumerate(model_names):
        ax_cm = fig.add_subplot(gs[1, k])
        preds  = [all_parsed[model][pid]["predicted"] for pid in common_pubids]
        cm     = confusion_matrix(truths, preds, labels=active_labels)
        im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        ax_cm.set_xticks(range(len(active_labels)))
        ax_cm.set_yticks(range(len(active_labels)))
        ax_cm.set_xticklabels(active_labels, fontsize=8)
        ax_cm.set_yticklabels(active_labels, fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=8)
        ax_cm.set_ylabel("Actual", fontsize=8)
        ax_cm.set_title(f"{model[:20]}\n{accuracies[model]:.0%}", fontsize=8)
        for i in range(len(active_labels)):
            for j in range(len(active_labels)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                           color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # --- Rows 2+: Per-model correctness heatmap rows ---
    # Sort examples by ground truth then best-model confidence
    best_model = model_names[best_idx]
    order = sorted(
        range(n),
        key=lambda i: (
            truths[i],
            -(all_parsed[best_model][common_pubids[i]]["prob"] or 0),
        ),
    )
    truth_sorted = [truths[i] for i in order]

    ax_hm = fig.add_subplot(gs[2:, :])
    heat = np.zeros((n_models, n))
    for row, model in enumerate(model_names):
        parsed = all_parsed[model]
        for col, i in enumerate(order):
            pid  = common_pubids[i]
            it   = parsed[pid]
            conf = it["prob"] if it["prob"] is not None else 0.5
            heat[row, col] = conf if it["correct"] else -conf

    im_hm = ax_hm.imshow(heat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_yticks(range(n_models))
    ax_hm.set_yticklabels([m[:22] for m in model_names], fontsize=8)
    ax_hm.set_xlabel("Examples sorted by ground truth", fontsize=8)
    ax_hm.set_title("Correctness heatmap — green=correct, red=incorrect, intensity=confidence")
    plt.colorbar(im_hm, ax=ax_hm, fraction=0.02, pad=0.01,
                 label="confidence (positive=correct)")

    for x in range(1, n):
        if truth_sorted[x] != truth_sorted[x - 1]:
            ax_hm.axvline(x - 0.5, color="white", linewidth=1.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and compare inference results")
    parser.add_argument("files", nargs="+", help="One or more results JSON files")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: auto-named from input files)")
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.files]

    # Load and parse each file
    all_parsed: dict[str, dict] = {}
    for path in file_paths:
        records = load_records(path)
        parsed  = parse_records(records)
        all_parsed[path.stem] = parsed
        n_correct = sum(it["correct"] for it in parsed.values())
        has_lp    = sum(1 for it in parsed.values() if it["prob"] is not None)
        print(f"{path.name}: {len(parsed)} examples  "
              f"accuracy={n_correct/len(parsed):.0%}  "
              f"logprobs={'yes' if has_lp else 'no'} ({has_lp}/{len(parsed)})")

    # Single-file path
    if len(file_paths) == 1:
        stem   = file_paths[0].stem
        parsed = all_parsed[stem]
        out    = Path(args.output) if args.output else Path("plots") / f"{stem}_plot.png"
        n      = len(parsed)
        print(f"\n{n} parseable examples — using {'detail' if n <= DETAIL_THRESHOLD else 'aggregate'} view")
        if n <= DETAIL_THRESHOLD:
            plot_single_detail(parsed, stem, out)
        else:
            plot_single_aggregate(parsed, stem, out)
        return

    # Multi-file: restrict to common pubids
    common = sorted(
        set.intersection(*[set(p.keys()) for p in all_parsed.values()])
    )
    print(f"\n{len(common)} examples common to all {len(file_paths)} files")
    if not common:
        print("No common pubids — cannot plot.")
        return

    out = Path(args.output) if args.output else Path("plots/comparison_plot_analyze.png")
    plot_multi(all_parsed, common, out)


if __name__ == "__main__":
    main()
