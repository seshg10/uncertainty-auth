"""
compare_results.py — Compare and combine inference results from multiple model JSON files.

Usage:
    # Equal-weight average across two files
    python compare_results.py results.json results_gpt3.5.json

    # Custom weights (must sum to 1 or will be normalised)
    python compare_results.py results.json results_gpt3.5.json --weights 0.7 0.3

    # Different combination method
    python compare_results.py results.json results_gpt3.5.json --method majority_vote

    # Save plot to custom path
    python compare_results.py results.json results_gpt3.5.json --output comparison.png

Adding a new combination method:
    1. Subclass Combiner and implement combine()
    2. Register it in COMBINERS at the bottom of this file
    3. Pass its key to --method
"""

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = ["yes", "no", "maybe"]
LABEL_COLORS = {"yes": "#4CAF50", "no": "#F44336", "maybe": "#FF9800"}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    return token.strip().lower().rstrip(".")


def find_answer_token(token_logprobs: list[dict]) -> dict | None:
    for entry in token_logprobs:
        if normalize_token(entry["token"]) in {"yes", "no", "maybe"}:
            return entry
    return None


def extract_label_probs(token_logprobs: list[dict]) -> tuple[dict | None, str | None]:
    """
    Return (normalised label->prob dict, chosen label) from the answer token.
    Aggregates case variants (' Yes', 'yes', 'YES' -> 'yes') and normalises
    so that yes+no+maybe sum to 1.
    """
    entry = find_answer_token(token_logprobs)
    if entry is None:
        return None, None

    probs: dict[str, float] = {l: 0.0 for l in LABELS}
    for alt in entry["top_logprobs"]:
        label = normalize_token(alt["token"])
        if label in probs:
            probs[label] += alt["prob"]

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    chosen = normalize_token(entry["token"])
    return probs, chosen


# ---------------------------------------------------------------------------
# Combiners
# ---------------------------------------------------------------------------

class Combiner(ABC):
    """Base class for combination strategies.

    To add a new strategy:
      - subclass this, set a `name` attribute, implement combine()
      - register the class in COMBINERS below
    """
    name: str

    @abstractmethod
    def combine(self, model_probs: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Args:
            model_probs: {model_name: {label: prob}} — one normalised distribution per model
        Returns:
            Combined {label: prob} distribution (should sum to ~1)
        """


class WeightedAverageCombiner(Combiner):
    """Weighted average of label probability distributions across models."""
    name = "weighted_avg"

    def __init__(self, weights: dict[str, float] | None = None):
        # weights: {model_name: weight}. None = equal weights.
        self.weights = weights

    def combine(self, model_probs: dict[str, dict[str, float]]) -> dict[str, float]:
        models = list(model_probs.keys())
        w = [self.weights[m] if self.weights else 1.0 for m in models]
        total_w = sum(w)
        w = [wi / total_w for wi in w]

        combined = {l: 0.0 for l in LABELS}
        for weight, model in zip(w, models):
            for label in LABELS:
                combined[label] += weight * model_probs[model].get(label, 0.0)
        return combined


class MajorityVoteCombiner(Combiner):
    """Each model votes for its argmax label; vote share becomes the combined prob."""
    name = "majority_vote"

    def combine(self, model_probs: dict[str, dict[str, float]]) -> dict[str, float]:
        votes = {l: 0 for l in LABELS}
        for probs in model_probs.values():
            winner = max(probs, key=probs.get)
            votes[winner] += 1
        total = sum(votes.values())
        return {k: v / total for k, v in votes.items()}


class ConfidenceWeightedCombiner(Combiner):
    """Weight each model by its confidence (max prob) before averaging."""
    name = "confidence_weighted"

    def combine(self, model_probs: dict[str, dict[str, float]]) -> dict[str, float]:
        confidences = {m: max(p.values()) for m, p in model_probs.items()}
        total_conf = sum(confidences.values())

        combined = {l: 0.0 for l in LABELS}
        for model, probs in model_probs.items():
            w = confidences[model] / total_conf
            for label in LABELS:
                combined[label] += w * probs.get(label, 0.0)
        return combined


class EntropyWeightedCombiner(Combiner):
    """Weight each model inversely by the entropy of its answer distribution.

    Low-entropy (confident) models get higher weight; high-entropy (uncertain)
    models are down-weighted. Weight for model m:

        w_m  =  1 / (H(p_m) + eps)   then normalised to sum to 1

    where H(p) = -sum_i p_i * log(p_i) is the Shannon entropy.
    A model that assigns all mass to one label has H=0 and gets maximum weight.
    A uniform distribution over three labels has H=log(3) ≈ 1.1 and gets minimum weight.
    """
    name = "entropy_weighted"

    def combine(self, model_probs: dict[str, dict[str, float]]) -> dict[str, float]:
        import math

        def entropy(probs: dict[str, float]) -> float:
            return -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)

        entropies = {m: entropy(p) for m, p in model_probs.items()}
        inv_entropies = {m: 1.0 / (h + 1e-10) for m, h in entropies.items()}
        total = sum(inv_entropies.values())
        weights = {m: v / total for m, v in inv_entropies.items()}

        combined = {l: 0.0 for l in LABELS}
        for model, probs in model_probs.items():
            for label in LABELS:
                combined[label] += weights[model] * probs.get(label, 0.0)
        return combined


# Registry — add new combiners here
COMBINERS: dict[str, type[Combiner]] = {
    WeightedAverageCombiner.name:    WeightedAverageCombiner,
    MajorityVoteCombiner.name:       MajorityVoteCombiner,
    ConfidenceWeightedCombiner.name: ConfidenceWeightedCombiner,
    EntropyWeightedCombiner.name:    EntropyWeightedCombiner,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_file(path: Path) -> dict[str, dict]:
    with path.open() as f:
        records = json.load(f)
    return {str(r["pubid"]): r for r in records}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-model PubMedQA inference results")
    parser.add_argument("files", nargs="+", help="Two or more results JSON files")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="Per-file weights for weighted_avg (same order as files)")
    parser.add_argument("--method", default="weighted_avg", choices=list(COMBINERS.keys()),
                        help="Combination method")
    parser.add_argument("--output", default="plots/comparison_plot.png", help="Output plot path")
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.files]
    all_data = {p.stem: load_file(p) for p in file_paths}
    model_names = list(all_data.keys())

    # Weights
    weights: dict[str, float] | None = None
    if args.weights:
        if len(args.weights) != len(file_paths):
            raise ValueError(f"--weights count ({len(args.weights)}) must match file count ({len(file_paths)})")
        total = sum(args.weights)
        weights = {m: w / total for m, w in zip(model_names, args.weights)}

    # Build combiner
    CombinerClass = COMBINERS[args.method]
    combiner = CombinerClass(weights=weights) if args.method == "weighted_avg" else CombinerClass()

    # Find common pubids
    common_pubids = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    print(f"Files     : {[p.name for p in file_paths]}")
    print(f"Method    : {args.method}" + (f"  weights={weights}" if weights else "  (equal weights)"))
    print(f"Examples  : {len(common_pubids)} common across all files\n")

    results = []
    for pubid in common_pubids:
        ground_truth = None
        model_probs: dict[str, dict[str, float]] = {}
        model_predictions: dict[str, str] = {}

        for model_name, data in all_data.items():
            if pubid not in data:
                continue
            record = data[pubid]
            if ground_truth is None:
                ground_truth = record["ground_truth"].lower()
            probs, chosen = extract_label_probs(record["token_logprobs"])
            if probs is None:
                print(f"  [warn] no answer token for pubid={pubid} in {model_name}")
                continue
            model_probs[model_name] = probs
            model_predictions[model_name] = chosen

        if not model_probs:
            continue

        combined = combiner.combine(model_probs)
        combined_pred = max(combined, key=combined.get)

        results.append({
            "pubid": pubid,
            "ground_truth": ground_truth,
            "combined_prediction": combined_pred,
            "combined_probs": combined,
            "model_probs": model_probs,
            "model_predictions": model_predictions,
            "correct": combined_pred == ground_truth,
        })

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("Per-model accuracy:")
    for model in model_names:
        n_correct = sum(
            1 for r in results
            if r["model_predictions"].get(model) == r["ground_truth"]
        )
        print(f"  {model:<30} {n_correct}/{len(results)} = {n_correct/len(results):.0%}")

    n_combined = sum(r["correct"] for r in results)
    print(f"\n  {'combined (' + args.method + ')':<30} {n_combined}/{len(results)} = {n_combined/len(results):.0%}")

    print("\nPer-example breakdown:")
    header = f"  {'pubid':<14} {'truth':<7} {'combined':<10} {'ok':<4}"
    for m in model_names:
        header += f"  {m:<22}"
    print(header)
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        line = f"  {r['pubid']:<14} {r['ground_truth']:<7} {r['combined_prediction']:<10} {mark:<4}"
        for m in model_names:
            pred = r["model_predictions"].get(m, "?")
            prob = r["model_probs"].get(m, {}).get(pred, 0.0)
            line += f"  {pred} (p={prob:.3f}){'':<10}"
        print(line)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    plot_overview(results, model_names, args.method, weights, out_path)
    plot_calibration(results, model_names, args.method,
                     out_path.with_name(out_path.stem + "_calibration" + out_path.suffix))


# ---------------------------------------------------------------------------
# Plot 1 — Overview: accuracy, confusion matrices, per-example heatmap
# ---------------------------------------------------------------------------

def plot_overview(
    results: list[dict],
    model_names: list[str],
    method: str,
    weights: dict | None,
    out_path: Path,
) -> None:
    import matplotlib.gridspec as gridspec

    all_rows = model_names + [f"combined ({method})"]
    n_models = len(all_rows)
    n = len(results)
    n_cols = n_models  # one confusion matrix column per row label

    fig = plt.figure(figsize=(max(5 * n_cols, 12), 14))
    weight_str = f"  weights={[round(v,2) for v in weights.values()]}" if weights else ""
    n_correct = sum(r["correct"] for r in results)
    fig.suptitle(
        f"Multi-model comparison — {method}{weight_str}   |   n={n}\n"
        f"Combined accuracy: {n_correct}/{n} = {n_correct/n:.0%}",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 3, 3], hspace=0.45)

    # ---- Row 0: Accuracy bar chart ----------------------------------------
    ax_acc = fig.add_subplot(gs[0])
    accuracies, labels_acc, colors_acc = [], [], []
    for model in model_names:
        acc = sum(
            1 for r in results if r["model_predictions"].get(model) == r["ground_truth"]
        ) / n
        accuracies.append(acc)
        labels_acc.append(model)
        colors_acc.append("#5C85D6")
    # combined
    accuracies.append(n_correct / n)
    labels_acc.append(f"combined\n({method})")
    colors_acc.append("#E07B39")

    bars = ax_acc.bar(labels_acc, accuracies, color=colors_acc, edgecolor="white", width=0.5)
    ax_acc.set_ylim(0, 1.15)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy per model vs combined")
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, acc in zip(bars, accuracies):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ---- Row 1: Confusion matrices ----------------------------------------
    gs_cm = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs[1], wspace=0.4)
    active_labels = sorted({r["ground_truth"] for r in results} |
                            {r["combined_prediction"] for r in results} |
                            {p for r in results for p in r["model_predictions"].values()})
    active_labels = [l for l in LABELS if l in active_labels]  # keep canonical order

    for col_idx, row_label in enumerate(all_rows):
        ax_cm = fig.add_subplot(gs_cm[col_idx])
        is_combined = col_idx == len(model_names)

        cm = np.zeros((len(active_labels), len(active_labels)), dtype=int)
        for r in results:
            pred = r["combined_prediction"] if is_combined else r["model_predictions"].get(row_label)
            truth = r["ground_truth"]
            if pred and truth in active_labels and pred in active_labels:
                i = active_labels.index(truth)
                j = active_labels.index(pred)
                cm[i, j] += 1

        im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        ax_cm.set_xticks(range(len(active_labels)))
        ax_cm.set_yticks(range(len(active_labels)))
        ax_cm.set_xticklabels(active_labels, fontsize=8)
        ax_cm.set_yticklabels(active_labels, fontsize=8)
        ax_cm.set_xlabel("Predicted", fontsize=8)
        if col_idx == 0:
            ax_cm.set_ylabel("Actual", fontsize=8)
        ax_cm.set_title(row_label, fontsize=8, fontweight="bold",
                        color="#E07B39" if is_combined else "#333333")
        for i in range(len(active_labels)):
            for j in range(len(active_labels)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=10, color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ---- Row 2: Per-example correctness heatmap ---------------------------
    # Sort examples by ground truth then by combined confidence (descending)
    label_order_map = {l: i for i, l in enumerate(active_labels)}
    sorted_results = sorted(
        results,
        key=lambda r: (label_order_map.get(r["ground_truth"], 99),
                       -r["combined_probs"].get(r["combined_prediction"], 0))
    )

    gs_hm = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2])
    ax_hm = fig.add_subplot(gs_hm[0])

    # Build RGBA heatmap: rows = models + combined, cols = examples
    # Colour: green if correct, red if incorrect; alpha = confidence
    heatmap = np.zeros((n_models, n, 4))  # RGBA
    GREEN = np.array([76, 175, 80]) / 255
    RED   = np.array([244, 67, 54]) / 255

    for col_idx, r in enumerate(sorted_results):
        for row_idx, row_label in enumerate(all_rows):
            is_combined = row_idx == len(model_names)
            if is_combined:
                pred = r["combined_prediction"]
                conf = r["combined_probs"].get(pred, 0.0)
                correct = r["correct"]
            else:
                pred = r["model_predictions"].get(row_label)
                conf = r["model_probs"].get(row_label, {}).get(pred, 0.0) if pred else 0.0
                correct = pred == r["ground_truth"] if pred else False

            color = GREEN if correct else RED
            alpha = 0.25 + 0.75 * conf  # min alpha 0.25 so low-confidence still visible
            heatmap[row_idx, col_idx, :3] = color
            heatmap[row_idx, col_idx, 3]  = alpha

    ax_hm.imshow(heatmap, aspect="auto", interpolation="nearest")
    ax_hm.set_yticks(range(n_models))
    ax_hm.set_yticklabels(all_rows, fontsize=8)
    ax_hm.set_xlabel(
        f"Examples (n={n}, sorted by ground truth then combined confidence)\n"
        "Green = correct, Red = incorrect   |   Colour intensity = model confidence",
        fontsize=8,
    )
    ax_hm.set_title("Per-example correctness heatmap", fontsize=9, fontweight="bold")

    # Vertical separator lines between ground-truth groups
    prev_label, col_idx = sorted_results[0]["ground_truth"], 0
    for col_idx, r in enumerate(sorted_results):
        if r["ground_truth"] != prev_label:
            ax_hm.axvline(col_idx - 0.5, color="white", linewidth=1.5, alpha=0.8)
            ax_hm.text(col_idx - 1, -0.7, prev_label, ha="right", fontsize=7,
                       color="gray", transform=ax_hm.get_xaxis_transform())
            prev_label = r["ground_truth"]
    ax_hm.text(col_idx, -0.7, prev_label, ha="right", fontsize=7,
               color="gray", transform=ax_hm.get_xaxis_transform())

    # Only show x tick labels if N is small enough
    if n <= 40:
        ax_hm.set_xticks(range(n))
        ax_hm.set_xticklabels([r["pubid"] for r in sorted_results],
                               rotation=90, fontsize=5)
    else:
        ax_hm.set_xticks([])

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Overview plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Calibration: reliability curves + confidence histograms
# ---------------------------------------------------------------------------

def plot_calibration(
    results: list[dict],
    model_names: list[str],
    method: str,
    out_path: Path,
) -> None:
    all_rows = model_names + [f"combined ({method})"]
    n_cols = len(all_rows)
    n_bins = 10

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    if n_cols == 1:
        axes = [[axes[0][0]], [axes[1][0]]]
    fig.suptitle(
        f"Calibration & Confidence Distribution — {method}",
        fontsize=13, fontweight="bold",
    )

    for col_idx, row_label in enumerate(all_rows):
        is_combined = col_idx == len(model_names)
        confidences, corrects = [], []

        for r in results:
            if is_combined:
                pred = r["combined_prediction"]
                conf = r["combined_probs"].get(pred, 0.0)
                correct = r["correct"]
            else:
                pred = r["model_predictions"].get(row_label)
                if pred is None:
                    continue
                conf = r["model_probs"].get(row_label, {}).get(pred, 0.0)
                correct = pred == r["ground_truth"]
            confidences.append(conf)
            corrects.append(int(correct))

        confidences = np.array(confidences)
        corrects    = np.array(corrects)

        # ---- Calibration curve (reliability diagram) ----
        ax_cal = axes[0][col_idx]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append(corrects[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect calibration")
        sc = ax_cal.scatter(bin_confs, bin_accs, c=bin_counts, cmap="YlOrRd",
                            s=80, zorder=5, edgecolors="black", linewidths=0.5)
        ax_cal.plot(bin_confs, bin_accs, color="#5C85D6", linewidth=1.5)
        plt.colorbar(sc, ax=ax_cal, label="# examples in bin", pad=0.02)
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.set_xlabel("Mean predicted confidence", fontsize=8)
        if col_idx == 0:
            ax_cal.set_ylabel("Actual accuracy", fontsize=8)
        ax_cal.set_title(row_label, fontsize=9, fontweight="bold",
                         color="#E07B39" if is_combined else "#333333")
        ax_cal.legend(fontsize=7)
        ax_cal.grid(True, alpha=0.3)

        # ECE annotation
        if bin_confs:
            ece = sum(abs(a - c) * cnt for a, c, cnt in zip(bin_accs, bin_confs, bin_counts))
            ece /= len(confidences)
            ax_cal.text(0.05, 0.92, f"ECE={ece:.3f}", transform=ax_cal.transAxes,
                        fontsize=8, color="#C62828", fontweight="bold")

        # ---- Confidence histogram ----
        ax_hist = axes[1][col_idx]
        correct_conf   = confidences[corrects == 1]
        incorrect_conf = confidences[corrects == 0]
        bins = np.linspace(0, 1, 21)
        ax_hist.hist(correct_conf,   bins=bins, alpha=0.65, color="#4CAF50",
                     label=f"correct (n={len(correct_conf)})",   edgecolor="white")
        ax_hist.hist(incorrect_conf, bins=bins, alpha=0.65, color="#F44336",
                     label=f"incorrect (n={len(incorrect_conf)})", edgecolor="white")
        ax_hist.set_xlabel("Predicted confidence", fontsize=8)
        if col_idx == 0:
            ax_hist.set_ylabel("Count", fontsize=8)
        ax_hist.set_title(f"Confidence distribution\n{row_label}", fontsize=8)
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Calibration plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
