"""
learned_model.py — Train classifiers on answer-token features extracted from
multiple model inference files, then evaluate on a held-out test set.

Features per model (per example):
  - One-hot of predicted label  (yes / no / maybe)
  - logprob of the chosen answer token
  - prob   of the chosen answer token
  - Aggregated yes / no / maybe probabilities from top_logprobs
  - Shannon entropy of the label distribution

Classifiers trained:
  Logistic Regression, Random Forest, Gradient Boosting, SVM

Individual model accuracies are shown as baselines.

--compare lets you pass additional results files from models that do NOT return
logprobs (e.g. o1, o3, gpt-5.5). These are never used as classifier features
but their raw prediction accuracy is shown in the plot for comparison.

Usage:
    python learned_model.py results.json result_gpt4o.json
    python learned_model.py results.json result_gpt4o.json --seed 7 --train-size 50
    python learned_model.py results.json result_gpt4o.json --compare results_gpt5.5.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

LABELS = ["yes", "no", "maybe"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    return token.strip().lower().rstrip(".")


def find_answer_token(token_logprobs: list[dict]) -> dict | None:
    for entry in token_logprobs:
        if normalize_token(entry["token"]) in {"yes", "no", "maybe"}:
            return entry
    return None


def extract_answer(token_logprobs: list[dict]) -> tuple[str, float, float, dict] | None:
    """Return (chosen_label, logprob, prob, normalised_label_probs) or None."""
    entry = find_answer_token(token_logprobs)
    if entry is None:
        return None

    raw: dict[str, float] = {l: 0.0 for l in LABELS}
    for alt in entry["top_logprobs"]:
        lbl = normalize_token(alt["token"])
        if lbl in raw:
            raw[lbl] += alt["prob"]

    total = sum(raw.values())
    norm = {k: v / total for k, v in raw.items()} if total > 0 else raw.copy()
    chosen = normalize_token(entry["token"])
    return chosen, entry["logprob"], entry["prob"], norm


def shannon_entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)


def build_feature_vector(records_by_model: dict[str, dict], model_names: list[str]) -> np.ndarray | None:
    """
    Build a 1-D feature vector for one example.

    For each model the vector contains (in order):
      pred_yes, pred_no, pred_maybe   — one-hot of predicted label
      logprob, prob                   — confidence of chosen token
      yes_prob, no_prob, maybe_prob   — full normalised distribution
      entropy                         — uncertainty of the distribution
    """
    vec = []
    for model in model_names:
        result = extract_answer(records_by_model[model]["token_logprobs"])
        if result is None:
            return None
        chosen, logprob, prob, norm_probs = result

        # One-hot of prediction
        for lbl in LABELS:
            vec.append(1.0 if chosen == lbl else 0.0)

        # Token confidence
        vec.append(logprob)
        vec.append(prob)

        # Full label distribution
        for lbl in LABELS:
            vec.append(norm_probs[lbl])

        # Entropy
        vec.append(shannon_entropy(norm_probs))

    return np.array(vec, dtype=float)


def feature_names_for(model_names: list[str]) -> list[str]:
    names = []
    for m in model_names:
        tag = m[:14]
        for lbl in LABELS:
            names.append(f"{tag}:pred_{lbl}")
        names.append(f"{tag}:logprob")
        names.append(f"{tag}:prob")
        for lbl in LABELS:
            names.append(f"{tag}:{lbl}_prob")
        names.append(f"{tag}:entropy")
    return names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_file(path: Path) -> dict[str, dict]:
    with path.open() as f:
        records = json.load(f)
    return {str(r["pubid"]): r for r in records}


import re as _re

def predict_from_text(generated_text: str) -> str | None:
    """Parse 'Answer: yes/no/maybe' from generated text (for no-logprob models)."""
    m = _re.search(r"\bAnswer\s*:\s*(yes|no|maybe)\b", generated_text, _re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # fallback: first standalone yes/no/maybe in the text
    m = _re.search(r"\b(yes|no|maybe)\b", generated_text, _re.IGNORECASE)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifiers on multi-model logprob features")
    parser.add_argument("files", nargs="+", help="Two or more results JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-size", type=int, default=50, help="Training set size")
    parser.add_argument("--output", default="plots/learned_model_plot.png", help="Output plot path")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Results files for no-logprob models (shown in plot only, not used as features)")
    args = parser.parse_args()

    file_paths   = [Path(f) for f in args.files]
    all_data     = {p.stem: load_file(p) for p in file_paths}
    model_names  = list(all_data.keys())

    # Common pubids
    common = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    print(f"Files       : {[p.name for p in file_paths]}")
    print(f"Common      : {len(common)} examples")

    # Build full dataset
    X_list, y_list, pubids = [], [], []
    for pubid in common:
        records_by_model = {m: all_data[m][pubid] for m in model_names}
        truth = all_data[model_names[0]][pubid]["ground_truth"].lower()
        vec   = build_feature_vector(records_by_model, model_names)
        if vec is None:
            print(f"  [warn] skipping pubid={pubid} — missing answer token")
            continue
        X_list.append(vec)
        y_list.append(truth)
        pubids.append(pubid)

    X = np.array(X_list)
    y = np.array(y_list)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Dataset     : {len(X)} usable examples, classes={list(le.classes_)}")

    # Train / test split
    rng = random.Random(args.seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_train    = min(args.train_size, len(idx) - 1)
    train_idx  = np.array(idx[:n_train])
    test_idx   = np.array(idx[n_train:])

    X_train, y_train = X[train_idx], y_enc[train_idx]
    X_test,  y_test  = X[test_idx],  y_enc[test_idx]
    y_test_str        = y[test_idx]

    print(f"Split       : {len(train_idx)} train / {len(test_idx)} test  (seed={args.seed})\n")

    # ------------------------------------------------------------------
    # Individual model baselines on the test set
    # ------------------------------------------------------------------
    baselines: dict[str, float] = {}
    for model in model_names:
        n_correct = sum(
            1 for i in test_idx
            if (extract_answer(all_data[model][pubids[i]]["token_logprobs"]) or [None])[0]
            == y[i]
        )
        baselines[model] = n_correct / len(test_idx)

    # ------------------------------------------------------------------
    # Compare-only models (no logprobs — accuracy shown in plot only)
    # ------------------------------------------------------------------
    compare_baselines: dict[str, float] = {}
    for cmp_path in [Path(f) for f in args.compare]:
        cmp_data  = load_file(cmp_path)
        cmp_name  = cmp_path.stem
        test_pubs = [pubids[i] for i in test_idx]
        available = [pid for pid in test_pubs if pid in cmp_data]
        if not available:
            print(f"[warn] --compare file {cmp_path.name} has no overlap with test set; skipping")
            continue
        n_correct = sum(
            1 for pid in available
            if predict_from_text(cmp_data[pid].get("generated_text", ""))
            == all_data[model_names[0]][pid]["ground_truth"].lower()
        )
        compare_baselines[cmp_name] = n_correct / len(available)
        print(f"Compare     : {cmp_name}  accuracy = {compare_baselines[cmp_name]:.0%}"
              f"  ({n_correct}/{len(available)} examples, no logprobs)")

    # ------------------------------------------------------------------
    # Train classifiers
    # ------------------------------------------------------------------
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed),
        "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=args.seed),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                           max_depth=3, random_state=args.seed),
        "SVM (RBF)":           SVC(kernel="rbf", C=1.0, probability=True, random_state=args.seed),
    }

    clf_results: dict[str, dict] = {}
    feat_importances: dict[str, np.ndarray] = {}
    feat_names = feature_names_for(model_names)

    print("=" * 60)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred     = clf.predict(X_test)
        y_pred_str = le.inverse_transform(y_pred)
        acc        = accuracy_score(y_test, y_pred)

        clf_results[name] = {
            "acc":      acc,
            "y_pred":   y_pred,
            "y_pred_str": y_pred_str,
        }

        if hasattr(clf, "feature_importances_"):
            feat_importances[name] = clf.feature_importances_

        print(f"\n{name}   accuracy = {acc:.0%}  ({int(acc*len(test_idx))}/{len(test_idx)})")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    best_name = max(clf_results, key=lambda k: clf_results[k]["acc"])
    print("=" * 60)
    print(f"Best classifier : {best_name}  ({clf_results[best_name]['acc']:.0%})")

    # Per-example breakdown for best classifier
    print(f"\nPer-example breakdown ({best_name}):")
    print(f"  {'pubid':<14} {'truth':<8} {'predicted':<10} {'ok'}")
    for i, test_i in enumerate(test_idx):
        truth = y[test_i]
        pred  = clf_results[best_name]["y_pred_str"][i]
        mark  = "✓" if pred == truth else "✗"
        print(f"  {pubids[test_i]:<14} {truth:<8} {pred:<10} {mark}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot(
        clf_results=clf_results,
        baselines=baselines,
        compare_baselines=compare_baselines,
        le=le,
        y_test=y_test,
        feat_importances=feat_importances,
        feat_names=feat_names,
        out_path=Path(args.output),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    clf_results: dict,
    baselines: dict,
    compare_baselines: dict,
    le: LabelEncoder,
    y_test: np.ndarray,
    feat_importances: dict,
    feat_names: list[str],
    out_path: Path,
) -> None:
    active_labels = list(le.classes_)
    best_name     = max(clf_results, key=lambda k: clf_results[k]["acc"])
    n_test        = len(y_test)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Learned Classifier — PubMedQA Multi-model Token Features",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ---- Panel 1: Accuracy bar chart (baselines + classifiers) ----------
    ax_acc = fig.add_subplot(gs[0, 0])

    bar_names  = list(baselines.keys()) + list(clf_results.keys()) + list(compare_baselines.keys())
    bar_accs   = (list(baselines.values()) + [r["acc"] for r in clf_results.values()]
                  + list(compare_baselines.values()))
    bar_colors = (["#9E9E9E"] * len(baselines) + ["#5C85D6"] * len(clf_results)
                  + ["#26A69A"] * len(compare_baselines))
    # Highlight best classifier
    best_offset = len(baselines) + list(clf_results.keys()).index(best_name)
    bar_colors[best_offset] = "#E07B39"

    bars = ax_acc.bar(range(len(bar_names)), bar_accs, color=bar_colors,
                      edgecolor="white", width=0.6)
    ax_acc.set_xticks(range(len(bar_names)))
    ax_acc.set_xticklabels(bar_names, rotation=25, ha="right", fontsize=7)
    ax_acc.set_ylim(0, 1.2)
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Test accuracy\n(grey=baseline, blue=classifier, orange=best, teal=no-logprob)")
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, acc in zip(bars, bar_accs):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                    f"{acc:.0%}", ha="center", fontsize=8, fontweight="bold")
    if compare_baselines:
        from matplotlib.patches import Patch
        ax_acc.legend(handles=[Patch(color="#26A69A", label="no logprobs — comparison only")],
                      fontsize=7, loc="upper left")

    # ---- Panel 2: Confusion matrix — best classifier --------------------
    ax_cm = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(y_test, clf_results[best_name]["y_pred"])
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    ax_cm.set_xticks(range(len(active_labels)))
    ax_cm.set_yticks(range(len(active_labels)))
    ax_cm.set_xticklabels(active_labels, fontsize=9)
    ax_cm.set_yticklabels(active_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted", fontsize=9)
    ax_cm.set_ylabel("Actual", fontsize=9)
    ax_cm.set_title(f"Confusion matrix\n{best_name} (best)")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ---- Panel 3: All classifiers accuracy + per-class breakdown --------
    ax_cls = fig.add_subplot(gs[0, 2])
    x = np.arange(len(active_labels))
    width = 0.8 / len(clf_results)
    clf_colors = ["#5C85D6", "#4CAF50", "#FF9800", "#9C27B0"]
    for k, (name, res) in enumerate(clf_results.items()):
        cm_k  = confusion_matrix(y_test, res["y_pred"], labels=range(len(active_labels)))
        per_class_acc = cm_k.diagonal() / (cm_k.sum(axis=1) + 1e-10)
        ax_cls.bar(x + k * width, per_class_acc, width=width,
                   color=clf_colors[k % len(clf_colors)], alpha=0.85,
                   label=name, edgecolor="white")
    ax_cls.set_xticks(x + width * (len(clf_results) - 1) / 2)
    ax_cls.set_xticklabels(active_labels, fontsize=9)
    ax_cls.set_ylim(0, 1.2)
    ax_cls.set_ylabel("Per-class Accuracy")
    ax_cls.set_title("Per-class accuracy\nby classifier")
    ax_cls.legend(fontsize=6, loc="upper right")
    ax_cls.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # ---- Panel 4: Feature importances (tree-based) ----------------------
    ax_fi = fig.add_subplot(gs[1, :])
    if feat_importances:
        # Show all tree-based models side by side for top features
        fi_colors = {"Random Forest": "#4CAF50", "Gradient Boosting": "#FF9800"}
        # Find top-N features by max importance across available models
        all_fi = np.stack(list(feat_importances.values()))
        top_idx = np.argsort(all_fi.max(axis=0))[::-1][:20]
        x_fi = np.arange(len(top_idx))
        n_fi_models = len(feat_importances)
        w_fi = 0.8 / n_fi_models

        for k, (fi_name, fi_vals) in enumerate(feat_importances.items()):
            color = fi_colors.get(fi_name, "#5C85D6")
            ax_fi.bar(x_fi + k * w_fi, fi_vals[top_idx], width=w_fi,
                      color=color, alpha=0.85, label=fi_name, edgecolor="white")

        ax_fi.set_xticks(x_fi + w_fi * (n_fi_models - 1) / 2)
        ax_fi.set_xticklabels([feat_names[i] for i in top_idx],
                               rotation=40, ha="right", fontsize=7)
        ax_fi.set_ylabel("Feature Importance")
        ax_fi.set_title("Top 20 feature importances (tree-based classifiers)")
        ax_fi.legend(fontsize=8)
        ax_fi.grid(axis="y", alpha=0.3)
    else:
        ax_fi.text(0.5, 0.5, "No tree-based importances available",
                   ha="center", va="center", transform=ax_fi.transAxes)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
