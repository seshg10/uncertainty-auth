
"""
learned_model_with_semantics_explanation_logprobs.py

Extends learned_model.py by adding features derived from the logprobs of the
*explanation* tokens (everything after "Explanation:") in each model's output.

Unlike the answer-token logprob (which measures confidence in the label), the
explanation-token logprobs measure how confidently the model generated its
*reasoning*. A model that is uncertain tends to produce lower, more variable
logprobs across its explanation.

New features per model:
  - mean_logprob   : average logprob over explanation tokens
  - min_logprob    : most uncertain single token in the explanation
  - var_logprob    : variance — uneven confidence signals patchy reasoning
  - mean_prob      : geometric-mean-style probability (exp of mean logprob)
  - length         : number of explanation tokens

Cross-model delta features:
  - delta_mean_logprob : model_A mean − model_B mean
  - delta_min_logprob  : model_A min  − model_B min

Trains four classifiers on three feature sets and compares:
  (a) logprob only          — baseline (identical to learned_model.py)
  (b) expl. logprobs only   — new features standalone
  (c) logprob + expl. logprobs — combined

Usage:
    python learned_model_with_semantics_explanation_logprobs.py \\
        results_gpt3.5_turbo.json results_gpt4o_turbo.json
    python learned_model_with_semantics_explanation_logprobs.py \\
        results_gpt3.5_turbo.json results_gpt4o_turbo.json --seed 7 --train-size 50
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

from learned_model import (
    LABELS,
    build_feature_vector,
    extract_answer,
    feature_names_for,
    load_file,
    normalize_token,
)

# ---------------------------------------------------------------------------
# Explanation logprob extraction
# ---------------------------------------------------------------------------

def extract_explanation_logprobs(token_logprobs: list[dict]) -> list[float] | None:
    """
    Return the logprobs of every token in the explanation section.
    Finds the answer token, then skips "\\nExplanation:" header tokens,
    and collects everything after.
    """
    answer_idx = None
    for i, entry in enumerate(token_logprobs):
        if normalize_token(entry["token"]) in {"yes", "no", "maybe"}:
            answer_idx = i
            break

    if answer_idx is None:
        return None

    # Scan forward for "Explanation" token, then skip past ":"
    expl_start = None
    for i in range(answer_idx + 1, len(token_logprobs)):
        if "explanation" in token_logprobs[i]["token"].lower():
            # next token is ":" — skip both
            expl_start = i + 2
            break

    if expl_start is None or expl_start >= len(token_logprobs):
        return None

    return [entry["logprob"] for entry in token_logprobs[expl_start:]]


def explanation_logprob_features(lps_a: list[float], lps_b: list[float]) -> np.ndarray:
    """
    Build a feature vector from two lists of explanation token logprobs.
    Returns None if either list is empty.
    """
    def stats(lps):
        arr = np.array(lps)
        return np.array([
            float(arr.mean()),          # mean_logprob
            float(arr.min()),           # min_logprob  (most uncertain token)
            float(arr.var()),           # var_logprob
            float(math.exp(arr.mean())),# mean_prob (geometric mean probability)
            float(len(arr)),            # explanation length (tokens)
        ])

    sa = stats(lps_a)
    sb = stats(lps_b)

    delta_mean = sa[0] - sb[0]
    delta_min  = sa[1] - sb[1]

    return np.concatenate([sa, sb, [delta_mean, delta_min]])


EXPL_FEAT_NAMES_PER_MODEL = [
    "mean_logprob", "min_logprob", "var_logprob", "mean_prob", "n_tokens"
]

def expl_feature_names(model_names: list[str]) -> list[str]:
    names = []
    for m in model_names:
        tag = m[:14]
        names += [f"{tag}:expl_{f}" for f in EXPL_FEAT_NAMES_PER_MODEL]
    names += ["delta:expl_mean_logprob", "delta:expl_min_logprob"]
    return names


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def make_classifiers(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=seed),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                           max_depth=3, random_state=seed),
        "SVM (RBF)":           SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
    }


def train_and_evaluate(X_tr, y_tr, X_te, y_te, le, seed, label) -> dict:
    clfs = make_classifiers(seed)
    results = {}
    print(f"\n{'='*60}\nFeature set: {label}\n{'='*60}")
    for name, clf in clfs.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        results[name] = {
            "acc":      acc,
            "y_pred":   y_pred,
            "feat_imp": getattr(clf, "feature_importances_", None),
        }
        print(f"\n{name}  accuracy = {acc:.0%}  ({int(acc*len(y_te))}/{len(y_te)})")
        print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Two results JSON files")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--output",     default="plots/learned_expl_logprobs_plot.png")
    args = parser.parse_args()

    if len(args.files) != 2:
        raise ValueError("Provide exactly two inference JSON files.")

    file_paths  = [Path(f) for f in args.files]
    all_data    = {p.stem: load_file(p) for p in file_paths}
    model_names = list(all_data.keys())
    name_a, name_b = model_names

    common = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    print(f"Files       : {[p.name for p in file_paths]}")
    print(f"Common      : {len(common)} examples")

    X_logprob_list, X_expl_list, y_list, pubids = [], [], [], []

    for pubid in common:
        records = {m: all_data[m][pubid] for m in model_names}
        truth   = all_data[name_a][pubid]["ground_truth"].lower()

        vec = build_feature_vector(records, model_names)
        if vec is None:
            continue

        lps_a = extract_explanation_logprobs(all_data[name_a][pubid]["token_logprobs"])
        lps_b = extract_explanation_logprobs(all_data[name_b][pubid]["token_logprobs"])
        if not lps_a or not lps_b:
            continue

        expl_vec = explanation_logprob_features(lps_a, lps_b)

        X_logprob_list.append(vec)
        X_expl_list.append(expl_vec)
        y_list.append(truth)
        pubids.append(pubid)

    X_logprob = np.array(X_logprob_list)
    X_expl    = np.array(X_expl_list)
    X_combined = np.hstack([X_logprob, X_expl])
    y         = np.array(y_list)
    le        = LabelEncoder()
    y_enc     = le.fit_transform(y)

    print(f"Dataset     : {len(y)} examples, classes={list(le.classes_)}")
    print(f"Feature dims: logprob={X_logprob.shape[1]}, "
          f"expl_logprob={X_expl.shape[1]}, combined={X_combined.shape[1]}")

    rng = random.Random(args.seed)
    idx = list(range(len(y)))
    rng.shuffle(idx)
    n_tr      = min(args.train_size, len(idx) - 1)
    train_idx = np.array(idx[:n_tr])
    test_idx  = np.array(idx[n_tr:])

    print(f"Split       : {len(train_idx)} train / {len(test_idx)} test  (seed={args.seed})\n")

    def split(X): return X[train_idx], X[test_idx]

    X_lp_tr,  X_lp_te  = split(X_logprob)
    X_ex_tr,  X_ex_te  = split(X_expl)
    X_co_tr,  X_co_te  = split(X_combined)
    y_tr, y_te          = y_enc[train_idx], y_enc[test_idx]

    res_lp   = train_and_evaluate(X_lp_tr, y_tr, X_lp_te, y_te, le, args.seed, "Logprob only (baseline)")
    res_ex   = train_and_evaluate(X_ex_tr, y_tr, X_ex_te, y_te, le, args.seed, "Explanation logprobs only")
    res_co   = train_and_evaluate(X_co_tr, y_tr, X_co_te, y_te, le, args.seed, "Logprob + Explanation logprobs")

    all_res = {
        "Logprob only":         res_lp,
        "Expl. logprobs only":  res_ex,
        "Logprob + Expl. logprobs": res_co,
    }

    # Summary table
    clf_names = list(res_lp.keys())
    col_w = 22
    print(f"\n{'='*75}")
    print(f"{'Classifier':<26}" + "".join(f"{k:>{col_w}}" for k in all_res))
    print("-" * 75)
    for name in clf_names:
        base = res_lp[name]["acc"]
        row  = f"{name:<26}"
        for label, res in all_res.items():
            acc   = res[name]["acc"]
            delta = acc - base
            sign  = "+" if delta > 0 else ""
            tag   = f"({sign}{delta:.0%})" if delta != 0 else ""
            row  += f"{acc:.0%} {tag:>6}{'':{col_w-10}}"
        print(row)

    feat_names_all = feature_names_for(model_names) + expl_feature_names(model_names)
    plot(all_res, res_lp, le, y_te, y[test_idx], X_ex_te, feat_names_all, Path(args.output))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(all_res, res_lp, le, y_te, y_te_str, X_ex_te, feat_names, out_path):
    clf_names     = list(res_lp.keys())
    active_labels = list(le.classes_)
    fset_labels   = list(all_res.keys())
    palette       = ["#5C85D6", "#9C27B0", "#4CAF50"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Explanation Token Logprobs as Semantic Divergence Features",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ---- Panel 1: Accuracy grouped bars ---------------------------------
    ax_acc = fig.add_subplot(gs[0, :2])
    x, width = np.arange(len(clf_names)), 0.8 / len(fset_labels)
    for k, (label, res) in enumerate(all_res.items()):
        accs = [res[n]["acc"] for n in clf_names]
        bars = ax_acc.bar(x + k * width, accs, width, label=label,
                          color=palette[k], alpha=0.85, edgecolor="white")
        for bar, acc in zip(bars, accs):
            ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                        f"{acc:.0%}", ha="center", fontsize=7, fontweight="bold")
    ax_acc.set_xticks(x + width)
    ax_acc.set_xticklabels(clf_names, rotation=12, ha="right", fontsize=8)
    ax_acc.set_ylim(0, 1.25)
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Test accuracy: baseline vs explanation logprob features")
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_acc.legend(fontsize=8)

    # ---- Panel 2: Delta heatmap -----------------------------------------
    ax_d = fig.add_subplot(gs[0, 2])
    non_base = fset_labels[1:]
    dm = np.array([[all_res[l][n]["acc"] - res_lp[n]["acc"] for n in clf_names]
                   for l in non_base])
    im = ax_d.imshow(dm, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    ax_d.set_xticks(range(len(clf_names)))
    ax_d.set_yticks(range(len(non_base)))
    ax_d.set_xticklabels(clf_names, rotation=30, ha="right", fontsize=7)
    ax_d.set_yticklabels(non_base, fontsize=7)
    ax_d.set_title("Δ vs logprob baseline")
    plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    for i in range(len(non_base)):
        for j in range(len(clf_names)):
            d = dm[i, j]
            ax_d.text(j, i, f"{'+' if d > 0 else ''}{d:.0%}",
                      ha="center", va="center", fontsize=7)

    # ---- Panel 3: Confusion matrix — best combined ----------------------
    best_co_name = max(all_res["Logprob + Expl. logprobs"],
                       key=lambda k: all_res["Logprob + Expl. logprobs"][k]["acc"])
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_te, all_res["Logprob + Expl. logprobs"][best_co_name]["y_pred"])
    im_cm = ax_cm.imshow(cm, cmap="Greens", aspect="auto")
    ax_cm.set_xticks(range(len(active_labels)))
    ax_cm.set_yticks(range(len(active_labels)))
    ax_cm.set_xticklabels(active_labels, fontsize=9)
    ax_cm.set_yticklabels(active_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion — Logprob+Expl\n{best_co_name} ({all_res['Logprob + Expl. logprobs'][best_co_name]['acc']:.0%})")
    plt.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ---- Panels 4-6: Explanation logprob feature distributions ----------
    expl_feat_labels = ["mean_logprob_A", "min_logprob_A", "var_logprob_A",
                        "mean_logprob_B", "min_logprob_B", "var_logprob_B",
                        "delta_mean", "delta_min"]
    best_co_pred = all_res["Logprob + Expl. logprobs"][best_co_name]["y_pred"]
    y_correct = best_co_pred == y_te

    for i, feat_label in enumerate(["mean_logprob_A", "min_logprob_A",
                                     "mean_logprob_B", "min_logprob_B",
                                     "delta_mean_logprob", "var_logprob_A"]):
        feat_idx = ["mean_logprob_A", "min_logprob_A", "var_logprob_A",
                    "mean_prob_A", "n_tokens_A",
                    "mean_logprob_B", "min_logprob_B", "var_logprob_B",
                    "mean_prob_B", "n_tokens_B",
                    "delta_mean", "delta_min"].index(feat_idx_map(feat_label))
        ax = fig.add_subplot(gs[1 + i // 3, 1 + i % 2 if i < 4 else i % 2])

    # Simpler approach: plot 4 most informative explanation features
    expl_display = [
        (0,  "mean_logprob (A)"),
        (1,  "min_logprob (A)"),
        (5,  "mean_logprob (B)"),
        (10, "delta_mean_logprob"),
    ]
    positions = [gs[1, 1], gs[1, 2], gs[2, 1], gs[2, 2]]
    for (feat_idx, feat_label), gspec in zip(expl_display, positions):
        ax = fig.add_subplot(gspec)
        vals = X_ex_te[:, feat_idx]
        ax.hist(vals[y_correct],  bins=20, alpha=0.65, color="#4CAF50",
                label="correct",   density=True)
        ax.hist(vals[~y_correct], bins=20, alpha=0.65, color="#F44336",
                label="incorrect", density=True)
        ax.set_xlabel(feat_label, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{feat_label}\ncorrect vs incorrect", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # ---- Panel 7: Feature importances -----------------------------------
    ax_fi = fig.add_subplot(gs[2, 0])
    tree_res = {n: r for n, r in all_res["Logprob + Expl. logprobs"].items()
                if r["feat_imp"] is not None}
    if tree_res:
        best_tree = max(tree_res, key=lambda k: tree_res[k]["acc"])
        fi = tree_res[best_tree]["feat_imp"]
        top_idx = np.argsort(fi)[::-1][:15]
        ax_fi.barh(range(len(top_idx)), fi[top_idx][::-1],
                   color=["#9C27B0" if "expl" in feat_names[i] or "delta" in feat_names[i]
                           else "#5C85D6" for i in top_idx[::-1]],
                   edgecolor="white")
        ax_fi.set_yticks(range(len(top_idx)))
        ax_fi.set_yticklabels([feat_names[i] for i in top_idx[::-1]], fontsize=7)
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title(f"Top features ({best_tree})\npurple = explanation logprob")
        ax_fi.grid(axis="x", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


def feat_idx_map(name):
    """Map display name to internal array index name."""
    m = {"mean_logprob_A": "mean_logprob_A", "min_logprob_A": "min_logprob_A",
         "mean_logprob_B": "mean_logprob_B", "delta_mean_logprob": "delta_mean"}
    return m.get(name, name)


if __name__ == "__main__":
    main()
