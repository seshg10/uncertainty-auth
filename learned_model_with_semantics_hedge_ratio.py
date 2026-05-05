"""
learned_model_with_semantics_hedge_ratio.py

Extends learned_model.py by adding linguistic uncertainty features derived
from each model's explanation text.

Hedge words signal epistemic uncertainty ("suggests", "may", "possibly").
Assertive words signal confidence ("demonstrates", "confirms", "clearly").
The ratio of hedge to assertive words, and the divergence between models,
are theoretically informative about whether a prediction is reliable.

New features per model:
  - hedge_count      : number of hedge words in explanation
  - assertive_count  : number of assertive words
  - hedge_ratio      : hedge / (hedge + assertive + 1)  — bounded [0,1]
  - total_words      : explanation length in words
  - hedge_density    : hedge_count / total_words

Cross-model delta features:
  - delta_hedge_ratio   : hedge_ratio_A − hedge_ratio_B
  - hedge_agreement     : 1 if both models hedge or both assert, 0 otherwise

Trains four classifiers on three feature sets and compares:
  (a) logprob only           — baseline (identical to learned_model.py)
  (b) hedge/assertive only   — new features standalone
  (c) logprob + hedge        — combined

Usage:
    python learned_model_with_semantics_hedge_ratio.py \\
        results_gpt3.5_turbo.json results_gpt4o_turbo.json
    python learned_model_with_semantics_hedge_ratio.py \\
        results_gpt3.5_turbo.json results_gpt4o_turbo.json --seed 7 --train-size 50
"""

import argparse
import random
import re
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
    feature_names_for,
    load_file,
)

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

HEDGE_WORDS = {
    "suggest", "suggests", "suggested", "suggesting",
    "may", "might", "could", "possibly", "possible", "possibility",
    "appears", "appear", "appeared", "seemingly", "seems", "seem",
    "indicate", "indicates", "indicated", "indicating",
    "unclear", "uncertain", "uncertainty", "inconclusive",
    "limited", "limiting", "potentially", "potential",
    "likely", "unlikely", "probable", "probably", "improbable",
    "however", "although", "despite", "nevertheless", "nonetheless",
    "mixed", "inconsistent", "variable", "some", "partial", "partially",
    "tend", "tends", "tended", "generally", "often", "sometimes",
    "preliminary", "tentative", "speculative",
}

ASSERTIVE_WORDS = {
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "confirm", "confirms", "confirmed", "confirming",
    "show", "shows", "shown", "showing",
    "prove", "proves", "proven", "proving",
    "establish", "establishes", "established", "establishing",
    "conclude", "concludes", "concluded", "concluding",
    "clearly", "definitively", "definitive", "definite",
    "strongly", "undoubtedly", "undeniably", "evidently",
    "significant", "significantly", "substantial", "substantially",
    "directly", "explicitly", "explicitly",
    "support", "supports", "supported", "supporting",
    "reveal", "reveals", "revealed", "revealing",
    "verify", "verifies", "verified", "verifying",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_explanation_text(generated_text: str) -> str:
    """Return the explanation portion of the generated text."""
    match = re.search(r"Explanation\s*:\s*(.+)", generated_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else generated_text.strip()


def hedge_features(explanation: str) -> np.ndarray:
    """
    Return a 5-element feature vector for one model's explanation:
      [hedge_count, assertive_count, hedge_ratio, total_words, hedge_density]
    """
    words = re.findall(r"\b\w+\b", explanation.lower())
    total = len(words)
    hedge      = sum(1 for w in words if w in HEDGE_WORDS)
    assertive  = sum(1 for w in words if w in ASSERTIVE_WORDS)
    hedge_ratio   = hedge / (hedge + assertive + 1)   # +1 avoids div-by-zero
    hedge_density = hedge / (total + 1)
    return np.array([float(hedge), float(assertive), hedge_ratio,
                     float(total), hedge_density])


HEDGE_FEAT_NAMES_PER_MODEL = [
    "hedge_count", "assertive_count", "hedge_ratio", "total_words", "hedge_density"
]

def hedge_feature_names(model_names: list[str]) -> list[str]:
    names = []
    for m in model_names:
        tag = m[:14]
        names += [f"{tag}:{f}" for f in HEDGE_FEAT_NAMES_PER_MODEL]
    names += ["delta:hedge_ratio", "hedge_agreement"]
    return names


def combined_hedge_features(expl_a: str, expl_b: str) -> np.ndarray:
    """Build the full hedge feature vector for an example (both models + deltas)."""
    fa = hedge_features(expl_a)
    fb = hedge_features(expl_b)
    delta_ratio     = fa[2] - fb[2]                        # delta_hedge_ratio
    # 1 if both hedge-dominant or both assertive-dominant
    agreement = float((fa[2] > 0.5) == (fb[2] > 0.5))
    return np.concatenate([fa, fb, [delta_ratio, agreement]])


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
    parser.add_argument("--output",     default="plots/learned_hedge_ratio_plot.png")
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

    X_logprob_list, X_hedge_list, y_list, pubids = [], [], [], []

    for pubid in common:
        records = {m: all_data[m][pubid] for m in model_names}
        truth   = all_data[name_a][pubid]["ground_truth"].lower()

        vec = build_feature_vector(records, model_names)
        if vec is None:
            continue

        expl_a = extract_explanation_text(all_data[name_a][pubid]["generated_text"])
        expl_b = extract_explanation_text(all_data[name_b][pubid]["generated_text"])

        X_logprob_list.append(vec)
        X_hedge_list.append(combined_hedge_features(expl_a, expl_b))
        y_list.append(truth)
        pubids.append(pubid)

    X_logprob = np.array(X_logprob_list)
    X_hedge   = np.array(X_hedge_list)
    X_combined = np.hstack([X_logprob, X_hedge])
    y         = np.array(y_list)
    le        = LabelEncoder()
    y_enc     = le.fit_transform(y)

    print(f"Dataset     : {len(y)} examples, classes={list(le.classes_)}")
    print(f"Feature dims: logprob={X_logprob.shape[1]}, "
          f"hedge={X_hedge.shape[1]}, combined={X_combined.shape[1]}")

    # Print a quick sample to verify lexicon is firing
    sample_hedge  = X_hedge[:, 0].mean()
    sample_assert = X_hedge[:, 1].mean()
    sample_ratio  = X_hedge[:, 2].mean()
    print(f"Lexicon check (mean over all): hedge={sample_hedge:.2f}, "
          f"assertive={sample_assert:.2f}, hedge_ratio={sample_ratio:.3f}")

    rng = random.Random(args.seed)
    idx = list(range(len(y)))
    rng.shuffle(idx)
    n_tr      = min(args.train_size, len(idx) - 1)
    train_idx = np.array(idx[:n_tr])
    test_idx  = np.array(idx[n_tr:])

    print(f"Split       : {len(train_idx)} train / {len(test_idx)} test  (seed={args.seed})\n")

    def split(X): return X[train_idx], X[test_idx]

    X_lp_tr, X_lp_te = split(X_logprob)
    X_hg_tr, X_hg_te = split(X_hedge)
    X_co_tr, X_co_te = split(X_combined)
    y_tr, y_te        = y_enc[train_idx], y_enc[test_idx]

    res_lp = train_and_evaluate(X_lp_tr, y_tr, X_lp_te, y_te, le, args.seed, "Logprob only (baseline)")
    res_hg = train_and_evaluate(X_hg_tr, y_tr, X_hg_te, y_te, le, args.seed, "Hedge/assertive only")
    res_co = train_and_evaluate(X_co_tr, y_tr, X_co_te, y_te, le, args.seed, "Logprob + Hedge/assertive")

    all_res = {
        "Logprob only":           res_lp,
        "Hedge/assertive only":   res_hg,
        "Logprob + Hedge":        res_co,
    }

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

    feat_names_all = feature_names_for(model_names) + hedge_feature_names(model_names)
    plot(all_res, res_lp, le, y_te, y[test_idx], X_hg_te,
         hedge_feature_names(model_names), feat_names_all, Path(args.output))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(all_res, res_lp, le, y_te, y_te_str, X_hg_te,
         hedge_feat_names, all_feat_names, out_path):
    clf_names     = list(res_lp.keys())
    active_labels = list(le.classes_)
    fset_labels   = list(all_res.keys())
    palette       = ["#5C85D6", "#FF9800", "#4CAF50"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Hedge/Assertive Ratio as Semantic Divergence Feature",
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
    ax_acc.set_title("Test accuracy: baseline vs hedge/assertive features")
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
    best_co_name = max(all_res["Logprob + Hedge"],
                       key=lambda k: all_res["Logprob + Hedge"][k]["acc"])
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_te, all_res["Logprob + Hedge"][best_co_name]["y_pred"])
    im_cm = ax_cm.imshow(cm, cmap="Oranges", aspect="auto")
    ax_cm.set_xticks(range(len(active_labels)))
    ax_cm.set_yticks(range(len(active_labels)))
    ax_cm.set_xticklabels(active_labels, fontsize=9)
    ax_cm.set_yticklabels(active_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion — Logprob+Hedge\n{best_co_name} ({all_res['Logprob + Hedge'][best_co_name]['acc']:.0%})")
    plt.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ---- Panels 4-7: Hedge feature distributions by label and correct ---
    best_co_pred = all_res["Logprob + Hedge"][best_co_name]["y_pred"]
    y_correct = best_co_pred == y_te

    display_feats = [
        (2,  "hedge_ratio (A)"),
        (7,  "hedge_ratio (B)"),
        (10, "delta_hedge_ratio"),
        (11, "hedge_agreement"),
    ]
    for (feat_idx, feat_label), gspec in zip(
        display_feats,
        [gs[1, 1], gs[1, 2], gs[2, 1], gs[2, 2]]
    ):
        ax = fig.add_subplot(gspec)
        vals = X_hg_te[:, feat_idx]
        ax.hist(vals[y_correct],  bins=20, alpha=0.65, color="#4CAF50",
                label="correct",   density=True)
        ax.hist(vals[~y_correct], bins=20, alpha=0.65, color="#F44336",
                label="incorrect", density=True)
        ax.set_xlabel(feat_label, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{feat_label}", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # ---- Panel 8: Feature importances -----------------------------------
    ax_fi = fig.add_subplot(gs[2, 0])
    tree_res = {n: r for n, r in all_res["Logprob + Hedge"].items()
                if r["feat_imp"] is not None}
    if tree_res:
        best_tree = max(tree_res, key=lambda k: tree_res[k]["acc"])
        fi = tree_res[best_tree]["feat_imp"]
        top_idx = np.argsort(fi)[::-1][:15]
        is_hedge = ["hedge" in all_feat_names[i] or "assertive" in all_feat_names[i]
                    or "delta" in all_feat_names[i] or "agreement" in all_feat_names[i]
                    for i in top_idx[::-1]]
        ax_fi.barh(range(len(top_idx)), fi[top_idx][::-1],
                   color=["#FF9800" if h else "#5C85D6" for h in is_hedge],
                   edgecolor="white")
        ax_fi.set_yticks(range(len(top_idx)))
        ax_fi.set_yticklabels([all_feat_names[i] for i in top_idx[::-1]], fontsize=7)
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title(f"Top features ({best_tree})\norange = hedge/assertive feature")
        ax_fi.grid(axis="x", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
