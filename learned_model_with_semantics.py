"""
learned_model_with_semantics.py

Extends learned_model.py by adding semantic divergence features derived from
the Explanation text produced by each model.

For every example the script:
  1. Extracts the explanation sentence(s) from each model's generated_text.
  2. Embeds each explanation with OpenAI text-embedding-3-small (results are
     cached to embeddings_cache.json so the API is only called once).
  3. Computes divergence features between the two embedding vectors in two ways:
       Scalar (4 features):
         - cosine_distance   = 1 - cos_sim(emb_A, emb_B)
         - l2_distance       = ||emb_A - emb_B||
         - norm_A, norm_B    = ||emb_A||, ||emb_B||
       PCA (--pca-dims, default 10 features):
         - PCA fitted on training-set embedding differences (emb_A - emb_B),
           capturing the principal directions of semantic divergence.
  4. Trains the same four classifiers on four feature sets and compares:
       (a) logprob only          — baseline identical to learned_model.py
       (b) logprob + 4 scalars   — existing scalar semantic features
       (c) PCA only              — PCA-reduced embedding difference, no logprobs
       (d) logprob + PCA         — logprobs augmented with PCA embedding features

Usage:
    python learned_model_with_semantics.py results_gpt3.5_turbo.json results_gpt4o_turbo.json
    python learned_model_with_semantics.py results_gpt3.5_turbo.json results_gpt4o_turbo.json \\
        --seed 7 --train-size 50 --pca-dims 10 --embed-model text-embedding-3-small
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Re-use helpers from learned_model.py
from learned_model import (
    LABELS,
    build_feature_vector,
    extract_answer,
    feature_names_for,
    load_file,
)

CACHE_FILE = Path("data/embeddings_cache.json")


# ---------------------------------------------------------------------------
# Explanation extraction
# ---------------------------------------------------------------------------

def extract_explanation(generated_text: str) -> str:
    """Return the text after 'Explanation:' or the full text if not found."""
    match = re.search(r"Explanation\s*:\s*(.+)", generated_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else generated_text.strip()


# ---------------------------------------------------------------------------
# Embedding with disk cache
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with CACHE_FILE.open() as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with CACHE_FILE.open("w") as f:
        json.dump(cache, f)


def get_embeddings(
    texts: list[str],
    keys: list[str],
    client: OpenAI,
    model: str,
    cache: dict,
) -> list[list[float]]:
    """
    Return embeddings for each text. Hits cache first; fetches missing ones in
    a single batched API call, then updates the cache.
    """
    missing_idx   = [i for i, k in enumerate(keys) if k not in cache]
    missing_texts = [texts[i] for i in missing_idx]

    if missing_texts:
        print(f"  Fetching {len(missing_texts)} embeddings from API ({model}) …")
        # Batch in chunks of 512 (API limit is 2048, but keep requests manageable)
        chunk_size = 512
        fetched: list[list[float]] = []
        for start in range(0, len(missing_texts), chunk_size):
            chunk = missing_texts[start : start + chunk_size]
            response = client.embeddings.create(input=chunk, model=model)
            fetched.extend([item.embedding for item in response.data])

        for i, emb in zip(missing_idx, fetched):
            cache[keys[i]] = emb
        save_cache(cache)
        print(f"  Cache updated → {CACHE_FILE}")

    return [cache[k] for k in keys]


# ---------------------------------------------------------------------------
# Semantic divergence features
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def semantic_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Return a 4-element vector:
      [cosine_distance, l2_distance, norm_a, norm_b]
    """
    return np.array([
        cosine_distance(emb_a, emb_b),
        float(np.linalg.norm(emb_a - emb_b)),
        float(np.linalg.norm(emb_a)),
        float(np.linalg.norm(emb_b)),
    ])


SEMANTIC_FEAT_NAMES = [
    "sem:cosine_dist",   # how differently the two models explain — high = disagreement
    "sem:l2_dist",       # absolute distance in embedding space
    "sem:norm_A",        # magnitude of model A's explanation embedding
    "sem:norm_B",        # magnitude of model B's explanation embedding
]


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


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    seed: int,
    label: str,
) -> dict:
    """Train all classifiers, return dict of results."""
    classifiers = make_classifiers(seed)
    results = {}
    print(f"\n{'='*60}")
    print(f"Feature set: {label}")
    print(f"{'='*60}")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        results[name] = {
            "acc":    acc,
            "y_pred": y_pred,
            "clf":    clf,
        }
        fi = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
        results[name]["feat_imp"] = fi

        print(f"\n{name}  accuracy = {acc:.0%}  ({int(acc*len(y_test))}/{len(y_test)})")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Two results JSON files")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train-size",  type=int,   default=50)
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="OpenAI embedding model")
    parser.add_argument("--pca-dims",   type=int,   default=10,
                        help="Number of PCA dimensions for embedding difference features")
    parser.add_argument("--output",      default="plots/learned_semantics_plot.png")
    args = parser.parse_args()

    if len(args.files) != 2:
        raise ValueError("Provide exactly two inference JSON files.")

    file_paths  = [Path(f) for f in args.files]
    all_data    = {p.stem: load_file(p) for p in file_paths}
    model_names = list(all_data.keys())
    name_a, name_b = model_names

    common = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    print(f"Files   : {[p.name for p in file_paths]}")
    print(f"Common  : {len(common)} examples")

    # ------------------------------------------------------------------
    # Build logprob feature vectors + collect explanations
    # ------------------------------------------------------------------
    X_logprob, y_all, pubids = [], [], []
    explanations_a, explanations_b = [], []

    for pubid in common:
        records = {m: all_data[m][pubid] for m in model_names}
        truth   = all_data[name_a][pubid]["ground_truth"].lower()
        vec     = build_feature_vector(records, model_names)
        if vec is None:
            continue

        X_logprob.append(vec)
        y_all.append(truth)
        pubids.append(pubid)
        explanations_a.append(extract_explanation(all_data[name_a][pubid]["generated_text"]))
        explanations_b.append(extract_explanation(all_data[name_b][pubid]["generated_text"]))

    X_logprob = np.array(X_logprob)
    y_all     = np.array(y_all)
    le        = LabelEncoder()
    y_enc     = le.fit_transform(y_all)
    n         = len(X_logprob)
    print(f"Dataset : {n} usable examples, classes={list(le.classes_)}")

    # ------------------------------------------------------------------
    # Fetch / load embeddings (OpenAI client only created on cache miss)
    # ------------------------------------------------------------------
    cache  = load_cache()
    keys_a = [f"{name_a}::{pid}" for pid in pubids]
    keys_b = [f"{name_b}::{pid}" for pid in pubids]
    missing = [k for k in keys_a + keys_b if k not in cache]
    client  = OpenAI() if missing else None

    print(f"\nEmbedding model: {args.embed_model}")
    if missing:
        print(f"  {len(missing)} embeddings not cached — fetching from API")
    else:
        print(f"  All embeddings loaded from cache ({CACHE_FILE})")

    embs_a = get_embeddings(explanations_a, keys_a, client, args.embed_model, cache)
    embs_b = get_embeddings(explanations_b, keys_b, client, args.embed_model, cache)

    embs_a = [np.array(e) for e in embs_a]
    embs_b = [np.array(e) for e in embs_b]

    # ------------------------------------------------------------------
    # Build semantic feature matrix
    # ------------------------------------------------------------------
    X_sem = np.array([semantic_features(ea, eb) for ea, eb in zip(embs_a, embs_b)])
    X_combined = np.hstack([X_logprob, X_sem])

    print(f"\nFeature dims — logprob: {X_logprob.shape[1]}, "
          f"semantic: {X_sem.shape[1]}, combined: {X_combined.shape[1]}")

    # ------------------------------------------------------------------
    # Train / test split (same split for fair comparison)
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train   = min(args.train_size, n - 1)
    train_idx = np.array(idx[:n_train])
    test_idx  = np.array(idx[n_train:])

    print(f"Split   : {len(train_idx)} train / {len(test_idx)} test  (seed={args.seed})\n")

    def split(X):
        return X[train_idx], X[test_idx]

    X_lp_tr, X_lp_te   = split(X_logprob)
    X_co_tr, X_co_te   = split(X_combined)
    y_tr,    y_te       = y_enc[train_idx], y_enc[test_idx]

    # ------------------------------------------------------------------
    # PCA on embedding differences — fit on train only, apply to all
    # ------------------------------------------------------------------
    X_diff = np.array([ea - eb for ea, eb in zip(embs_a, embs_b)])  # (n, embed_dim)
    n_components = min(args.pca_dims, X_diff.shape[1], len(train_idx))
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(X_diff[train_idx])
    X_pca = pca.transform(X_diff)  # (n, pca_dims)

    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA     : {n_components} components explain {var_explained:.1%} of embedding-diff variance\n")

    X_pca_tr, X_pca_te             = split(X_pca)
    X_lp_pca_tr, X_lp_pca_te       = split(np.hstack([X_logprob, X_pca]))

    # ------------------------------------------------------------------
    # Train on all four feature sets
    # ------------------------------------------------------------------
    res_logprob   = train_and_evaluate(X_lp_tr,     y_tr, X_lp_te,     y_te, le,
                                       args.seed, "Logprob only (baseline)")
    res_scalar    = train_and_evaluate(X_co_tr,     y_tr, X_co_te,     y_te, le,
                                       args.seed, "Logprob + 4 scalar semantics")
    res_pca_only  = train_and_evaluate(X_pca_tr,    y_tr, X_pca_te,    y_te, le,
                                       args.seed, f"PCA-{n_components} only")
    res_lp_pca    = train_and_evaluate(X_lp_pca_tr, y_tr, X_lp_pca_te, y_te, le,
                                       args.seed, f"Logprob + PCA-{n_components}")

    # Summary table
    all_res = {
        "Logprob only":              res_logprob,
        "+ 4 scalar sem":            res_scalar,
        f"PCA-{n_components} only":  res_pca_only,
        f"Logprob+PCA-{n_components}": res_lp_pca,
    }
    col_w = 16
    print(f"\n{'='*75}")
    header = f"{'Classifier':<26}" + "".join(f"{k:>{col_w}}" for k in all_res)
    print(header)
    print("-" * 75)
    for clf_name in res_logprob:
        row = f"{clf_name:<26}"
        base_acc = res_logprob[clf_name]["acc"]
        for label, res in all_res.items():
            acc   = res[clf_name]["acc"]
            delta = acc - base_acc
            sign  = "+" if delta > 0 else ("" if delta == 0 else "")
            tag   = f" ({sign}{delta:.0%})" if delta != 0 else ""
            row  += f"{acc:.0%}{tag:>7}{'':{col_w - 9}}"
        print(row)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    feat_names_lp  = feature_names_for(model_names)
    feat_names_pca = [f"pca_{i+1}" for i in range(n_components)]
    feat_names_lp_pca = feat_names_lp + feat_names_pca

    plot(
        all_res=all_res,
        res_lp=res_logprob,
        le=le,
        y_te=y_te,
        y_te_str=y_all[test_idx],
        X_sem_te=X_sem[test_idx],
        X_pca_te=X_pca_te,
        pca=pca,
        feat_names_combined=feat_names_lp + SEMANTIC_FEAT_NAMES,
        feat_names_lp_pca=feat_names_lp_pca,
        n_components=n_components,
        var_explained=var_explained,
        out_path=Path(args.output),
    )




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    all_res:              dict,        # {feature_set_label: clf_results}
    res_lp:               dict,        # logprob-only results (delta reference)
    le:                   LabelEncoder,
    y_te:                 np.ndarray,
    y_te_str:             np.ndarray,
    X_sem_te:             np.ndarray,  # (n_test, 4) scalar semantic features
    X_pca_te:             np.ndarray,  # (n_test, pca_dims)
    pca:                  PCA,
    feat_names_combined:  list,        # logprob + 4 scalar names
    feat_names_lp_pca:    list,        # logprob + pca names
    n_components:         int,
    var_explained:        float,
    out_path:             Path,
) -> None:
    active_labels   = list(le.classes_)
    clf_names       = list(res_lp.keys())
    feat_set_labels = list(all_res.keys())
    n_fsets         = len(feat_set_labels)
    palette         = ["#5C85D6", "#E07B39", "#9C27B0", "#4CAF50"]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Semantic Divergence Features: Scalar vs PCA\n"
        f"(PCA-{n_components} on embedding difference, {var_explained:.1%} variance explained)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)

    # ---- Panel 1: Grouped accuracy bars (all four feature sets) ---------
    ax_acc = fig.add_subplot(gs[0, :3])
    x     = np.arange(len(clf_names))
    width = 0.8 / n_fsets

    for k, (fset_label, res) in enumerate(all_res.items()):
        accs = [res[n]["acc"] for n in clf_names]
        bars = ax_acc.bar(x + k * width, accs, width,
                          label=fset_label, color=palette[k], alpha=0.85, edgecolor="white")
        for bar, acc in zip(bars, accs):
            ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                        f"{acc:.0%}", ha="center", fontsize=7, fontweight="bold")

    ax_acc.set_xticks(x + width * (n_fsets - 1) / 2)
    ax_acc.set_xticklabels(clf_names, rotation=12, ha="right", fontsize=8)
    ax_acc.set_ylim(0, 1.25)
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Test accuracy across all feature sets")
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_acc.legend(fontsize=8, loc="upper right")

    # ---- Panel 2: Delta heatmap vs logprob baseline ---------------------
    ax_delta = fig.add_subplot(gs[0, 3])
    non_base_labels = feat_set_labels[1:]
    delta_matrix = np.array([
        [all_res[lbl][n]["acc"] - res_lp[n]["acc"] for n in clf_names]
        for lbl in non_base_labels
    ])
    im = ax_delta.imshow(delta_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    ax_delta.set_xticks(range(len(clf_names)))
    ax_delta.set_yticks(range(len(non_base_labels)))
    ax_delta.set_xticklabels(clf_names, rotation=30, ha="right", fontsize=7)
    ax_delta.set_yticklabels(non_base_labels, fontsize=7)
    ax_delta.set_title("Δ accuracy vs\nlogprob baseline")
    plt.colorbar(im, ax=ax_delta, fraction=0.046, pad=0.04)
    for i in range(len(non_base_labels)):
        for j in range(len(clf_names)):
            d = delta_matrix[i, j]
            ax_delta.text(j, i, f"{'+' if d > 0 else ''}{d:.0%}",
                          ha="center", va="center", fontsize=7, color="black")

    # ---- Panels 3 & 4: Confusion matrices for PCA-only and Logprob+PCA --
    pca_only_label = [l for l in feat_set_labels if "PCA" in l and "Logprob" not in l][0]
    lp_pca_label   = [l for l in feat_set_labels if "PCA" in l and "Logprob" in l][0]

    for col_idx, (fset_label, cmap) in enumerate(
        [(pca_only_label, "Purples"), (lp_pca_label, "Greens")]
    ):
        res      = all_res[fset_label]
        best_clf = max(res, key=lambda k: res[k]["acc"])
        ax_cm    = fig.add_subplot(gs[1, col_idx * 2 : col_idx * 2 + 2])
        cm       = confusion_matrix(y_te, res[best_clf]["y_pred"])
        im_cm    = ax_cm.imshow(cm, cmap=cmap, aspect="auto")
        ax_cm.set_xticks(range(len(active_labels)))
        ax_cm.set_yticks(range(len(active_labels)))
        ax_cm.set_xticklabels(active_labels, fontsize=9)
        ax_cm.set_yticklabels(active_labels, fontsize=9)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"{fset_label}\nbest: {best_clf} ({res[best_clf]['acc']:.0%})")
        plt.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
        for i in range(len(active_labels)):
            for j in range(len(active_labels)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                           color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ---- Panel 5: PCA scree plot ----------------------------------------
    ax_scree = fig.add_subplot(gs[2, 0])
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax_scree.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
                 color="#9C27B0", alpha=0.7, edgecolor="white")
    ax_scree.step(range(1, n_components + 1), cumvar, where="mid",
                  color="black", linewidth=1.5, label="Cumulative")
    ax_scree.axhline(var_explained, color="gray", linestyle="--", linewidth=0.8)
    ax_scree.set_xlabel("PCA component")
    ax_scree.set_ylabel("Explained variance ratio")
    ax_scree.set_title(f"PCA scree plot\n({var_explained:.1%} total)")
    ax_scree.legend(fontsize=8)
    ax_scree.grid(alpha=0.3)

    # ---- Panel 6: PCA dim1 vs dim2, coloured by ground truth ------------
    ax_pca2d = fig.add_subplot(gs[2, 1])
    label_colors = {"yes": "#4CAF50", "no": "#F44336", "maybe": "#FF9800"}
    for lbl in np.unique(y_te_str):
        mask = y_te_str == lbl
        ax_pca2d.scatter(
            X_pca_te[mask, 0],
            X_pca_te[mask, 1] if X_pca_te.shape[1] > 1 else np.zeros(mask.sum()),
            c=label_colors.get(lbl, "#9E9E9E"), label=lbl,
            alpha=0.7, edgecolors="white", linewidths=0.3, s=50,
        )
    ax_pca2d.set_xlabel("PCA dim 1")
    ax_pca2d.set_ylabel("PCA dim 2")
    ax_pca2d.set_title("Embedding-diff PCA (test)\ncoloured by ground truth")
    ax_pca2d.legend(fontsize=8)
    ax_pca2d.grid(alpha=0.3)

    # ---- Panels 7 & 8: Scalar semantic feature histograms ---------------
    sem_labels    = ["cosine_dist", "l2_dist", "norm_A", "norm_B"]
    scalar_label  = [l for l in feat_set_labels if "scalar" in l][0]
    best_scalar   = max(all_res[scalar_label], key=lambda k: all_res[scalar_label][k]["acc"])
    y_correct     = all_res[scalar_label][best_scalar]["y_pred"] == y_te

    for i, feat_label in enumerate(sem_labels):
        ax = fig.add_subplot(gs[2, 2] if i < 2 else gs[2, 3])
        vals = X_sem_te[:, i]
        if i % 2 == 0:
            ax.cla()
            ax.set_title("Scalar semantic features\ncorrect vs incorrect")
        ax.hist(vals[y_correct],  bins=15, alpha=0.6, color="#4CAF50",
                label="correct",   density=True)
        ax.hist(vals[~y_correct], bins=15, alpha=0.6, color="#F44336",
                label="incorrect", density=True)
        ax.set_xlabel(feat_label, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
