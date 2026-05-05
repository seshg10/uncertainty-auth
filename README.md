# uncertainty-auth

Research project exploring **uncertainty estimation and confidence calibration** in LLMs using biomedical question answering. The core idea: does a model's token-level probability (logprob) on its answer reflect how likely it is to be correct? Can logprobs serve as a reliable uncertainty signal?

## Individual model results

Raw logprob-based accuracy on 1000 PubMedQA examples (predicted label = argmax of answer-token logprob distribution):

| Model | Accuracy | ECE |
|---|---|---|
| GPT-3.5-turbo | 60% (603/1000) | 0.293 |
| GPT-4o | 55% (545/1000) | 0.383 |

ECE = Expected Calibration Error (lower is better). GPT-3.5-turbo is better calibrated despite the lower ECE; GPT-4o is overconfident (ECE=0.383), pushing confidence high even when wrong.

## Feature comparison results

Results on 950 test examples (50 train, seed=42), two models (GPT-3.5-turbo + GPT-4o-turbo):

| Feature set | Best classifier | Accuracy |
|---|---|---|
| Logprob only (baseline) | Random Forest | 77% |
| Logprob + explanation logprobs | Random Forest | **79%** |
| Logprob + hedge/assertive | Random Forest | **79%** |
| Logprob + 4 scalar embedding | Random Forest | ~77% |
| Logprob + PCA-10 embedding | Random Forest | ~77% |
| Semantic features alone | — | 49–55% |

## Dataset

[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`, 1000 examples) — biomedical yes/no questions grounded in PubMed abstracts. Each example has a question, supporting context paragraphs, and a gold-standard `yes` / `no` label.

## Project structure

```
uncertainty-auth/
├── load_pubmedqa.py                                  # Explore dataset (prints 5 examples)
├── run_inference.py                                  # Run PubMedQA through an LLM, save logprobs
├── analyze_results.py                                # Single-model: plot token probs vs ground truth
├── compare_results.py                                # Multi-model comparison with combination strategies
├── learned_model.py                                  # Classifiers on answer-token logprob features
├── learned_model_with_semantics.py                   # Adds embedding divergence features (scalar + PCA)
├── learned_model_with_semantics_explanation_logprobs.py  # Adds explanation-token logprob features
├── learned_model_with_semantics_hedge_ratio.py       # Adds hedge/assertive word ratio features
├── results.json                                      # Inference output (auto-generated)
└── embeddings_cache.json                             # Cached OpenAI embeddings (auto-generated)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install datasets openai matplotlib numpy scikit-learn
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running inference

```bash
python run_inference.py \
    --model gpt-3.5-turbo \
    --base-url https://api.openai.com/v1 \
    --output results.json \
    --max-examples 100 \
    --top-logprobs 5
```

**Resume support** — if the run is interrupted, re-running with the same `--output` file skips already-processed examples.

To start fresh, delete the output file:
```bash
rm results.json
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Model name (as registered with the provider) |
| `--base-url` | `http://localhost:8000/v1` | API endpoint (vLLM, OpenAI, Groq, etc.) |
| `--output` | `results.json` | Output file |
| `--max-examples` | all 1000 | Cap number of examples |
| `--top-logprobs` | 5 | Top-k token alternatives to capture per position (max 20) |
| `--max-tokens` | 256 | Max response length |

## Compatible inference backends

Any OpenAI-compatible endpoint that returns logprobs works:

| Backend | Notes |
|---|---|
| **OpenAI** (`api.openai.com/v1`) | GPT-3.5, GPT-4, GPT-4o |
| **Groq** (`api.groq.com/openai/v1`) | Free tier, fast — Llama 3, Mixtral |
| **Together AI** | Llama 3, Mistral, and others |
| **Fireworks AI** | Llama 3, Mixtral, and others |
| **vLLM** (`localhost:8000/v1`) | Self-hosted, full logprob access, requires GPU |

For non-OpenAI providers, pass your key via `OPENAI_API_KEY`:
```bash
OPENAI_API_KEY=$GROQ_API_KEY python run_inference.py \
    --model llama-3.1-8b-instant \
    --base-url https://api.groq.com/openai/v1
```

## Output format

`results.json` is a list of records, one per example:

```json
{
  "pubid": "21645374",
  "question": "...",
  "context_texts": ["..."],
  "ground_truth": "yes",
  "model": "gpt-3.5-turbo",
  "timestamp": "...",
  "generated_text": "Answer: Yes\nExplanation: ...",
  "token_logprobs": [
    {
      "token": " Yes",
      "logprob": -0.0069,
      "prob": 0.9931,
      "top_logprobs": [
        {"token": " Yes", "logprob": -0.0069, "prob": 0.9931},
        {"token": " yes", "logprob": -5.00,   "prob": 0.0067},
        ...
      ]
    },
    ...
  ],
  "first_token_logprobs": { ... }
}
```

The **answer token** is the first `yes`/`no` token in `token_logprobs`. Its `top_logprobs` show the model's distribution over alternatives at that decision point — the primary signal for uncertainty estimation.

## Analyzing results

```bash
python analyze_results.py
```

Prints per-example token/logprob/prob values to terminal and saves `results_plot.png` showing the raw top-k token probabilities for the answer decision token, with correct (green) / incorrect (red) labels.

## Comparing multiple models

`compare_results.py` loads two or more inference files, combines their probability distributions, and evaluates the combined prediction against ground truth.

```bash
# Equal-weight average (default)
python compare_results.py results.json result_gpt4o.json

# Custom weights (auto-normalised)
python compare_results.py results.json result_gpt4o.json --weights 0.7 0.3

# Choose a combination method
python compare_results.py results.json result_gpt4o.json --method majority_vote
python compare_results.py results.json result_gpt4o.json --method confidence_weighted
python compare_results.py results.json result_gpt4o.json --method entropy_weighted
```

### Combination methods

| Method | How it works |
|---|---|
| `weighted_avg` | Weighted average of yes/no/maybe probability distributions (default) |
| `majority_vote` | Each model votes for its top label; vote share becomes the combined prob |
| `confidence_weighted` | Weights each model by its max probability (more confident = higher weight) |
| `entropy_weighted` | Weights each model inversely by Shannon entropy — uncertain models are down-weighted automatically |

Adding a new method: subclass `Combiner`, implement `combine()`, register in `COMBINERS`.

### Output plots

Two figures are saved (e.g. `comparison_plot.png` and `comparison_plot_calibration.png`):

**Overview plot** — accuracy bar chart, confusion matrices per model + combined, and a per-example correctness heatmap (rows = models, columns = examples sorted by ground truth). Scales to 1000 examples — each cell is colour-coded green/correct or red/incorrect with intensity proportional to confidence.

**Calibration plot** — reliability diagrams (predicted confidence vs actual accuracy) with ECE annotation, and confidence histograms split by correct/incorrect predictions. Key for assessing whether logprobs are well-calibrated.

## Learned classifier

`learned_model.py` extracts token-level features from each model's answer token and trains a set of classifiers on a random 50/50 train/test split.

```bash
python learned_model.py results.json result_gpt4o.json

# Different seed or split size
python learned_model.py results.json result_gpt4o.json --seed 7
python learned_model.py results.json result_gpt4o.json --train-size 70
```

### Features (per model, per example)

| Feature | Description |
|---|---|
| `pred_yes/no/maybe` | One-hot encoding of the predicted label |
| `logprob` | Log-probability of the chosen answer token |
| `prob` | Probability of the chosen answer token |
| `yes_prob / no_prob / maybe_prob` | Aggregated normalised probability for each label from `top_logprobs` |
| `entropy` | Shannon entropy of the label distribution — high entropy = uncertain model |

For two models this gives 18 features total.

### Classifiers trained

Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF). Individual model accuracies are shown as baselines. The output plot includes an accuracy comparison bar chart, confusion matrix for the best classifier, per-class accuracy breakdown, and feature importances for tree-based models.

## Semantic divergence features

The three scripts below all extend `learned_model.py` by appending additional features derived from each model's explanation text. All follow the same structure: train classifiers on (a) logprob only, (b) new features only, (c) logprob + new features, and compare.

### Embedding divergence (`learned_model_with_semantics.py`)

Embeds each model's explanation with `text-embedding-3-small` and computes divergence features. Embeddings are cached to `embeddings_cache.json` after the first run — subsequent runs are free.

```bash
python learned_model_with_semantics.py results_gpt3.5_turbo.json results_gpt4o_turbo.json

# More PCA dimensions
python learned_model_with_semantics.py results_gpt3.5_turbo.json results_gpt4o_turbo.json --pca-dims 20
```

Four feature sets are compared:

| Feature set | Description |
|---|---|
| Logprob only | Baseline — identical to `learned_model.py` |
| Logprob + 4 scalar | cosine distance, L2 distance, norm A, norm B |
| PCA-N only | N-component PCA of embedding difference vector (emb_A − emb_B) |
| Logprob + PCA-N | Logprob features combined with PCA embedding features |

The output plot includes a grouped accuracy bar chart, a Δ-accuracy heatmap vs baseline, confusion matrices for the PCA feature sets, a PCA scree plot, a 2D scatter of the PCA space coloured by ground truth, and scalar feature histograms split by correct/incorrect.

> **Note:** Embedding distance captures topical divergence (do the models discuss different concepts?) rather than epistemic divergence (is one model more uncertain?). Gains over the logprob baseline are typically small.

---

### Explanation token logprobs (`learned_model_with_semantics_explanation_logprobs.py`)

Extracts logprobs from the explanation tokens (everything after `Explanation:`) in `token_logprobs`. No API calls required — all data is already in the inference JSON.

```bash
python learned_model_with_semantics_explanation_logprobs.py \
    results_gpt3.5_turbo.json results_gpt4o_turbo.json
```

New features per model (12 total + 2 cross-model deltas):

| Feature | Description |
|---|---|
| `mean_logprob` | Average logprob over explanation tokens — lower = less confident reasoning |
| `min_logprob` | Most uncertain single token in the explanation |
| `var_logprob` | Variance — high variance signals patchy, uneven confidence |
| `mean_prob` | Geometric-mean probability of explanation tokens |
| `n_tokens` | Length of explanation in tokens |
| `delta_mean_logprob` | Difference in mean logprob between model A and model B |
| `delta_min_logprob` | Difference in min logprob between model A and model B |

**Observed performance:** Random Forest gains ~+2% over the logprob-only baseline when explanation logprobs are combined with answer-token logprobs.

---

### Hedge/assertive word ratio (`learned_model_with_semantics_hedge_ratio.py`)

Counts linguistic uncertainty markers (hedge words: *suggests*, *may*, *possibly*) and confidence markers (assertive words: *demonstrates*, *confirms*, *clearly*) in each model's explanation. No API calls required.

```bash
python learned_model_with_semantics_hedge_ratio.py \
    results_gpt3.5_turbo.json results_gpt4o_turbo.json
```

New features per model (10 total + 2 cross-model):

| Feature | Description |
|---|---|
| `hedge_count` | Number of hedge words in the explanation |
| `assertive_count` | Number of assertive words |
| `hedge_ratio` | `hedge / (hedge + assertive + 1)` — bounded [0, 1] |
| `total_words` | Explanation length in words |
| `hedge_density` | `hedge_count / total_words` |
| `delta_hedge_ratio` | Difference in hedge ratio between model A and model B |
| `hedge_agreement` | 1 if both models hedge or both assert, 0 if they diverge |

**Observed performance:** Random Forest gains ~+2% when combined with logprob features. Hedge features alone perform near majority-class baseline (~51%), as LLMs tend to hedge uniformly regardless of actual uncertainty.

