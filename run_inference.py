"""
Inference script: run PubMedQA labeled examples through a vLLM-served LLM.

For each example the script:
  - builds a chat prompt asking the model to answer yes / no / maybe
  - calls the vLLM OpenAI-compatible completions endpoint with logprobs enabled
  - saves the generated text, per-token logprobs, and top-k alternatives to JSON

Usage:
    python run_inference.py \
        --model <model-name-as-registered-in-vllm> \
        --output results.json \
        [--max-examples 100] \
        [--top-logprobs 5] \
        [--max-tokens 256] \
        [--base-url http://localhost:8000/v1]

The output file is written incrementally so partial runs are not lost.
Re-running with the same --output file resumes from where it left off.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a biomedical question-answering assistant. "
    "Given a research question and supporting context excerpts from PubMed abstracts, "
    "answer with exactly one of: yes, no, or maybe. "
    "Then provide a brief explanation (1-3 sentences). "
    "Format your response as:\nAnswer: <yes|no|maybe>\nExplanation: <explanation>"
)


def build_user_message(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Context {i + 1}]\n{ctx.strip()}" for i, ctx in enumerate(contexts)
    )
    return f"{context_block}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    question: str,
    contexts: list[str],
    max_tokens: int,
    top_logprobs: int,
) -> dict:
    """Call vLLM and return a structured result dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, contexts)},
    ]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0.0,          # greedy — deterministic for reproducibility
        logprobs=True,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
    )
    # Retry loop: strip unsupported parameters reported by the API (newer models
    # like gpt-5.5 reject max_tokens, temperature=0, and/or logprobs).
    for _attempt in range(4):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            err = str(exc)
            fixed = False
            if "max_tokens" in err and "max_completion_tokens" in err:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                fixed = True
            if "temperature" in err and ("not support" in err or "unsupported" in err.lower()):
                kwargs.pop("temperature", None)
                fixed = True
            if "logprobs" in err and "not supported" in err:
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                fixed = True
            if not fixed:
                raise

    choice = response.choices[0]
    generated_text = choice.message.content or ""

    # Serialize per-token logprob data
    token_logprobs = []
    if choice.logprobs and choice.logprobs.content:
        for tl in choice.logprobs.content:
            token_logprobs.append({
                "token": tl.token,
                "logprob": tl.logprob,
                "prob": math.exp(tl.logprob),
                "top_logprobs": [
                    {
                        "token": alt.token,
                        "logprob": alt.logprob,
                        "prob": math.exp(alt.logprob),
                    }
                    for alt in (tl.top_logprobs or [])
                ],
            })

    # Convenience: extract top-k logprobs for the very first token.
    # This is the yes/no/maybe decision token and is most useful for
    # downstream uncertainty estimation.
    first_token_logprobs = token_logprobs[0] if token_logprobs else None

    return {
        "generated_text": generated_text,
        "token_logprobs": token_logprobs,
        "first_token_logprobs": first_token_logprobs,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing_results(output_path: Path) -> dict[str, dict]:
    """Return a mapping pubid -> result for any already-processed examples."""
    if not output_path.exists():
        return {}
    with output_path.open() as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] Could not parse existing {output_path}; starting fresh.")
            return {}
    return {str(r["pubid"]): r for r in data}


def save_results(output_path: Path, results: list[dict]) -> None:
    tmp = output_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMedQA inference with logprobs")
    parser.add_argument("--model", required=True, help="Model name as registered in vLLM")
    parser.add_argument("--output", default=None, help="Output JSON file (default: auto-generated from model/examples/date)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap number of examples (default: all 1000)")
    parser.add_argument("--top-logprobs", type=int, default=5, help="Number of top-k token alternatives to capture (max 20)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for model response")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Seconds to wait before retrying a failed call")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per example on API error")
    args = parser.parse_args()

    # Load dataset first so we know the actual example count for the filename
    print("Loading PubMedQA pqa_labeled …")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    print(f"  {len(dataset)} examples to process")

    # Auto-generate output filename if not provided
    if args.output is None:
        date_str   = datetime.now(timezone.utc).strftime("%Y%m%d")
        model_slug = args.model.replace("/", "-").replace(":", "-")
        args.output = f"data/results_{model_slug}_{len(dataset)}ex_{date_str}.json"

    output_path = Path(args.output)
    print(f"  Output: {output_path}")

    # Resume support
    done = load_existing_results(output_path)
    print(f"  {len(done)} already processed — will skip these")

    client = OpenAI(base_url=args.base_url)  # uses OPENAI_API_KEY env var; vLLM ignores it

    results: list[dict] = list(done.values())
    errors: list[dict] = []

    for i, example in enumerate(dataset):
        pubid = str(example["pubid"])
        if pubid in done:
            continue

        question = example["question"]
        contexts = example["context"]["contexts"]
        ground_truth = example["final_decision"]

        print(f"[{i + 1}/{len(dataset)}] pubid={pubid}  label={ground_truth}  ", end="", flush=True)

        attempt = 0
        result = None
        while attempt < args.max_retries:
            try:
                result = call_model(
                    client=client,
                    model=args.model,
                    question=question,
                    contexts=contexts,
                    max_tokens=args.max_tokens,
                    top_logprobs=args.top_logprobs,
                )
                break
            except Exception as exc:
                attempt += 1
                print(f"\n  [error attempt {attempt}] {exc}", end="")
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay)
                else:
                    print(f"\n  [skip] giving up on pubid={pubid}")
                    errors.append({"pubid": pubid, "error": str(exc)})

        if result is None:
            continue

        record = {
            "pubid": pubid,
            "question": question,
            "context_texts": contexts,
            "ground_truth": ground_truth,
            "model": args.model,
            "system_prompt": SYSTEM_PROMPT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }

        results.append(record)
        done[pubid] = record

        # Incremental save after every example
        save_results(output_path, results)

        finish = result.get("finish_reason", "?")
        n_tokens = len(result.get("token_logprobs", []))
        print(f"ok  ({n_tokens} tokens, finish={finish})")

    print(f"\nDone. {len(results)} results written to {output_path}")
    if errors:
        print(f"  {len(errors)} examples failed: {[e['pubid'] for e in errors]}")


if __name__ == "__main__":
    main()
