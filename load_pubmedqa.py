from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

for i, example in enumerate(dataset.select(range(5))):
    print(f"--- Example {i+1} ---")
    print(f"Question: {example['question']}")
    print(f"Context (abstracts): {example['context']['contexts'][:1]}...")  # first abstract only
    print(f"Long answer: {example['long_answer'][:200]}...")
    print(f"Final answer: {example['final_decision']}")
    print()
