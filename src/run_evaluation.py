"""
Evaluate the fine-tuned model on data/test_trimmed.jsonl using ROUGE-1/2/L.
Run from the project root:    python src/run_evaluation.py --limit 100
"""

import json
from pathlib import Path
import argparse
import evaluate
from inference import Summarizer

MODEL_DIR = "models/briefgov-bart-final"
TEST_FILE = Path("data/test_trimmed.jsonl")
RESULTS_FILE = Path("results/rouge_scores.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate on this many rows (for a quick check)")
    args = parser.parse_args()

    rouge = evaluate.load("rouge")
    summarizer = Summarizer(MODEL_DIR)

    preds, refs = [], []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.limit:
        records = records[:args.limit]

    print(f"Evaluating on {len(records)} chunk-summary pairs...")

    for i, rec in enumerate(records):
        # each "source" is already a single chunk (<=1024 tokens), so
        # summarize_chunk is the right call here, not the full summarize()
        pred = summarizer.summarize_chunk(rec["source"])
        preds.append(pred)
        refs.append(rec["target"])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} done")

    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}
    print("ROUGE scores:", result)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()