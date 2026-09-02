"""
Truncate over-long 'target' summaries
in already-generated jsonl files to a safe decoder length.

"""

import json
from pathlib import Path
from transformers import BartTokenizerFast

MAX_TARGET_TOKENS = 120  # safely under BART's typical 128-token decoder limit
TOKENIZER_NAME = "facebook/bart-base"


def trim_target(text: str, tokenizer, max_tokens: int = MAX_TARGET_TOKENS) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    tokenizer = BartTokenizerFast.from_pretrained(TOKENIZER_NAME)
    data_dir = Path("data")

    for split in ["train", "validation", "test"]:
        in_path = data_dir / f"{split}_processed.jsonl"
        out_path = data_dir / f"{split}_trimmed.jsonl"

        if not in_path.exists():
            print(f"Skipping {split} — file not found")
            continue

        n = 0
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                rec["target"] = trim_target(rec["target"], tokenizer)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

        print(f"{split}: trimmed {n} records -> {out_path}")


if __name__ == "__main__":
    main()