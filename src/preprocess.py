"""
Preprocessing pipeline for MeetingBank transcripts:
1. Clean transcript/summary text (filler words, whitespace)
2. Tokenize with BART tokenizer
3. Chunk transcripts into <=1024-token pieces
4. Greedy target matching: align each chunk with the summary
   sentences that maximize ROUGE-1 overlap
"""

import re
import json
import argparse
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import BartTokenizerFast
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MAX_CHUNK_TOKENS = 1024
TOKENIZER_NAME = "facebook/bart-base"

FILLER_WORDS = [
    r"\bum\b", r"\buh\b", r"\boh\b", r"\bbasically\b",
    r"\bliterally\b", r"\breally\b", r"\byou know\b", r"\bi mean\b",
]
FILLER_PATTERN = re.compile("|".join(FILLER_WORDS), flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Remove filler words/phrases and normalize whitespace."""
    if not text:
        return ""
    text = FILLER_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_transcript(text: str, tokenizer, max_tokens: int = MAX_CHUNK_TOKENS):
    """
    Split a cleaned transcript into non-overlapping chunks,
    each within max_tokens when tokenized by the BART tokenizer.
    Splits on sentence boundaries so chunks don't cut mid-sentence.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))

        # a single sentence longer than max_tokens - hard-truncate it alone
        if sent_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
            truncated_ids = tokenizer.encode(
                sent, add_special_tokens=False, truncation=True, max_length=max_tokens
            )
            chunks.append(tokenizer.decode(truncated_ids, skip_special_tokens=True))
            continue

        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [sent], sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def greedy_target_match(chunk: str, summary_sentences: list, scorer) -> str:
    """
    Greedily select summary sentences that maximize ROUGE-1 gain
    against this chunk, stopping when no sentence improves the score.
    Returns the matched summary text for this chunk.
    """
    selected = []
    selected_text = ""
    best_score = 0.0

    remaining = list(summary_sentences)
    improved = True

    while improved and remaining:
        improved = False
        best_candidate = None
        best_candidate_score = best_score

        for sent in remaining:
            candidate_text = (selected_text + " " + sent).strip()
            score = scorer.score(candidate_text, chunk)["rouge1"].fmeasure
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = sent

        if best_candidate is not None:
            selected.append(best_candidate)
            selected_text = (selected_text + " " + best_candidate).strip()
            best_score = best_candidate_score
            remaining.remove(best_candidate)
            improved = True

    return selected_text if selected_text else (summary_sentences[0] if summary_sentences else "")


def preprocess_split(dataset_split, tokenizer, scorer):
    """Run cleaning, chunking, and greedy matching over one dataset split."""
    records = []

    for row in dataset_split:
        transcript = clean_text(row["transcript"])
        summary = clean_text(row["summary"])

        if not transcript or not summary:
            continue

        chunks = chunk_transcript(transcript, tokenizer)
        summary_sentences = sent_tokenize(summary)

        for chunk in chunks:
            matched_summary = greedy_target_match(chunk, summary_sentences, scorer)
            if matched_summary:
                records.append({
                    "id": row["id"],
                    "uid": row["uid"],
                    "source": chunk,
                    "target": matched_summary,
                })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and dataset...")
    tokenizer = BartTokenizerFast.from_pretrained(TOKENIZER_NAME)
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    dataset = load_dataset("huuuyeah/meetingbank")

    for split in args.splits:
        print(f"Processing split: {split} ({len(dataset[split])} rows)")
        records = preprocess_split(dataset[split], tokenizer, scorer)

        out_path = output_dir / f"{split}_processed.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  -> {len(records)} chunk-summary pairs written to {out_path}")


if __name__ == "__main__":
    main()