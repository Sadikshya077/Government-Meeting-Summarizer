"""
Summarizer wrapper: chunks a transcript the same way preprocess.py did,
runs each chunk through the fine-tuned BART model, and joins the results.
"""

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize

MAX_INPUT_TOKENS = 1024
MAX_TARGET_TOKENS = 128


class Summarizer:
    def __init__(self, model_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BartForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def _chunk(self, text: str, max_tokens: int = MAX_INPUT_TOKENS):
        sentences = sent_tokenize(text)
        chunks, current, current_len = [], [], 0
        for sent in sentences:
            n = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if current_len + n > max_tokens:
                if current:
                    chunks.append(" ".join(current))
                current, current_len = [sent], n
            else:
                current.append(sent)
                current_len += n
        if current:
            chunks.append(" ".join(current))
        return chunks

    @torch.no_grad()
    def summarize_chunk(self, text: str) -> str:
        """Summarize a single chunk that's already within the token limit."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_length=MAX_TARGET_TOKENS,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize(self, transcript: str) -> str:
        """Summarize a full (possibly long) transcript by chunking first."""
        chunks = self._chunk(transcript)
        chunk_summaries = [self.summarize_chunk(c) for c in chunks]
        return " ".join(chunk_summaries)