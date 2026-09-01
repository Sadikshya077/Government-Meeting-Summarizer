# BriefGov: Abstractive Summarization of Government Meeting Transcripts

BriefGov fine-tunes a pre-trained **BART** model to generate concise, abstractive
summaries of long government (city council) meeting transcripts. It is trained and
evaluated on **MeetingBank**, a public benchmark dataset of 1,366 US city council
meetings with paired transcripts and official human-written summaries.

## Problem

Government meetings generate long, dense transcripts (avg. ~28k tokens per meeting)
that are time-consuming for officials, staff, and the public to review. This project
automates that process — turning a raw meeting transcript into a short, readable
summary of the key points and decisions.

## Approach

1. **Preprocessing** — clean transcripts (remove filler words, normalize whitespace),
   tokenize with the BART tokenizer, and split long transcripts into chunks that fit
   within BART's 1024-token input limit.
2. **Greedy target matching** — since reference summaries are written for the whole
   meeting (not per-chunk), each transcript chunk is aligned with the subset of
   summary sentences that maximizes ROUGE-1 overlap.
3. **Fine-tuning** — `facebook/bart-base` is fine-tuned on the resulting
   (chunk, summary) pairs using the Hugging Face `Seq2SeqTrainer`.
4. **Evaluation** — model performance is measured with ROUGE-1, ROUGE-2, ROUGE-L,
   and ROUGE-Lsum against held-out reference summaries.
5. **Serving** — the fine-tuned model is exposed via a lightweight **FastAPI**
   endpoint that accepts a transcript and returns a generated summary.

## Dataset

[MeetingBank](https://arxiv.org/abs/2305.17529) (Hu et al., 2023) — 1,366 city council
meetings across 6 US cities (Seattle, King County, Denver, Boston, Alameda, Long Beach),
totaling 3,579 hours of meetings and 6,892 summarization instances. No manual data
collection or labeling was required.

## Project Structure

\```
briefgov/
├── data/           # dataset (not tracked in git — see .gitignore)
├── notebooks/       # exploratory analysis
├── src/              # preprocessing, training, evaluation scripts
├── api/               # FastAPI app for serving the model
├── models/          # fine-tuned model checkpoints (not tracked in git)
├── results/         # evaluation outputs, plots, logs
├── requirements.txt
└── README.md
\```

## Status

🚧 In progress.

## Tech Stack

Python · PyTorch · Hugging Face Transformers & Datasets · ROUGE · FastAPI