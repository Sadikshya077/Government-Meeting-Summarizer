# 🏛️ BriefGov - Government Meeting Summarizer

BriefGov is an abstractive text summarization system for government meeting
transcripts. It fine-tunes a BART model on the [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank)
dataset to generate concise summaries of city council and public meeting
transcripts, served through a FastAPI backend with a simple browser UI. 🎤➡️📝

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-BART-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal)
![ROUGE--1](https://img.shields.io/badge/ROUGE--1-41.93-brightgreen)

## 📋 Overview

- **Task**: abstractive summarization of long government meeting transcripts
- **Model**: `facebook/bart-base`, fine-tuned on MeetingBank
- **Pipeline**: transcript cleaning → chunking → greedy target alignment →
  fine-tuning → evaluation → API → UI
- **Serving**: FastAPI backend (`/summarize` endpoint) + static HTML/JS demo UI

## 🧩 Architecture

```
Raw transcript
      │
      ▼
Clean text (remove filler words, normalize whitespace)
      │
      ▼
Chunk into ≤1024-token pieces (sentence-boundary aware)
      │
      ▼
Greedy ROUGE-1 alignment of summary sentences to each chunk
      │
      ▼
Fine-tune facebook/bart-base (Seq2SeqTrainer, 3 epochs)
      │
      ▼
FastAPI /summarize endpoint ── loads model once at startup
      │
      ▼
Static HTML/JS UI ── calls the API, displays the summary
```

## 📁 Project Structure

```text
briefgov/
├── api/
│   └── main.py              # FastAPI app (/summarize, /health)
├── data/
│   ├── train_trimmed.jsonl
│   ├── validation_trimmed.jsonl
│   └── test_trimmed.jsonl
├── models/
│   └── briefgov-bart-final/ # fine-tuned model weights + tokenizer
├── notebooks/
│   └── 01_explore_dataset.ipynb
├── results/
│   └── rouge_scores.json
├── src/
│   ├── preprocess.py        # cleaning, chunking, target alignment
│   ├── trim_targets.py      # truncates over-long targets to 120 tokens
│   ├── inference.py         # Summarizer class used by API and evaluation
│   └── run_evaluation.py    # ROUGE evaluation on the test set
├── web/
│   └── index.html           # simple browser demo UI
└── requirements.txt
```

## ⚙️ Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

The fine-tuned model (`models/briefgov-bart-final/`) was trained on a Kaggle
GPU notebook and downloaded separately — see `notebooks/01_explore_dataset.ipynb`
for the data exploration and `src/preprocess.py` / `src/trim_targets.py` for
the preprocessing pipeline used to build the training data.

## 🚀 Running the API

```bash
uvicorn api.main:app --port 8000
```

Test it:

**PowerShell (Windows):**
```powershell
$body = @{ transcript = "Your transcript text here..." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/summarize -Method Post -Body $body -ContentType "application/json"
```

**curl (Mac/Linux):**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Your transcript text here..."}'
```

Or visit `http://localhost:8000/docs` for interactive Swagger API docs. 📎

## 🖥️ Running the Demo UI

With the API running, open `web/index.html` directly in a browser (or serve it
with VS Code's Live Server extension). Paste a transcript and click
**Summarize**.

## 🧪 Running Evaluation

```bash
python src/run_evaluation.py --limit 100   # quick sample
python src/run_evaluation.py               # full test set
```

## 📊 Results

Evaluated on a 100-sample subset of the held-out test set (chunk-level
source/target pairs):

| Metric   | Score |
|----------|-------|
| ROUGE-1  | 41.93 |
| ROUGE-2  | 26.04 |
| ROUGE-L  | 35.58 |
| ROUGE-Lsum | 35.47 |

## ⚠️ Notes and Limitations

- **Summary style**: MeetingBank's reference summaries are written in a
  formal, clerk-style register (e.g. *"Recommendation to receive supporting
  documentation into the record, conclude the public hearing, and..."*),
  reflecting how government meeting minutes are actually recorded. The
  fine-tuned model correctly adopts this style rather than producing
  free-form plain-English summaries — this is expected behavior given the
  training data, not a defect.
- **Evaluation scope**: results above are on chunk-level source/target pairs
  (the same unit the model was trained on), not full raw transcripts. This
  is the standard evaluation unit for this pipeline but tends to score
  higher than whole-document summarization would.
- **Text-only input**: this version accepts transcript text directly. Audio
  transcription (e.g. via WhisperX) is a natural extension but out of scope
  for this version.
- **Chunking at inference**: longer transcripts are split into ≤1024-token
  chunks (sentence-boundary aware) and summarized independently, then joined.

## 🛠️ Tech Stack

Python, PyTorch, Hugging Face Transformers, Datasets, Evaluate, FastAPI,
Kaggle (GPU fine-tuning), NLTK.