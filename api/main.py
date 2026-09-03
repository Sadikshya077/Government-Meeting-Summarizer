"""
FastAPI backend for BriefGov.
Run from project root: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from inference import Summarizer

MODEL_DIR = "models/briefgov-bart-final"

summarizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    summarizer = Summarizer(MODEL_DIR)
    yield


app = FastAPI(title="BriefGov API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only — restrict this if you ever deploy publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    transcript: str = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Empty transcript")
    summary = summarizer.summarize(req.transcript)
    return SummarizeResponse(summary=summary)


@app.get("/health")
def health():
    return {"status": "ok"}