# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLPBasics is a from-scratch study implementation of classical and modern NLP techniques, following *Speech and Language Processing* (Jurafsky & Martin, 3rd ed.). The goal is mathematical understanding and manual implementation rather than using high-level library wrappers. The primary language is Brazilian Portuguese for corpora and comments.

## Environment Setup

The project uses `uv` for dependency management (Python 3.12):

```bash
uv sync                         # Create venv and install dependencies
source .venv/bin/activate       # Activate venv (Linux/macOS)
uv run python module/script.py  # Run without activating
```

The local virtualenvs `nlp_env/` and `.venv/` are gitignored. Dependencies are declared in `pyproject.toml`; `uv.lock` pins exact versions.

## Running Code

```bash
# Run a retriever script directly (from repo root)
uv run python information_retrieval/tfidf_retriever.py
uv run python information_retrieval/bm25_retriever.py

# Launch Jupyter for notebooks
uv run jupyter notebook
```

There are no tests or linting configurations yet.

## Architecture

Each NLP topic lives in its own directory. Currently only `information_retrieval/` is implemented; others are planned per the README roadmap.

### `information_retrieval/`

Class hierarchy:

```
BaseRetriever (ABC)       ← template method for retrieve()
    ├── TFIDFRetriever    ← TF×IDF scoring
    └── BM25Retriever     ← Okapi BM25 scoring

CorpusIndex               ← composed into BaseRetriever (not a base class)
```

Files:

- **`corpus.py`** — Sample corpora (`dataset_IR_v1`, `dataset_tfidf_v1`, `query_tfidf_v1`).
- **`index.py`** — `CorpusIndex`: preprocessing, sorted vocab, raw count matrix, inverted index. No scoring logic.
- **`base_retriever.py`** — `BaseRetriever(ABC)`: builds `CorpusIndex`, owns `retrieve()` template and `_cosine_similarity()`.
- **`tfidf_retriever.py`** — `TFIDFRetriever`: implements log-normalized TF × IDF scoring.
- **`bm25_retriever.py`** — `BM25Retriever`: implements Okapi BM25 scoring (k1=1.5, b=0.75 defaults).
- **`cosinse_sim.py`** — Standalone cosine similarity prototype (exploratory, not imported elsewhere).
- **`IR Notebook.ipynb`** — Interactive notebook for experiments.

Retrieval pipeline: `corpus → BaseRetriever.__init__ (CorpusIndex → build_score_matrix) → retrieve(query)`.

Imports within `information_retrieval/` use bare module names, so scripts must be run from within that directory or with the path adjusted.

## Adding New Modules

Follow the pattern of `information_retrieval/`: create a new directory for each topic, with a corpus/data file, a core implementation file, and optionally a Jupyter notebook.