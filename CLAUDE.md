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
# Run a module script directly
uv run python information_retrieval/tfidf.py

# Launch Jupyter for notebooks
uv run jupyter notebook
```

There are no tests or linting configurations yet.

## Architecture

Each NLP topic lives in its own directory. Currently only `information_retrieval/` is implemented; others are planned per the README roadmap.

### `information_retrieval/`

- **`tfidf_corpus.py`** — Sample corpora used as input data (e.g., `dataset_IR_v1`, `dataset_tfidf_v1`).
- **`tfidf.py`** — Core implementation. Contains two classes:
  - `TFIDF`: Builds vocab, BoW matrix, inverted index, and computes TF (log-normalized), IDF, and TF-IDF scores.
  - `TFIDFRetriever`: Wraps `TFIDF` to perform query retrieval using the inverted index for candidate filtering, then ranks by cosine similarity.
- **`cosinse_sim.py`** — Standalone cosine similarity prototype (exploratory script, not imported elsewhere).
- **`IR Notebook.ipynb`** — Interactive notebook for experiments.

The retrieval pipeline: `corpus → TFIDF (preprocess → build_vocab → build_bow → build_inverted_index → compute_tfidf_with_bow) → TFIDFRetriever.retrieve(query)`.

Imports within `information_retrieval/` use bare module names (e.g., `from tfidf_corpus import *`), so scripts must be run from within that directory or with the path adjusted.

## Adding New Modules

Follow the pattern of `information_retrieval/`: create a new directory for each topic, with a corpus/data file, a core implementation file, and optionally a Jupyter notebook.