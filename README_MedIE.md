# Biomedical Information Extraction System

Work in progress. A chemical–disease information extraction system built on the BC5CDR benchmark, using BiomedBERT as the underlying language model.

## Goal

The aim is to build an end-to-end pipeline that (a) identifies Chemical and Disease mentions in biomedical abstracts via named-entity recognition, and (b) links each mention to its MeSH identifier, using [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr) as the training and evaluation corpus.

## Current Status

| Component                          | Status          |
| ---------------------------------- | --------------- |
| Dataset exploration and statistics | Implemented     |
| BM25 retrieval baseline            | Implemented     |
| Preprocessing pipeline             | In progress     |
| NER fine-tuning (BiomedBERT)       | Planned         |
| Entity linking to MeSH             | Planned         |
| Evaluation with `seqeval`          | Planned         |

## What's Implemented

### Dataset exploration (`scripts/data_exploration.py`)

Loads BC5CDR via HuggingFace `datasets` and reports:

- Counts of Chemical and Disease entity mentions in the training split.
- Coverage of MeSH normalisation (how many entity mentions lack a MeSH identifier).
- Token-length distribution using the BiomedBERT tokeniser, and the number of documents exceeding the 512-token limit — relevant for planning chunking or long-context strategies during fine-tuning.

### BM25 retrieval baseline (`scripts/ir_baseline.py`)

Simple BM25 baseline over the training corpus using `rank-bm25`, returning the top-k most relevant documents for a given query. Serves as a reference point against which later retrieval or IE components can be compared.

## Models and Dataset

- **Model (planned):** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- **Dataset:** [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr) (BioCreative V Chemical-Disease Relation) — biomedical abstracts annotated with Chemical and Disease mentions and MeSH identifiers.

## Repo Structure

```
Med_IE_System/
├── main.py                      # entry point (stub)
├── pyproject.toml               # dependencies (managed with uv)
├── scripts/
│   ├── data_exploration.py      # BC5CDR statistics and tokenisation analysis
│   ├── ir_baseline.py           # BM25 retrieval baseline
│   └── preprocessing.py         # (in progress)
└── uv.lock
```

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Create a virtual environment:**
   ```bash
   uv venv
   ```
3. **Activate it:**
   ```bash
   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```
4. **Install dependencies:**
   ```bash
   uv sync
   ```
5. **For notebooks:** open the project in VS Code and select the `Med_IE_System` kernel.

### Running scripts

```bash
uv run python scripts/data_exploration.py
uv run python scripts/ir_baseline.py
```

Key dependencies: `transformers`, `datasets`, `torch`, `rank-bm25`, `seqeval`, `bioc`, `scikit-learn`.
