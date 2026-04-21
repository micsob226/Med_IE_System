# Sentiment Analysis on Movie Reviews with BERT and RoBERTa

Fine-tuning `bert-base-uncased` and `roberta-base` for binary sentiment classification on the Stanford SST-2 dataset, and comparing their performance.

## Overview

This project compares two transformer encoders on a standard sentiment-analysis benchmark. The pipeline covers data loading and filtering, tokenisation, fine-tuning with the HuggingFace `Trainer` API, evaluation on a held-out test set, and analysis of the learned hidden-state representations across layers.

## Results

Test-set accuracy on SST-2:

| Model          | Test Accuracy |
| -------------- | ------------- |
| BERT (base)    | 88.4%         |
| RoBERTa (base) | 90.9%         |

RoBERTa outperforms BERT by ~2.5 percentage points under the same training configuration, consistent with findings in the literature.

## Models and Dataset

- **Models:** `bert-base-uncased`, `roberta-base` (HuggingFace Transformers)
- **Dataset:** [Stanford SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) — binary sentiment labels on movie-review sentences
- **Preprocessing:** Filtered empty/whitespace-only entries, shuffled with fixed seed (42), sampled subsets for training and evaluation
- **Training config:** batch size 32, 4 epochs, learning rate 1e-5, weight decay 0.08, evaluation and checkpointing each epoch, best model loaded at end
- **Evaluation:** Accuracy (via `evaluate`), confusion matrices visualised with seaborn

## What's in the repo

- `Binary_Sentiment_Analysis.ipynb` — end-to-end notebook: data loading, tokenisation, BERT fine-tuning and evaluation, RoBERTa fine-tuning and evaluation, confusion matrices, and hidden-state export for the TensorBoard Embedding Projector.

## Requirements

```
transformers
datasets
evaluate
accelerate
torch
scikit-learn
seaborn
matplotlib
```

## How to Run

The notebook is designed to run on Google Colab (mounts Google Drive for checkpoint storage). To run locally, remove the Drive-mount cell and adjust the `output_dir` paths in the `TrainingArguments`.

```bash
pip install transformers datasets evaluate accelerate
jupyter notebook Binary_Sentiment_Analysis.ipynb
```

## Notes

The project additionally exports per-layer hidden states to TSV files compatible with the [TensorBoard Embedding Projector](https://projector.tensorflow.org/), allowing inspection of how sentence representations evolve across transformer layers.
