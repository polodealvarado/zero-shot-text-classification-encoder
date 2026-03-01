# Zero-Shot Text Classification

**Classify text into arbitrary categories — without retraining.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Hub-yellow.svg)](https://huggingface.co/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

A modular framework for zero-shot text classification that encodes texts and candidate labels into a shared embedding space using 7 model variants inspired by state-of-the-art approaches in NLP and information retrieval. Includes synthetic data generation, automated benchmarking, an LLM-as-Judge evaluation pipeline, and an interactive Gradio playground.

---

## Available Models

| Model | Approach | Inspiration | Key Idea |
|:------|:---------|:------------|:---------|
| **BiEncoder** | Shared encoder + dot product | [Sentence-BERT](https://arxiv.org/abs/1908.10084) | Simple baseline, fastest inference |
| **ProjectionBiEncoder** | Projection head + L2 norm + learnable temperature | [CLIP](https://arxiv.org/abs/2103.00020) | Learned similarity space with contrastive loss |
| **LateInteraction** | Token-level MaxSim scoring | [ColBERT](https://arxiv.org/abs/2004.12832) | Fine-grained token-level alignment |
| **PolyEncoder** | Learnable poly-codes + cross-attention | [Poly-encoders](https://arxiv.org/abs/1905.01969) | Text representation adapts to each label |
| **DynQuery** | Dynamic label queries via cross-attention | [DyREx](https://arxiv.org/abs/2210.15048) | Labels adapt to text context before scoring |
| **SpanClass** | Span extraction + top-K selection + label matching | [GLiNER](https://arxiv.org/abs/2311.08526) | Sub-sentence evidence spans scored per label |
| **ConvMatch** | Multi-scale CNN over pretrained embeddings | [TextCNN](https://arxiv.org/abs/1408.5882) | No transformer — fast n-gram pattern matching |

All models share a unified interface and support `save_pretrained()` / `from_pretrained()` / `push_to_hub()` via HuggingFace Hub.

---

## Project Structure

```
models/
├── base.py                  # BiEncoderModel — shared BERT + dot product + sigmoid
├── projection.py            # ProjectionBiEncoderModel — CLIP-inspired
├── late_interaction.py      # LateInteractionModel — ColBERT-style token-level MaxSim
├── polyencoder.py           # PolyEncoderModel — learnable poly-codes + cross-attention
├── dynquery.py              # DynQueryModel — DyREx-inspired dynamic label queries
├── spanclass.py             # SpanClassModel — GLiNER-inspired span-attentive classification
└── convmatch.py             # ConvMatchModel — multi-scale CNN, no transformer

scripts/
├── generate_data.py         # Synthetic data generation with Gemini 2.5 Flash
├── analyze_data.py          # Dataset quality analysis
├── train.py                 # Training loop with checkpointing and auto-push to Hub
├── benchmark.py             # Multi-model benchmark with ranked results table
├── llm_judge.py             # LLM-as-Judge evaluation with adaptive difficulty
├── dataset.py               # ZeroShotDataset + negative sampling + data loading
└── playground.py            # Interactive Gradio playground

config.yaml                  # All configuration (model, training, data, benchmark)
```

---

## Pipeline Overview

```
 generate_data.py          train.py              benchmark.py         llm_judge.py
 ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │  Gemini 2.5   │ ──▶  │  Fine-tune    │ ──▶  │  Compare all  │ ──▶  │  Adaptive     │
 │  Flash        │      │  BERT encoder │      │  variants     │      │  difficulty   │
 │               │      │               │      │               │      │  evaluation   │
 │  Synthetic    │      │  Checkpoints  │      │  F1, P, R     │      │               │
 │  multi-label  │      │  + best model │      │  Latency      │      │  Gemini       │
 │  data         │      │  selection    │      │  Params       │      │  generates +  │
 │               │      │               │      │               │      │  judges       │
 └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
        │                      │                      │                      │
    data/{ts}/          model_output/          results/              results/
                        {encoder}/{type}/{ts}  benchmark.json        llm_judge_results.json
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/polodealvarado/biencoder_project.git
cd biencoder_project

# Install dependencies with uv
uv sync
```

---

## Data Generation

Generate synthetic multi-label training data using Gemini 2.5 Flash:

```bash
uv run python scripts/generate_data.py
uv run python scripts/generate_data.py --target-samples 500 --samples-per-request 25
```

Each run produces a timestamped HuggingFace `DatasetDict` with pre-computed `train`/`test` splits. All downstream scripts automatically resolve the latest generated dataset.

Set `dataset.export_json: true` in `config.yaml` to also export the full dataset as `data/synthetic_data.json`.

### Dataset Analysis

```bash
uv run python scripts/analyze_data.py
```

Computes: label distribution & entropy, vocabulary TTR & hapax legomena, text length stats, multi-label complexity, label co-occurrence, train/test label overlap, and stratification balance.

---

## Training

```bash
# Train default model (BiEncoder, bert-base-uncased)
uv run python scripts/train.py

# Train a specific variant with a different encoder
uv run python scripts/train.py --model-type late_interaction --encoder roberta-base
```

Available model types: `biencoder` | `projection_biencoder` | `late_interaction` | `polyencoder` | `dynquery` | `spanclass` | `convmatch`

### Auto-push to HuggingFace Hub

When `training.push_to_hub: true` in `config.yaml`, the training script automatically manages model uploads:

1. **Repo creation**: The repo ID is derived as `<hf_username>/<model_type>` (e.g., `polodealvarado/biencoder`, `polodealvarado/late_interaction`). If the repo doesn't exist, it is created automatically.
2. **Quality gate**: If the repo already exists, the new model's best metric (F1 by default) is compared against the current Hub version. The upload only proceeds if the new model improves on the existing one.
3. **Uploaded artifacts**: Model weights, tokenizer, `training_meta.json`, and an auto-generated model card.

This means training any variant will create and maintain its own dedicated repo on the Hub without manual intervention. Requires `HF_TOKEN` in `.env`.

---

## Benchmarking

Compare all trained models without retraining. Fetches models from HuggingFace Hub by default, with fallback to local `model_output/` scan.

```bash
# Benchmark from HF Hub (reads benchmark.models from config.yaml)
uv run python scripts/benchmark.py

# Benchmark from local directory
uv run python scripts/benchmark.py --model-dir model_output
```

Output:

```
======================================================================
  BENCHMARK — 4 model(s)
======================================================================
  Device: mps
  Test: 200 samples
  - bert_base/late_interaction (bert-base-uncased)
  - bert_base/projection_biencoder (bert-base-uncased)
  - bert_base/biencoder (bert-base-uncased)
  - bert_base/polyencoder (bert-base-uncased)
======================================================================

  [1/4] bert_base/late_interaction (bert-base-uncased)
         Inference: 1.87 ms/sample
  [2/4] bert_base/projection_biencoder (bert-base-uncased)
         Inference: 1.31 ms/sample
  ...

=============================================================================================================
BENCHMARK RESULTS
=============================================================================================================
Encoder / Model                                Params   Train(s) Infer(ms/sample)        P        R       F1
-------------------------------------------------------------------------------------------------------------
bert_base/late_interaction                109,580,421       52.3         1.87   0.8701   0.8312   0.8502 << BEST
bert_base/projection_biencoder            109,744,645       48.1         1.31   0.8512   0.8103   0.8303
bert_base/biencoder                       109,482,245       45.2         1.23   0.8234   0.7891   0.8059
bert_base/polyencoder                     109,612,037       55.0         1.54   0.7901   0.7503   0.7697    worst
=============================================================================================================

======================================================================
  BENCHMARK COMPLETE — 4 model(s) evaluated
======================================================================
  Best F1: 0.8502 — bert_base/late_interaction
  Results saved to results/benchmark.json
======================================================================
```

### LLM-as-Judge Evaluation

Evaluates models with **adaptive difficulty**: Gemini generates evaluation samples at progressive difficulty levels (Easy → Medium → Hard), the model predicts on them, and Gemini judges the predictions. Difficulty adapts automatically based on model performance each round.

| Level | Name | Correct Labels | Candidates | Description |
|:------|:-----|:---------------|:-----------|:------------|
| 1 | Easy | 1 | 3-4 | Clear text, semantically distant labels |
| 2 | Medium | 2-3 | 5-6 | Multi-label, some semantic proximity |
| 3 | Hard | 2-4 | 7-8 | Ambiguous text, semantically close labels |

Adaptation rules: avg ≥ 8 → level up, avg < 5 → level down, otherwise stay.

```bash
# Default: 6 rounds x 5 samples
uv run python scripts/llm_judge.py

# Custom rounds and batch size
uv run python scripts/llm_judge.py --num-rounds 10 --batch-size 3

# Specific model from HuggingFace Hub
uv run python scripts/llm_judge.py --model polodealvarado/polyencoder
```

---

## Interactive Playground

A Gradio-based UI for testing models interactively. Supports both local and HuggingFace Hub models.

```bash
uv run python scripts/playground.py
```

---

## Inference

Load a pre-trained model from a local path or HuggingFace Hub:

```python
from models import BiEncoderModel

model = BiEncoderModel.from_pretrained("polodealvarado/biencoder")

predictions = model.predict(
    texts=["The stock market crashed yesterday amid inflation fears."],
    labels=[["Finance", "Sports", "Biology", "Politics"]],
)
# [{"text": "The stock market...", "scores": {"Finance": 0.97, "Politics": 0.42, ...}}]
```

---

## Configuration

All settings are centralized in `config.yaml`:

```yaml
training:
  type: "polyencoder"                         # biencoder | projection_biencoder | late_interaction | polyencoder | dynquery | spanclass | convmatch
  base_model: "bert-base-uncased"
                                              # Variant-specific parameters (ignored by models that don't use them)
  projection_dim: 256                         # ProjectionBiEncoder / ConvMatch: projection head output dim
  contrastive_alpha: 0.1                      # ProjectionBiEncoder: weight for InfoNCE loss
  num_poly_codes: 16                          # PolyEncoder: number of learnable poly-codes
  token_projection_dim: 128                   # LateInteraction: token projection dim
  num_attention_heads: 4                      # DynQuery: number of cross-attention heads
  max_span_width: 5                           # SpanClass: maximum span width in tokens
  top_k_spans: 8                              # SpanClass: number of top spans to select
  num_filters: 128                            # ConvMatch: filters per convolution kernel
  seed: 42
  num_steps: 1000
  batch_size: 2
  learning_rate: 0.00002
  optimizer: "adamw"
  log_every: 50
  save_every: 10
  eval_every: 25
  model_dir: "model_dir"
  push_to_hub: true
  best_metric: "f1"
  save_total_limit: 2
  private: false

dataset:
  data_dir: "data_dir"                        # base dir for generated datasets (each run: data/{YYYYMMDD_HHMMSS}/)
  target_samples: 1000                        # number of samples to generate
  min_labels: 1
  max_labels: 10
  max_negatives: 3
  test_ratio: 0.2
  export_json: false                           # export dataset as data/synthetic_data.json
  push_to_hub: true                           # auto-push dataset to HF Hub after generation
  dataset_repo_id: ""
  private: false

benchmark:
  models:                                     # HF Hub repo IDs for default benchmark
    # - "username/zero-shot-biencoder"
    # - "username/zero-shot-polyencoder"
```

---

## HuggingFace Hub Integration

Both models and datasets can be automatically pushed to HuggingFace Hub:

- **Models** (`training.push_to_hub`): Uploaded only if the new model improves on the current Hub version (compared by F1). Repo ID is derived as `<username>/<model_type>`.
- **Datasets** (`dataset.push_to_hub`): Uploaded to the repo specified in `dataset.dataset_repo_id`.

Each model repo includes: weights, tokenizer, `training_meta.json`, and an auto-generated model card.

---

## Documentation

| Resource | Description |
|:---------|:------------|
| [Development Notes](documentation/development.md) | Architecture decisions, implementation details, and design rationale |
| [Literature Review](documentation/literature_review.md) | Papers 2019-2025, approach comparison, speed/accuracy trade-offs |
| [Novel Architecture](documentation/novel_architecture.md) | Novel architecture proposals inspired by Zaratiana's research |
