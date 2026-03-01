# Development Notes

---

## 1. Project Structure and Setup

### Structure

A directory structure with `data/`, `scripts/`, `model.py`, `dataset.py`, `config.yaml`, and `README.md`.

### Implementation

The project evolved beyond the suggested flat structure into a modular architecture:

```
models/                      # Package instead of single model.py
├── __init__.py              # MODEL_REGISTRY for dynamic dispatch
├── base.py                  # BiEncoderModel
├── projection.py            # ProjectionBiEncoderModel
├── late_interaction.py       # LateInteractionModel
├── polyencoder.py           # PolyEncoderModel
├── dynquery.py              # DynQueryModel
├── spanclass.py             # SpanClassModel
└── convmatch.py             # ConvMatchModel

scripts/
├── generate_data.py         # Synthetic data generation
├── analyze_data.py          # Dataset quality analysis
├── train.py                 # Training loop
├── benchmark.py             # Multi-model benchmark
├── llm_judge.py             # LLM-as-Judge evaluation
├── dataset.py               # Dataset class + data loading
└── playground.py            # Gradio interactive UI

config.yaml                  # Centralized configuration
config.example.yaml          # Template for new users
```

**Why a `models/` package instead of a single `model.py`?** Each model variant has ~150-200 lines of architecture code. Putting four variants in one file would create a 600+ line monolith. The package approach keeps each variant self-contained while sharing a common interface through `MODEL_REGISTRY` in `__init__.py`. This registry enables dynamic model dispatch throughout the codebase — `train.py`, `benchmark.py`, and `playground.py` all instantiate models via `MODEL_REGISTRY[model_type]` without importing specific classes.

### Configuration

The `config.yaml` centralizes all settings in three sections:

- **`training`**: Model architecture, hyperparameters, checkpointing, Hub push settings
- **`dataset`**: Data generation parameters, negative sampling, Hub push settings
- **`benchmark`**: HF Hub repo IDs for model comparison

CLI flags (`--model-type`, `--encoder`, `--config`) can override config values, giving flexibility without editing YAML. Variant-specific parameters (`projection_dim`, `num_poly_codes`, `token_projection_dim`, `contrastive_alpha`) live under `training` and are silently ignored by models that don't use them — this avoids needing separate config sections per variant.

---

## 2. Model Development

### Implementation

A BiEncoder model with shared encoder, mask-aware mean pooling, dot product scoring, sigmoid activation, and HuggingFace Hub integration (`save_pretrained`, `from_pretrained`, `push_to_hub`).

### What was implemented

Four model variants, all sharing the same interface:

| Model | Inspiration | Scoring Mechanism | Key Idea |
|:------|:------------|:------------------|:---------|
| BiEncoder | Sentence-BERT | Dot product | Simple baseline, fastest inference |
| ProjectionBiEncoder | CLIP | Cosine similarity + learned temperature | Learned metric space with contrastive loss |
| LateInteraction | ColBERT | Token-level MaxSim | Fine-grained token alignment |
| PolyEncoder | Poly-encoders | Attention-weighted dot product | Text representation adapts per label |

All four inherit from `nn.Module` and `PyTorchModelHubMixin`, providing `save_pretrained()`, `from_pretrained()`, and `push_to_hub()` out of the box.

### BiEncoderModel (base.py)

The baseline implementation:

1. **Shared encoder**: Single `AutoModel` encodes both texts and labels
2. **Mask-aware mean pooling**: `(last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)` — ensures padding tokens don't contaminate embeddings
3. **Dot product scoring**: `scores = bmm(label_embeddings, text_embeddings)` produces raw similarity
4. **Sigmoid activation**: Converts to independent per-label probabilities (multi-label, not softmax)
5. **Masked BCE loss**: `BCE(scores, targets) * mask` — only computes loss on real labels, ignores padding

**Why sigmoid instead of softmax?** This is multi-label classification — a text can belong to multiple categories simultaneously. Softmax would force probabilities to sum to 1, creating competition between labels. Sigmoid treats each label independently.

### ProjectionBiEncoderModel (projection.py)

Extends the BiEncoder with three CLIP-inspired additions:

1. **Projection head**: Linear layer (768 → 256) maps embeddings to a lower-dimensional space. This lets the model learn a task-specific similarity metric rather than relying on raw BERT embeddings.

2. **L2 normalization**: After projection, embeddings are normalized to unit length. This constrains all representations to a hypersphere, making dot product equivalent to cosine similarity and stabilizing training.

3. **Learnable temperature**: Initialized to `log(1/0.07) ≈ 2.66` (CLIP's initialization). Scales similarity scores before sigmoid: `scores = sigmoid(cosine_sim * exp(temperature))`. This lets the model learn how "peaked" its confidence distribution should be.

**Dual loss function**: Supports standard masked BCE plus an optional InfoNCE contrastive loss:

```
total_loss = BCE_loss + contrastive_alpha * InfoNCE_loss
```

The InfoNCE loss is symmetric — it computes both text→label and label→text contrastive losses and averages them. This pulls matching text-label pairs together while pushing non-matching pairs apart in the embedding space, complementing the per-sample BCE objective with a batch-level structure signal.

### LateInteractionModel (late_interaction.py)

Inspired by ColBERT, this model operates at the **token level** instead of pooling to single vectors:

1. **No pooling**: Keeps all token embeddings from the encoder
2. **Token projection**: Linear layer (768 → 128) projects each token to a lower dimension
3. **L2 normalization**: Each token embedding is normalized independently
4. **MaxSim scoring**: For each text token, find the maximum similarity with any label token, then average across text tokens

```
MaxSim(text, label) = (1/|text_tokens|) * Σ_i max_j(text_token_i · label_token_j)
```

**Why MaxSim?** It captures fine-grained semantic alignment. If a text says "The stock market crashed yesterday" and a label is "Finance", MaxSim finds that "stock" and "market" individually have high similarity with "Finance", even if other text tokens don't. This is richer than a single pooled vector comparison.

**Trade-off**: More expressive but slower — complexity is O(text_tokens × label_tokens) per pair, versus O(1) for dot product in BiEncoder.

### PolyEncoderModel (polyencoder.py)

The most architecturally complex variant, using **learnable poly-codes** for adaptive text representation:

1. **Poly-codes**: A learnable parameter matrix `[m, D]` (default m=16) initialized with small random values (stddev=0.02). These are m different "lenses" through which to view the text.

2. **First attention (poly-attention)**: Each poly-code attends over all text tokens:
   ```
   attention = softmax(poly_codes @ text_tokens.T)
   context_vectors = attention @ text_tokens  → [B, m, D]
   ```
   This produces m different "views" or "aspects" of the text.

3. **Second attention (label-conditioned)**: Each label embedding attends over the m context vectors:
   ```
   attention = softmax(label_embedding @ context_vectors.T)
   text_repr = attention @ context_vectors  → [B, D]
   ```
   The text representation **adapts to each candidate label**.

4. **Dot product + sigmoid**: Final score from adapted text representation and label embedding.

**Why is adaptive representation important?** Consider the text "Apple released a new iPhone with improved battery life." When comparing against the label "Technology", the poly-codes might emphasize "iPhone" and "released". When comparing against "Business", they might emphasize "Apple" and "released". The BiEncoder produces one fixed text embedding regardless of label — the PolyEncoder produces a different text embedding for each candidate.

**Trade-off**: More expressive than BiEncoder (label-conditioned), more efficient than full cross-attention (text encoded once, lightweight label attention), but slower than BiEncoder due to the dual attention mechanism.

### Unified Interface

All four models implement the same methods:

- `forward(texts, batch_labels)` → `(scores, mask)`
- `compute_loss(scores, targets, mask)` → scalar loss
- `predict(texts, batch_labels)` → list of `{text, scores}` dicts

This allows `train.py`, `benchmark.py`, and `playground.py` to work with any variant through `MODEL_REGISTRY` without variant-specific code paths.

---

## 3. Data Processing

### Implementation

A dataset class with data loading, preprocessing, and negative sampling.

### What was implemented

#### Negative Sampling (dataset.py)

The core training strategy. For each text, the dataset dynamically samples incorrect labels at each epoch:

```
Text: "The stock market crashed"
Positive labels: ["Finance", "Economy"]         → target = 1.0
Sampled negatives: ["Biology", "Sports", "Music"] → target = 0.0
```

Key design decisions:

1. **Random number of negatives** (1 to `max_negatives`): Prevents the model from learning a fixed positive/negative ratio. If we always used 3 negatives, the model could learn "~40% of labels are correct" as a shortcut instead of actually understanding semantics.

2. **Shuffled order**: Positives and negatives are shuffled together before returning. Without this, the model could learn "first labels are always positive" — a positional shortcut.

3. **Global label pool from train+test**: Negatives are sampled from ALL labels in the dataset, not just the current split. This exposes the model to the full label vocabulary during training, improving generalization.

4. **Dynamic per-step**: Negatives are re-sampled every time `__getitem__` is called, so the model sees different negative combinations across DataLoader passes. Since training is step-based, a single pass through the DataLoader covers all samples once; when the DataLoader is exhausted the outer `while step < num_steps` loop starts a new pass with fresh negatives. This acts as a form of data augmentation.

5. **Deterministic evaluation via seed**: The test dataset accepts an optional `seed` parameter. When set, each `__getitem__(idx)` creates a local `random.Random(seed + idx)` so the same negatives and shuffle order are produced every time for the same sample. This makes evaluation fully deterministic — the same model state always produces the same metrics. Training keeps `seed=None` for random negatives.

#### Custom Collate Function

Standard PyTorch collation expects uniform tensor sizes, but each example has a different number of labels (positives + variable negatives). The custom `collate_fn` keeps texts and labels as lists of strings, deferring padding to the model's forward method where the tokenizer handles it naturally.

#### Data Loading

Two paths, unified interface:

- **`load_and_split(data_path)`**: Loads from local HuggingFace DatasetDict (Arrow format on disk)
- **`load_and_split_from_hub(repo_id)`**: Loads directly from HuggingFace Hub via `load_dataset()`

Both return the same `(train_data, test_data, all_labels)` tuple. Scripts prioritize Hub loading when `dataset_repo_id` is configured, falling back to local `data_dir`.

---

## 4. Training Setup

### Implementation

GPU training support and a training script that loads config, processes data, and trains the model.

### What was implemented

#### Device Handling

Auto-detection with priority: CUDA > Apple MPS > CPU:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

MPS support is important since development was done on Apple Silicon.

#### Training Loop (train.py)

Step-based training (not epoch-based) with configurable intervals:

- **`log_every`**: Print loss statistics with ETA
- **`eval_every`**: Run full evaluation on test set, track best metric
- **`save_every`**: Save checkpoint with pruning

**Optimizer**: AdamW with configurable learning rate. AdamW decouples weight decay from the gradient update, which is standard for fine-tuning transformers.

**Evaluation**: Computes Precision, Recall, and F1 on the test set with threshold=0.5. These are the standard multi-label classification metrics.

**Unified eval/checkpoint block**: A single `evaluate()` call runs whenever either `eval_every` or `save_every` triggers on a given step. This avoids the previous design where eval and checkpoint had separate `evaluate()` calls — which produced different metrics for the same model state due to non-deterministic negative sampling. Now there is one evaluation, one `[eval]` log line, and one source of truth for best model tracking.

**Clean log output**: Each step produces at most one `[eval]` line with metrics. When a new best is found, the line shows `*best (saved)`. Checkpoint saves happen silently in the background. Example:

```
  Step   200/1000 | Loss: 0.5762 | 110s elapsed | ETA: 441s
  [eval] P: 0.8373 | R: 0.9804 | F1: 0.9032
  Step   250/1000 | Loss: 0.4660 | 136s elapsed | ETA: 409s
  [eval] P: 0.9497 | R: 0.9524 | F1: 0.9510 *best (saved)
```

#### Best Model Tracking

The training loop tracks the best model by a configurable metric (`best_metric`, default: `f1`) evaluated on the test set:

1. On every eval step, compare the current metric against `best_value`
2. If it's a new best, save model weights + tokenizer to `output_dir/best_model/`
3. At the end of training, write `training_meta.json` (with `best_step` field) to `best_model/`
4. The final model (last step) is also saved to `output_dir/` with its own metadata

This separation ensures the best weights are never lost — even if the model overfits or degrades after the best step, `best_model/` always contains the peak performance weights. The Hub push uses the best model's files and metrics.

#### Checkpoint Management

A pruning strategy keeps disk usage bounded:

1. Save checkpoint every `save_every` steps
2. Keep at most `save_total_limit` most recent checkpoints (default: 2)
3. Oldest checkpoints are pruned automatically

The best model is saved separately in `best_model/` and is never subject to pruning. Checkpoints serve as periodic snapshots for resumability, while `best_model/` is the definitive output.

#### Reproducibility

Full seed control across all sources of randomness:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Plus a `torch.Generator` seeded for the DataLoader's shuffling. This ensures identical runs given the same config.

#### HuggingFace Hub Integration

The push-to-hub logic is **quality-gated**:

1. Derive repo ID as `{hf_username}/{model_type}`
2. If the repo already exists, download its `training_meta.json`
3. Compare the new model's `best_metric` with the Hub model's
4. **Only push if the new model is better**

This prevents accidental regression — you can retrain freely and only improvements get published. Each push includes: model weights, tokenizer, `training_meta.json`, and an auto-generated model card (README.md) with training details and metrics.

---

## 5. Data Generation

### Implementation

Use an LLM to generate diverse synthetic examples in `{"text": ..., "labels": [...]}` format.

### What was implemented

#### LLM Choice: Gemini 2.5 Flash

We use Gemini 2.5 Flash. It's fast, cost-effective, and produces high-quality diverse outputs. The `GOOGLE_API_KEY` is loaded from `.env`.

#### Generation Strategy (generate_data.py)

1. **Batched generation**: Requests `samples_per_request` examples per API call (default: 50) to minimize API overhead
2. **Diverse prompt**: Instructs Gemini to cover 16+ domains (science, tech, sports, politics, health, arts, education, business, environment, history, philosophy, law, entertainment, travel, food, psychology) with varied styles (news headlines, statements, questions, descriptions)
3. **Temperature 1.0**: Maximum creativity/diversity
4. **Batch ID in prompt**: Each request includes a unique batch ID to vary outputs
5. **No repetition constraint**: Prompt explicitly asks for no topic/label repetition across examples

#### Robustness

- **JSON repair**: LLMs sometimes truncate JSON output. The parser walks backward through malformed JSON to find the last complete object and closes the array
- **Validation**: Each example is checked for structure (dict with "text" string and "labels" list), label count bounds, and type correctness
- **Graceful degradation**: Failed batches are skipped, generation continues
- **Rate limiting**: 2-second sleep between API calls

#### Output Format

Saved as a timestamped HuggingFace `DatasetDict` with pre-computed `train`/`test` splits:

```
data_dir/
  20260301_103000/
    train/     # Arrow format
    test/      # Arrow format
    dataset_dict.json
```

#### Dataset Quality Analysis (analyze_data.py)

Automatically runs after generation to validate quality:

- **Label distribution & entropy**: Measures if labels are uniformly distributed (higher entropy = more balanced)
- **Vocabulary stats (TTR)**: Type-Token Ratio measures vocabulary diversity. Higher TTR = less repetitive text
- **Hapax legomena**: Words appearing exactly once — indicates vocabulary richness
- **Text length stats**: Distribution of sentence lengths (min, max, mean, percentiles)
- **Multi-label complexity**: Percentage of examples with multiple labels, average labels per example
- **Label co-occurrence**: Most common label pairs — reveals thematic clusters
- **Train/test overlap**: Verifies that test labels also appear in training (critical for evaluation validity)
- **Stratification balance**: Measures if label proportions are preserved across train/test splits

---

## 6. Benchmark Setup (benchmark.py)

Compares all model variants on a shared test set:

- **Metrics**: F1, Precision, Recall (from training metadata)
- **Inference latency**: ms/sample measured on full test set with warmup
- **Parameter count**: Total trainable parameters
- **Training time**: Seconds to train

Models are discovered from HuggingFace Hub (configured in `benchmark.models`) or local directory. Results are saved to `results/benchmark.json` and printed as a ranked table marking best/worst performers.

---

## 7. Literature Review

Available in `documentation/literature_review.md`. Covers zero-shot classification approaches from 2019-2025, including BiEncoder architectures, late interaction models, poly-encoders, CLIP-style contrastive learning, and LLM-based methods.

---

## 8. Synthetic Data Generation Script

`generate_data.py` with Gemini 2.5 Flash, batched generation, JSON repair, validation, automatic quality analysis, and optional Hub push. See section 5 above for details.

--- 

## 9. LLM-as-Judge Evaluation (llm_judge.py)

Uses Gemini 2.5 Flash to qualitatively evaluate model predictions:

- **Scoring**: Each prediction scored 0-10 on correctness
- **Reasoning**: Natural language explanation of the score
- **Error analysis**: Identifies missed labels and false positives
- **Aggregation**: Average correctness score, distribution (perfect/good/ok/poor)

The prompt provides the LLM with the input text, ground truth labels, candidate labels (including negatives), and the model's predicted scores with the 0.5 threshold rule. Temperature is set to 0.1 for deterministic evaluation. Results are saved to `results/llm_judge_results.json`.

---

## 10. Interactive Playground (playground.py)

Gradio web UI for manual exploration:

- Dropdown for local models + text input for HF Hub repo IDs
- Multi-text input (one per line) with comma-separated labels
- Adjustable prediction threshold (slider 0.0-1.0)
- Displays model metadata (type, encoder, params, metrics)
- Results table with per-label scores and "Predicted" column based on threshold
- Model caching for fast repeated predictions

---

## 11. Architectural Decisions Summary

### Why shared encoder for texts and labels?

Both texts and labels are natural language. A shared encoder maps them to the same semantic space, enabling meaningful similarity computation. Separate encoders would need to learn alignment between two different spaces — harder with limited data.

### Why BCE instead of cross-entropy?

Binary Cross Entropy treats each label independently (multi-label). Cross-entropy with softmax would force labels to compete (mutually exclusive). In zero-shot classification, a text about "Climate change policy" should score high on both "Environment" and "Politics" simultaneously.

### Why step-based training instead of epoch-based?

With synthetic data, the dataset size is configurable and can change between runs. Step-based training gives consistent training duration regardless of dataset size. It also allows finer control over evaluation and checkpointing frequency.

### Why quality-gated Hub push?

Prevents accidental regression. You can experiment with hyperparameters freely — only improvements get published. The comparison uses the configured `best_metric` (default: F1), so the quality gate is objective and automatic.

### Why dynamic negative sampling instead of fixed negatives?

Fixed negatives create a static dataset where the model can memorize patterns. Dynamic sampling means every DataLoader pass presents different negative combinations, acting as data augmentation. The random number of negatives (1 to `max_negatives`) prevents the model from learning ratio shortcuts.

### Why global label pool from train+test?

If negatives only came from training labels, the model would never see test-only labels during training. Using the global pool exposes the model to the full label vocabulary, improving its ability to discriminate unseen labels at evaluation time.

### Why deterministic test evaluation?

Without seeding, negative sampling in the test set is random — two `evaluate()` calls on the same model produce different F1 scores. This caused a concrete problem: the training loop had separate eval and checkpoint blocks, each calling `evaluate()` independently. At the same step, they'd report different metrics (e.g., F1=0.9227 vs F1=0.9292), making best model tracking unreliable.

The fix uses `random.Random(seed + idx)` per test sample, making negative selection and shuffle order deterministic. This ensures evaluation is a pure function of model weights — same weights always produce the same metrics. Training keeps random negatives for diversity.

### Why a unified eval/checkpoint block?

The original design had two separate blocks: one for periodic evaluation (`eval_every`) and one for checkpoint saving (`save_every`). Both independently called `evaluate()`, which caused:

1. **Redundant computation**: When both triggered on the same step, the model was evaluated twice
2. **Inconsistent tracking**: The checkpoint block silently updated `best_eval_f1` without a `*best` marker, so the eval block's best tracking was unreliable
3. **Confusing logs**: Two lines of metrics per step (`[EVAL]` and `[CKPT]`) with different values

The unified block runs a single `evaluate()` when either interval triggers, uses the result for both best tracking and logging, and saves a checkpoint if needed — all from one code path.

### Why save best model separately from checkpoints?

Checkpoint pruning keeps only the N most recent checkpoints to bound disk usage. In the previous design, the best checkpoint was protected from pruning, but this complicated the pruning logic. By saving the best model to a dedicated `best_model/` directory (outside the checkpoint rotation), the best weights are always preserved regardless of pruning, and the pruning logic simplifies to "delete oldest".
