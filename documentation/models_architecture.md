# Model Architecture — Zero-Shot Text Classification

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Model-by-Model Deep Dive](#2-model-by-model-deep-dive)
   - [BiEncoder](#21-biencoder-basepy)
   - [ProjectionBiEncoder](#22-projectionbiencoder-projectionpy)
   - [LateInteraction](#23-lateinteraction-late_interactionpy)
   - [PolyEncoder](#24-polyencoder-polyencoderpy)
   - [DynQuery](#25-dynquery-dynquerypy)
   - [SpanClass](#26-spanclass-spanclasspy)
   - [ConvMatch](#27-convmatch-convmatchpy)
3. [Head-to-Head Comparison Table](#3-head-to-head-comparison-table)
4. [Interaction Spectrum — From No Interaction to Full Interaction](#4-interaction-spectrum)
5. [Design Decisions Shared Across All Models](#5-shared-design-decisions)
6. [When to Pick Which Model](#6-when-to-pick-which-model)
7. [Key Interview Talking Points](#7-key-interview-talking-points)
8. [References](#8-references)

---

## 1. Framework Overview

All 7 models live in a `models/` package and are registered in `MODEL_REGISTRY` for dynamic dispatch. Every model implements the same interface:

```
forward(texts, batch_labels) → (scores [B, max_labels], mask [B, max_labels])
compute_loss(scores, targets, mask) → scalar
predict(texts, labels) → list[dict]
```

This means `train.py`, `benchmark.py`, and `playground.py` work with **any** variant without model-specific code paths. All models inherit from `nn.Module` + `PyTorchModelHubMixin`, which gives free `save_pretrained()`, `from_pretrained()`, and `push_to_hub()`.

**Loss function**: Masked Binary Cross Entropy everywhere (multi-label, not softmax). Sigmoid treats each label independently — a text can belong to "Environment" and "Politics" simultaneously.

**Encoder**: All transformer-based models use a single shared encoder (e.g., `bert-base-uncased`) for both texts and labels. This forces both into the same semantic space, which is essential for zero-shot generalization to unseen labels.

---

## 2. Model-by-Model Deep Dive

### 2.1 BiEncoder (`base.py`)

**Paper**: Sentence-BERT (Reimers & Gurevych, EMNLP 2019)

#### Architecture

```
Text  ──→ Shared BERT ──→ Mean Pooling ──→ text_emb [D]
                                                        ──→ dot product ──→ sigmoid ──→ score
Label ──→ Shared BERT ──→ Mean Pooling ──→ label_emb [D]
```

#### How It Works

1. A single transformer encoder produces token-level embeddings for both text and label.
2. **Mask-aware mean pooling** collapses tokens to a single vector: `(last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)` — padding tokens are excluded.
3. **Dot product** between text and label vectors produces a raw similarity score.
4. **Sigmoid** converts to independent per-label probability.

#### Why This Design

- **Simplicity**: The simplest possible architecture that maps text and labels to the same space. Acts as a strong baseline against which all other models are measured.
- **Inference speed**: Label embeddings can be **pre-computed and cached**. At inference time, only the input text needs encoding. Scoring is a single matrix multiplication.
- **Linear scaling**: Comparing one text against N labels costs O(N) dot products, not O(N) forward passes.

#### Pros

- Fastest inference of all transformer-based variants
- Smallest memory footprint — stores one vector per label
- Label embeddings are cacheable and reusable across inputs
- Easy to debug and reason about
- Strong baseline — often surprisingly competitive

#### Cons

- **No interaction** between text and label representations after encoding
- The text has one fixed embedding regardless of which label it's being compared to
- Mean pooling loses fine-grained token-level information
- Raw BERT embeddings are not optimized for similarity comparison (no projection head)

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `model_name` | `bert-base-uncased` | Base transformer encoder |
| `max_num_labels` | `5` | Max candidate labels per sample |

---

### 2.2 ProjectionBiEncoder (`projection.py`)

**Paper**: CLIP (Radford et al., ICML 2021) + SimCSE (Gao et al., EMNLP 2021)

#### Architecture

```
Text  ──→ Shared BERT ──→ Mean Pool ──→ Linear(768→256) ──→ L2 Norm ──→ text_emb [256]
                                                                                        ──→ cosine_sim × temperature ──→ sigmoid
Label ──→ Shared BERT ──→ Mean Pool ──→ Linear(768→256) ──→ L2 Norm ──→ label_emb [256]
```

#### How It Works

Three additions on top of BiEncoder, all inspired by CLIP:

1. **Projection head**: `nn.Linear(768, 256)` maps from the raw BERT embedding space into a **learned similarity space**. Raw BERT embeddings were trained for masked language modeling, not similarity — the projection learns a task-specific metric.
2. **L2 normalization**: After projection, all embeddings are constrained to the unit hypersphere. Dot product on L2-normalized vectors **equals cosine similarity**, removing the influence of embedding magnitude.
3. **Learnable temperature**: A scalar parameter initialized to `log(1/0.07) ≈ 2.66` (CLIP's initialization). Scales cosine similarity before sigmoid: `sigmoid(cos_sim × exp(τ))`. This lets the model learn how "peaked" or "flat" its confidence distribution should be.

#### Dual Loss Function

```
total_loss = BCE_loss + α × InfoNCE_loss
```

- **BCE** handles the primary multi-label classification objective.
- **InfoNCE** (symmetric, like CLIP) provides a **batch-level contrastive signal**: pull matching text-label pairs together, push non-matching pairs apart. It uses in-batch negatives — other samples in the same batch become automatic negatives.
- `contrastive_alpha` (default 0.1) controls the balance. InfoNCE acts as a regularizer, not the primary loss.

#### Why This Design

- CLIP showed that projection + L2 norm + learnable temperature dramatically improves zero-shot transfer in vision-language models. The same principles are architecture-agnostic and apply to text-text matching.
- The projection head adds <0.2M parameters on top of 110M (BERT-base) — negligible cost, meaningful gain.
- InfoNCE provides structural signal that BCE alone misses: BCE only says "this label is correct/incorrect for this text", while InfoNCE says "this text-label pair is more similar than all other pairs in the batch".

#### Pros

- Minimal parameter overhead over BiEncoder (~200K extra params)
- Learned similarity space is more discriminative than raw BERT space
- Temperature scaling adapts confidence calibration automatically
- InfoNCE exploits in-batch negatives for free (no extra data needed)
- Same inference speed as BiEncoder — projection is negligible

#### Cons

- InfoNCE assumes the first positive label per sample is the anchor — simplified from the full multi-label setting
- Projection dimension is a hyperparameter that needs tuning
- Temperature can become unstable during training (very large or very small)

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `projection_dim` | `256` | Dimensionality of the similarity space |
| `contrastive_alpha` | `0.1` | Weight of InfoNCE vs BCE |
| `log_temperature` | `2.6593` | Initial temperature (CLIP default) |

---

### 2.3 LateInteraction (`late_interaction.py`)

**Paper**: ColBERT (Khattab & Zaharia, SIGIR 2020)

#### Architecture

```
Text  ──→ Shared BERT ──→ Linear(768→128) ──→ L2 Norm ──→ token_embs [seq_t, 128]
                                                                                     ──→ MaxSim ──→ sigmoid
Label ──→ Shared BERT ──→ Linear(768→128) ──→ L2 Norm ──→ token_embs [seq_l, 128]
```

#### How It Works

The key difference from BiEncoder: **no pooling**. Token-level embeddings are preserved.

1. Both text and label tokens are encoded independently by the shared transformer.
2. Each token is projected down from 768 to 128 dimensions and L2-normalized.
3. **MaxSim scoring**: For each text token, find the maximum cosine similarity with **any** label token. Then average across all text tokens.

```
MaxSim(text, label) = (1/|text_tokens|) × Σ_i max_j(text_token_i · label_token_j)
```

4. Sigmoid converts MaxSim scores to probabilities.

#### Why This Design

- **Soft term matching**: MaxSim finds the best token-to-token alignment for each text token. If a text says "The stock market crashed yesterday" and a label is "Finance", MaxSim discovers that "stock" and "market" individually have high similarity with "Finance", even though "yesterday" doesn't.
- **Token-level granularity preserves information** that mean pooling discards. The meaning of "bank" in "river bank" vs "investment bank" is better captured when neighboring tokens participate.
- **Still independent encoding**: Text and labels are encoded separately. Label token embeddings can be pre-computed (unlike CrossEncoder).

#### Pros

- More expressive than BiEncoder — captures fine-grained token alignment
- Labels and texts encoded independently — label tokens can be cached
- Effective at matching specific words/phrases to label semantics
- Naturally handles multi-word labels ("Machine Learning" → two informative tokens)

#### Cons

- **Slower**: Scoring is O(seq_text × seq_label) per pair vs O(1) for dot product
- **Higher memory**: Stores all token embeddings, not one vector per text/label
- **Loop-based scoring** in current implementation (Python loops over batch items) — could be optimized with batched operations
- Less benefit when labels are single words (only one label token to MaxSim against)

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `token_projection_dim` | `128` | Per-token embedding dimension (ColBERT uses 128) |

---

### 2.4 PolyEncoder (`polyencoder.py`)

**Paper**: Poly-encoders (Humeau et al., ICLR 2020)

#### Architecture

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │            FIRST ATTENTION (Poly-codes → Text)              │
Text ──→ BERT ──→  │  m poly-codes attend over text tokens → m context vectors   │
  token_embs       │  attn = softmax(poly_codes @ text_tokens.T)                 │
  [seq, D]         │  context = attn @ text_tokens → [m, D]                      │
                    └──────────────────────────┬──────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────────┐
                    │           SECOND ATTENTION (Label → Context Vectors)        │
Label ──→ BERT ──→ │  label_emb attends over m context vectors                   │
  mean pool → [D]  │  attn = softmax(label_emb @ context_vectors.T)              │
                    │  text_repr = attn @ context_vectors → [D]                   │
                    └──────────────────────────┬──────────────────────────────────┘
                                               │
                                    dot(text_repr, label_emb) → sigmoid → score
```

#### How It Works

Two-stage attention mechanism:

1. **Poly-codes** (`nn.Parameter` of shape `[m, D]`, default m=16): m learnable "query vectors" that represent different **aspects** or **views** of the text. They attend over text tokens to extract m context vectors.
2. **Label-conditioned attention**: Each candidate label embedding then attends over the m context vectors. This produces a **different text representation for each label**.
3. **Dot product + sigmoid**: The label-conditioned text representation is scored against the label embedding.

#### Why This Design

Consider the text: *"Apple released a new iPhone with improved battery life while markets reacted positively."*

- **BiEncoder**: Produces ONE fixed text embedding. Same vector compared to "Technology" and "Finance".
- **PolyEncoder**: When scoring "Technology", the second attention emphasizes context vectors that captured "iPhone" and "battery". When scoring "Finance", it emphasizes vectors about "markets" and "reacted positively".

The text representation **adapts to each candidate label**. This is the PolyEncoder's core advantage.

Compared to a full CrossEncoder (which requires a complete forward pass per label), the PolyEncoder encodes the text **once**, then does lightweight attention per label. This gives most of the expressiveness at a fraction of the cost.

#### Pros

- **Label-adaptive text representation** — more expressive than BiEncoder
- Text tokens are encoded once — only lightweight attention per label
- Number of poly-codes (m) is a tunable expressiveness/speed knob
- Captures multiple text aspects simultaneously (e.g., topic, sentiment, entity mentions)

#### Cons

- **Cannot cache text representations** — they depend on the label
- More complex architecture: two attention stages, harder to debug
- Performance is sensitive to the number of poly-codes (m). Too few → under-expressive. Too many → overfitting.
- **Loop-based scoring** in current implementation (iterates per label)
- The advantage is marginal when labels are very different from each other (easy classification)

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `num_poly_codes` | `16` | Number of text "views" (m). Higher = more expressive, slower |

---

### 2.5 DynQuery (`dynquery.py`)

**Paper**: DyREx (Zaratiana et al., NeurIPS 2022 Workshop)

#### Architecture

```
Text  ──→ Shared BERT ──→ token_embs [seq, D]  ──→  K, V for cross-attention
                          ↓ mean pool → text_pooled [D]
                                                                                ──→ dot product ──→ sigmoid
Label ──→ Shared BERT ──→ mean pool → label_emb [D] ──→ Q for cross-attention
                                                          ↓
                                              cross-attention(Q=label, K=text, V=text)
                                                          ↓
                                              refined_label_emb [D]
```

#### How It Works

1. Text is encoded at the **token level** (no pooling for cross-attention keys/values).
2. Labels are encoded with **mean pooling** to get initial label embeddings.
3. **Cross-attention**: Each label embedding acts as a **query** that attends over text tokens (keys and values). The output is a **text-conditioned label representation**.
4. The refined label embedding is scored against the pooled text representation via dot product + sigmoid.

The cross-attention uses `nn.MultiheadAttention` with 4 heads and proper key padding masking.

#### Why This Design

**DynQuery is the complement of PolyEncoder**:

| Aspect | PolyEncoder | DynQuery |
|:-------|:------------|:---------|
| What adapts? | Text representation adapts to label | **Label representation adapts to text** |
| Mechanism | Label attends over poly-code context vectors | Label attends over **text tokens** |
| Intuition | "Show the text from the label's perspective" | "Refine what the label means given this specific text" |

The key insight: the word "Bank" should mean something different depending on whether the text discusses rivers or finance. In BiEncoder, "Bank" always has the same embedding. In DynQuery, "Bank" attends over the text tokens and **absorbs context**: near "river" it shifts toward geography; near "loan" it shifts toward finance.

Inspired by DyREx (Zaratiana et al., 2022), which dynamically refines query representations for extractive QA. We adapt the same principle from QA queries to classification labels.

#### Pros

- **Handles label polysemy** naturally — "Bank" adapts to context
- Minimal overhead: one `nn.MultiheadAttention` layer (~2M params on 768-dim)
- Complementary to PolyEncoder — could theoretically be combined (bidirectional adaptation)
- Labels attend directly over text tokens — richer signal than PolyEncoder's compressed context vectors

#### Cons

- **Loses label cacheability**: Label embeddings must be recomputed per input text. In BiEncoder, label embeddings are computed once and reused.
- Cross-attention over text tokens has O(num_labels × seq_len) complexity
- Risk of overfitting: the cross-attention could learn to "copy" text info into labels, bypassing the similarity mechanism
- **Loop-based implementation** per text in the batch

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `num_attention_heads` | `4` | Number of heads in cross-attention |

---

### 2.6 SpanClass (`spanclass.py`)

**Paper**: GLiNER (Zaratiana et al., NAACL 2024)

#### Architecture

```
Text ──→ Shared BERT ──→ token_embs [seq, D]
                            │
                            ▼
              Enumerate spans (i, j) where j-i ≤ W
              span_repr = FFN(h_start ∥ h_end) → [D]
              span_score = Linear(span_repr) → scalar
                            │
                            ▼
              Top-K spans by relevance score → [K, D]
                            │
                            ▼
              Span-label similarity: [K, D] @ [num_labels, D].T → [K, num_labels]
              Attention-weighted aggregation → per-label score → sigmoid
```

#### How It Works

1. Text tokens are encoded at the token level.
2. **Span enumeration**: All contiguous spans `(start, end)` up to width W are generated. Each span is represented by `FFN(h_start ∥ h_end)` — concatenating start and end token embeddings and projecting through a 2-layer FFN.
3. **Span relevance scoring**: A learned linear scorer assigns a relevance score to each span.
4. **Top-K selection**: Only the K most relevant spans are kept. This acts as an **information bottleneck**, focusing computation on the most informative text regions.
5. **Span-label scoring**: Each top-K span is compared to each label via dot product. The final per-label score is an **attention-weighted sum** over span-label similarities, where attention weights come from span relevance scores.

#### Why This Design

All other models reduce the text to a single global representation (mean pooling or attention-weighted). SpanClass preserves **sub-sentence structure**.

Consider: *"Apple released a new iPhone with improved battery life while markets reacted positively."*

- For label "Technology", the relevant evidence is the span "released a new iPhone with improved battery life"
- For label "Finance", the relevant evidence is "markets reacted positively"

A mean-pooled vector blends both signals together. SpanClass identifies the most relevant spans and matches them to labels individually.

**Directly adapted from GLiNER**: The GLiNER architecture uses `FFN(h_i ∥ h_j)` span representations for NER. We use the same representation but for classification — instead of extracting entity spans, we extract **evidence spans** that support each label.

#### Pros

- **Captures sub-sentence evidence** — identifies WHERE in the text supports each label
- **Interpretability**: Top-K spans reveal WHY the model chose a label (useful for debugging and trust)
- Information bottleneck via top-K prevents noisy spans from diluting scores
- Directly connected to GLiNER's proven span-based approach
- Novel application: span-based methods are well-studied for NER but rarely applied to text classification

#### Cons

- **O(n^2 × W) span enumeration** — quadratic in sequence length, though bounded by max_span_width
- Multiple hyperparameters to tune (max_span_width, top_k_spans)
- Top-K selection is not fully differentiable (gradients don't flow through the selection indices)
- Heavier computation than BiEncoder/ProjectionBiEncoder
- **Loop-based** span enumeration and scoring per text

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `max_span_width` | `5` | Maximum span length (tokens). Larger → captures longer phrases, but O(n×W) spans |
| `top_k_spans` | `8` | Number of spans kept after selection. Larger → more evidence, more noise |

---

### 2.7 ConvMatch (`convmatch.py`)

**Papers**: TextCNN (Kim, EMNLP 2014) + Contrastive String Representation (Zaratiana et al., 2021)

#### Architecture

```
Text ──→ BERT word embeddings (frozen tokenizer, fine-tunable embeddings)
           ↓
         ┌──────────────────────────────────────┐
         │  Parallel Conv1D:                     │
         │    kernel=2 → ReLU → GlobalMaxPool    │
         │    kernel=3 → ReLU → GlobalMaxPool    │
         │    kernel=4 → ReLU → GlobalMaxPool    │
         │    kernel=5 → ReLU → GlobalMaxPool    │
         └──────────────────────────────────────┘
           ↓ concatenate [4 × num_filters]
         Linear(4×128, 256) ──→ L2 Norm ──→ text_emb [256]
                                                             ──→ dot product ──→ sigmoid
Label ──→ (same CNN pipeline) ──→ label_emb [256]
```

#### How It Works

1. **Pretrained word embeddings**: Extracts BERT's `word_embeddings` weight matrix (the first layer), then **discards the entire transformer**. These embeddings are fine-tunable.
2. **Multi-scale CNN**: Four parallel `Conv1D` filters with kernel sizes [2, 3, 4, 5] capture bigrams, trigrams, 4-grams, and 5-grams respectively.
3. **Global max pooling**: For each filter, take the maximum activation across the sequence. This extracts the strongest n-gram match.
4. **Concatenation + Projection**: The 4 × 128 = 512 concatenated features are projected to a 256-dim similarity space.
5. **L2 normalization**: Same as ProjectionBiEncoder — embeddings on the unit hypersphere, dot product = cosine similarity.
6. **Scoring**: Standard dot product + sigmoid.

#### Why This Design

**Not everything needs a transformer.** For topic classification, the discriminative signal is often local: "stock market" → Finance, "neural network" → AI. CNNs capture these n-gram patterns natively and are orders of magnitude faster than self-attention.

Clever engineering detail: instead of training embeddings from scratch (GloVe, fastText), ConvMatch **steals BERT's pretrained word embeddings** as initialization. This gives the quality of transformer pre-training without the cost of running the transformer at inference.

Connects to work on contrastive string representation learning using surface-level patterns — the philosophy that lightweight models with the right inductive bias can be surprisingly effective.

#### Pros

- **Orders of magnitude faster** than any transformer variant at inference — no self-attention
- **Tiny model**: ~2-5M parameters vs 110M for BERT-base
- Deployable on edge devices / CPU-only environments
- N-gram patterns are naturally suited for topic/category matching
- Leverages pretrained BERT embeddings without transformer overhead
- Provides genuine **architectural diversity** in the model zoo

#### Cons

- **No long-range dependencies**: CNNs only see local windows. "The market is NOT crashing" and "The market IS crashing" have similar n-grams but opposite meanings.
- No positional encoding — word order beyond the kernel window is lost
- **Short labels are problematic**: A single-word label like "Finance" gives the CNN very little signal. Multi-word labels or descriptions work better.
- Cannot benefit from advances in transformer pre-training (new encoders like DeBERTa, GTE, BGE)
- Lower accuracy ceiling than transformer-based models on nuanced tasks

#### Key Hyperparameters

| Parameter | Default | Purpose |
|:----------|:--------|:--------|
| `num_filters` | `128` | Filters per kernel size. Total CNN features = 4 × num_filters |
| `projection_dim` | `256` | Similarity space dimension |

---

## 3. Head-to-Head Comparison Table

| Model | Inspiration | Interaction Type | Scoring | Params (extra) | Inference Speed | Label Caching | Expressiveness |
|:------|:------------|:-----------------|:--------|:---------------|:----------------|:-------------|:---------------|
| **BiEncoder** | Sentence-BERT | None | Dot product | 0 | Fastest (transformer) | Yes | Low |
| **ProjectionBiEncoder** | CLIP | None (learned space) | Cosine × τ | ~200K | Fastest (transformer) | Yes | Low-Medium |
| **LateInteraction** | ColBERT | Token-level (late) | MaxSim | ~100K | Medium | Yes (per token) | Medium |
| **PolyEncoder** | Poly-encoders | Text adapts to label | Attention + dot | ~12K (poly-codes) | Medium-Fast | No (text) | Medium-High |
| **DynQuery** | DyREx | Label adapts to text | Cross-attn + dot | ~2.4M | Medium | No (label) | Medium-High |
| **SpanClass** | GLiNER | Sub-sentence spans | Span-label attn | ~1.8M | Slow | Partial | High |
| **ConvMatch** | TextCNN | N-gram patterns | Dot product | ~2-5M (total) | Fastest (overall) | Yes | Low |

---

## 4. Interaction Spectrum

The 7 models form a spectrum from **no interaction** to **deep interaction** between text and label:

```
No Interaction                                                    Deep Interaction
      │                                                                    │
      ▼                                                                    ▼
  ConvMatch    BiEncoder    Projection    LateInteraction    PolyEncoder    DynQuery    SpanClass
      │            │            │               │                │            │            │
  CNN n-grams   dot product  cosine sim    token MaxSim      text adapts   label adapts  span-label
  (no attn)     (1 vector)   (1 vector)    (all tokens)      (to label)    (to text)     matching
```

**Key insight**: Moving right on this spectrum generally increases accuracy but decreases inference speed. The challenge is finding the right point on this trade-off curve for your deployment constraints.

---

## 5. Shared Design Decisions

These apply to all 7 models and are important interview talking points:

### Why Sigmoid + BCE, Not Softmax + Cross-Entropy?

This is **multi-label classification**. A text about "Climate change policy" should score high on BOTH "Environment" and "Politics". Softmax forces probabilities to sum to 1, creating artificial competition between labels. Sigmoid treats each label independently.

### Why a Shared Encoder?

Both texts and labels are natural language. A shared encoder forces them into the same semantic space, enabling similarity comparison. Separate encoders would need to learn cross-space alignment — much harder with limited data.

### Why Masked BCE?

Each sample can have a different number of candidate labels (positives + variable negatives). The mask ensures loss is only computed over real label positions, not padding.

### Why Dynamic Negative Sampling?

- **Random count** (1 to max_negatives): prevents the model from learning a fixed positive/negative ratio shortcut.
- **Shuffled order**: prevents positional shortcuts ("first labels are always positive").
- **Re-sampled per epoch**: acts as data augmentation.
- **Global label pool** (train + test): exposes the model to all labels during training.

---

## 6. When to Pick Which Model

| Scenario | Best Model | Why |
|:---------|:-----------|:----|
| **Production API with large label set** | BiEncoder or ProjectionBiEncoder | Label caching, fast inference, linear scaling |
| **Maximum accuracy, small label set** | DynQuery or SpanClass | Deep interaction captures nuance |
| **Token-level matching matters** (e.g., technical terms) | LateInteraction | MaxSim finds specific token alignments |
| **Labels are ambiguous/polysemous** | DynQuery | Labels adapt to context ("Bank" → finance vs geography) |
| **Need to explain predictions** | SpanClass | Top-K spans show which text regions support each label |
| **Edge deployment / CPU-only** | ConvMatch | Tiny model, no transformer overhead |
| **Multi-aspect texts** (mixed topics) | PolyEncoder or SpanClass | Multiple "views" of the text, or sub-sentence evidence |
| **Quick baseline to iterate from** | BiEncoder | Simplest, fastest to train, surprisingly strong |

---

## 7. Key Interview Talking Points

### "Why did you build 7 models instead of 1?"

Each model represents a different point on the **interaction vs. efficiency trade-off spectrum**. The progression from BiEncoder to SpanClass systematically explores how much text-label interaction is needed for zero-shot classification. This is a deliberate architectural study, not feature bloat.

### "How do they relate to the literature?"

| Model | Lineage | Adaptation |
|:------|:--------|:-----------|
| BiEncoder | Sentence-BERT (2019) | Standard baseline for semantic similarity |
| ProjectionBiEncoder | CLIP (2021) | Vision-language contrastive → text-text matching |
| LateInteraction | ColBERT (2020) | Passage retrieval → label matching |
| PolyEncoder | Poly-encoders (2020) | Dialogue response selection → label scoring |
| DynQuery | DyREx (2022) | Extractive QA query refinement → label refinement |
| SpanClass | GLiNER (2024) | NER span scoring → evidence span classification |
| ConvMatch | TextCNN (2014) + Contrastive String Repr. (2021) | CNN classification + contrastive string repr. |

### "What's the biggest architectural insight?"

**Asymmetric adaptation matters.** PolyEncoder adapts the text to the label. DynQuery adapts the label to the text. These are complementary approaches to the same problem: "how do we make representations context-dependent without full cross-encoding?" A future direction is combining both (bidirectional adaptation).

### "What would you do differently?"

1. **Batch the loop-based scoring**: LateInteraction, PolyEncoder, DynQuery, and SpanClass all iterate over batch items. Batched tensor operations would be 5-10x faster.
2. **Better base encoders**: All models use BERT-base. Modern encoders (DeBERTa-v3, GTE, BGE) would likely improve all transformer-based variants without code changes.
3. **FilterClass (two-stage)**: For very large label spaces, a cheap BiEncoder filter + expensive scorer (SpanClass/DynQuery) would combine the best of both worlds — inspired by the Filtered Semi-Markov CRF approach.
4. **Label descriptions**: Short labels like "Finance" give weak signal. Expanding them to "Finance: money, markets, banking, investment" would help all models, especially ConvMatch.

### "How is zero-shot generalization possible?"

Because the encoder is pre-trained on massive text corpora (BERT's masked language modeling). It already has a notion that "stock market" is semantically close to "Finance". Fine-tuning with synthetic data teaches the model to **use that pre-existing knowledge for classification** — matching text regions to label descriptions via similarity. Labels unseen during training still work because the encoder already understands their semantics.

---

## 8. References

1. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
2. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
3. Khattab, O. & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR 2020.
4. Humeau, S. et al. (2020). *Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring.* ICLR 2020.
5. Zaratiana, U. et al. (2022). *DyREx: Dynamic Query Representation for Extractive Question Answering.* NeurIPS 2022 Workshop.
6. Zaratiana, U. et al. (2024). *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.* NAACL 2024.
7. Zaratiana, U. et al. (2023). *Filtered Semi-Markov CRF.* EMNLP 2023 Findings.
8. Zaratiana, U. et al. (2021). *Contrastive String Representation Learning using Synthetic Data.* arXiv:2110.04217.
9. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014.
10. Gao, T. et al. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP 2021.
11. Yin, W. et al. (2019). *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach.* EMNLP 2019.
