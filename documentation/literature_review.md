# Literature Review: Zero-Shot Text Classification

## 1. Comparison of Approaches

### 1.1 BiEncoder (Dual Encoder)

**How it works**: Encodes text and labels independently into a shared embedding space using a single (or dual) transformer encoder. Similarity is computed via dot product or cosine similarity.

**Pros:**
- Fast inference: label embeddings can be pre-computed and cached
- Scales linearly with number of labels O(N)
- Simple architecture, easy to implement and debug
- Label embeddings are reusable across queries

**Cons:**
- No cross-attention between text and label tokens
- Limited expressiveness — text and label representations are fixed
- Relies on the encoder's ability to produce good general-purpose embeddings

**Best for**: Production systems with large label sets, real-time classification

### 1.2 ProjectionBiEncoder (CLIP-Style)

**How it works**: Extends the BiEncoder with learned projection heads that map text and label embeddings into a shared similarity space. Uses L2 normalization + learnable temperature scaling, inspired by CLIP's contrastive training.

**Pros:**
- Same speed as BiEncoder — projections add negligible overhead (<0.1% parameters)
- Learned similarity space can be better calibrated than raw dot product
- Temperature scaling improves confidence separation
- Optional InfoNCE contrastive loss improves embedding quality

**Cons:**
- Marginal improvement over BiEncoder in some settings
- Additional hyperparameters (projection_dim, temperature init)
- Still no cross-attention between text and label tokens

**Best for**: Drop-in improvement over BiEncoder when training data is available

### 1.3 LateInteraction (ColBERT-Style)

**How it works**: Encodes text and labels independently (like BiEncoder) but retains token-level embeddings instead of pooling. Scoring uses MaxSim: for each text token, find the max similarity with any label token, then sum.

**Pros:**
- More expressive than BiEncoder — captures token-level alignment
- Text and label encodings are still independent (can pre-compute labels)
- Effective "soft term matching" via MaxSim
- Good balance of speed and accuracy

**Cons:**
- Higher memory usage (stores all token embeddings, not just one vector per label)
- More complex scoring function
- Storage requirements scale with sequence length

**Best for**: When token-level matching matters, passage ranking scenarios

### 1.4 PolyEncoder

**How it works**: Encodes text with m learnable "poly-codes" that attend over token embeddings to produce m context vectors. Each candidate label then attends over these m vectors to get a label-conditioned text representation. Score is the dot product.

**Pros:**
- More expressive than BiEncoder — text representation adapts to each label
- More efficient than CrossEncoder — text tokens encoded once, then lightweight attention per label
- Configurable trade-off (more poly-codes = more expressive but slower)

**Cons:**
- More complex architecture and training
- Cannot fully pre-compute text representations (label-dependent attention)
- Performance sensitive to number of poly-codes (m)
- Marginal gains over BiEncoder in some tasks

**Best for**: Response selection in dialogue systems, moderate label sets

### 1.5 DynQuery (DyREx-Inspired Cross-Attention)

**How it works**: Encodes text and labels independently, then refines label representations via cross-attention where labels (queries) attend over text tokens (keys/values). The refined label embeddings capture text-specific evidence. Final scoring is dot product between refined labels and pooled text.

**Pros:**
- Label representations adapt dynamically to each input text
- Modular: text and labels are encoded independently before interaction
- Cross-attention lets labels "query" the text for relevant evidence
- More expressive than BiEncoder while keeping text encoding reusable

**Cons:**
- Requires a cross-attention pass per sample (cannot fully pre-compute)
- More parameters than BiEncoder (multi-head attention layer)
- Iterative per-sample scoring in the current implementation

**Best for**: Tasks where label-text interaction is important but full cross-encoding is too expensive

### 1.6 SpanClass (GLiNER-Inspired Span Scoring)

**How it works**: Instead of using a single pooled text representation, enumerates candidate spans from the text (up to a max width), represents each span via FFN(h_start || h_end), selects the top-K most relevant spans, and scores labels via attention-weighted aggregation over span-label similarities.

**Pros:**
- Captures sub-sentence evidence — a label might match a specific phrase, not the whole text
- Top-K selection acts as an information bottleneck, focusing on relevant regions
- Attention-weighted aggregation lets multiple spans contribute with learned importance
- Theoretically stronger for long texts with localized evidence

**Cons:**
- Span enumeration is O(seq_len × max_span_width) — can be expensive for long texts
- More complex architecture with multiple learned components (span FFN, scorer)
- Performance depends heavily on max_span_width and top_k_spans hyperparameters
- Higher memory usage during span enumeration

**Best for**: Classification tasks where evidence is localized in specific phrases or sub-sentences

### 1.7 ConvMatch (Multi-Scale CNN)

**How it works**: Replaces the transformer encoder entirely with multi-scale 1D convolutions (kernel sizes 2, 3, 4, 5) over pretrained word embeddings from BERT. Global max pooling extracts the strongest activation per filter, and a projection layer maps to a shared similarity space with L2 normalization.

**Pros:**
- Orders of magnitude faster than transformer-based models at inference
- CNNs capture local n-gram patterns sufficient for topic/category classification
- Multi-scale design captures both bigram collocations and longer phrasal patterns
- Pretrained BERT embeddings provide strong initialization without full transformer cost

**Cons:**
- No contextual (bidirectional) encoding — tokens are context-independent before convolution
- Cannot capture long-range dependencies beyond the largest kernel size
- Less expressive for nuanced semantic similarity
- Likely lower accuracy on tasks requiring deep understanding

**Best for**: High-throughput production systems where speed is critical, topic classification with surface-level patterns

---

## 2. Speed/Accuracy Trade-offs

| Architecture | Encoding | Scoring | Inference Speed | Accuracy | Memory |
|---|---|---|---|---|---|
| **BiEncoder** | Independent (mean pool) | Dot product | Very fast | Good | Low |
| **ProjectionBiEncoder** | Independent + projection | Cosine × temperature | Very fast | Good+ | Low |
| **LateInteraction** | Independent (token-level) | MaxSim | Medium | Good+ | High |
| **PolyEncoder** | Token-level + poly-codes | Dot product (conditioned) | Fast | Good+ | Medium |
| **DynQuery** | Independent + cross-attention | Dot product (refined) | Medium | Good+ | Medium |
| **SpanClass** | Independent (spans) | Attention-weighted sim | Medium-slow | Good+ | High |
| **ConvMatch** | CNN (no transformer) | Dot product (projected) | Fastest | Moderate | Lowest |

### Key observations:

1. **BiEncoder is the Pareto-optimal choice** for most production scenarios: it offers the best speed-to-accuracy ratio with the simplest architecture.

2. **ProjectionBiEncoder** is the cheapest upgrade over BiEncoder — projection heads add <0.1% parameters with measurable gains from learned similarity space and temperature scaling.

3. **LateInteraction (ColBERT)** provides meaningful accuracy gains by preserving token-level information, at the cost of ~10× storage and 2-3× inference time.

4. **PolyEncoder** bridges BiEncoder and CrossEncoder but is optimized for the dialogue/retrieval setting. For zero-shot classification with few labels, the overhead may not justify the gains.

5. **DynQuery** introduces the lightest form of text-label interaction — cross-attention adds expressiveness without full cross-encoding.

6. **SpanClass** is the most fine-grained approach — span-level scoring is especially valuable for long texts where evidence is localized.

7. **ConvMatch** sacrifices accuracy for extreme speed — useful as a fast baseline or in latency-critical deployments where n-gram patterns suffice.

---

## 3. Literature Review (2019–2025)

### 3.1 Foundational Works

#### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
**Reimers & Gurevych, 2019** | EMNLP 2019

The foundational paper for BiEncoder approaches to sentence similarity. Key contributions:
- Showed that BERT's native [CLS] embedding or mean pooling is suboptimal for similarity tasks
- Proposed siamese/triplet network fine-tuning of BERT for sentence embeddings
- Mean pooling consistently outperformed [CLS] token embeddings
- Made semantic similarity search practical with BERT (1000× faster than CrossEncoder)

**Relevance to our work**: Our BiEncoder uses the same mean pooling strategy. Sentence-BERT demonstrated that fine-tuned BERT can produce high-quality sentence embeddings for similarity comparison.

#### Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach
**Yin et al., 2019** | EMNLP 2019

Framed zero-shot classification as Natural Language Inference (NLI):
- Convert classification to hypothesis-premise pairs: "This text is about {label}"
- Fine-tune on NLI datasets (MNLI, SNLI) and transfer to classification
- Established benchmarks: Yahoo Answers, AG News, Yelp, DBpedia

**Relevance**: The NLI approach is the basis for models like `facebook/bart-large-mnli` on HuggingFace. Our BiEncoder approach is an alternative that doesn't require NLI framing but learns a similar semantic space.

### 3.2 Retrieval Architectures

#### ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
**Khattab & Zaharia, 2020** | SIGIR 2020

Introduced late interaction for passage retrieval:
- Keep token-level representations (no pooling)
- MaxSim scoring: soft term matching between query and passage tokens
- 170× faster than BERT re-rankers with comparable quality
- Token-level pre-computation enables efficient retrieval

**Relevance**: Our LateInteractionModel directly implements ColBERT's MaxSim approach, adapted from passage retrieval to zero-shot classification.

#### Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring
**Humeau et al., 2020** | ICLR 2020

Proposed the poly-encoder architecture:
- m learnable "poly-codes" extract multiple views of the input
- Label-conditioned attention produces adaptive text representations
- Bridges the gap between BiEncoder speed and CrossEncoder accuracy
- Evaluated on dialogue response selection tasks (Ubuntu, Reddit)

**Relevance**: Our PolyEncoderModel implements this architecture for zero-shot classification. The original paper focused on dialogue; we adapt it to label scoring.

### 3.3 Span-Based and Entity-Centric Models

#### GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer
**Zaratiana et al., 2023** | arXiv

Introduced a span-based approach for generalist NER:
- Enumerates candidate spans and scores them against entity type embeddings
- Span representation via start/end token concatenation + FFN projection
- Demonstrated strong zero-shot transfer to unseen entity types
- Key insight: localized span evidence is more discriminative than pooled sentence representations

**Relevance**: Our SpanClassModel directly adapts GLiNER's span enumeration and scoring approach from NER to zero-shot text classification. We add top-K span selection and attention-weighted aggregation to aggregate evidence from multiple spans per label.

#### DyREx: Dynamic Relation Extraction
**Zaratiana et al., 2024** | arXiv

Proposed dynamic query refinement for relation extraction:
- Uses cross-attention to refine relation type representations conditioned on the input context
- Relation queries attend over token-level text representations to capture relevant evidence
- Modular design: text and relation types encoded independently, then interact via lightweight attention

**Relevance**: Our DynQueryModel adapts DyREx's cross-attention mechanism from relation extraction to zero-shot classification. Labels (as queries) attend over text tokens (keys/values) to produce text-aware label representations, enabling dynamic adaptation without full cross-encoding.

### 3.4 CNN-Based Text Classification

#### Convolutional Neural Networks for Sentence Classification
**Kim, 2014** | EMNLP 2014

The foundational paper for CNN-based text classification:
- Applied multi-channel CNN filters over pre-trained word vectors (word2vec)
- Parallel filters with different kernel sizes capture n-gram patterns at multiple scales
- Global max-over-time pooling extracts the most salient feature per filter
- Achieved competitive results with much simpler architecture than RNNs

**Relevance**: Our ConvMatchModel builds on Kim's multi-scale CNN architecture. Key adaptations: (1) uses pretrained BERT embeddings instead of word2vec, (2) adds a projection layer for a learned similarity space with L2 normalization, (3) applies the CNN to both text and labels for zero-shot matching rather than fixed-class classification.

### 3.5 Contrastive Learning

#### Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Radford et al., 2021** | ICML 2021

Introduced contrastive pre-training for vision-language alignment:
- Dual encoder (image + text) with projection heads
- L2 normalization + learnable temperature parameter
- InfoNCE contrastive loss with in-batch negatives
- Demonstrated remarkable zero-shot transfer to downstream tasks

**Relevance**: Our ProjectionBiEncoderModel borrows three key ideas from CLIP: (1) projection heads to create a learned similarity space, (2) L2 normalization for cosine similarity, (3) learnable temperature scaling. These techniques are architecture-agnostic and improve any BiEncoder.

#### SimCSE: Simple Contrastive Learning of Sentence Embeddings
**Gao et al., 2021** | EMNLP 2021

Simplified contrastive learning for sentence embeddings:
- Unsupervised: use dropout as minimal data augmentation
- Supervised: leverage NLI labels for positive/negative pairs
- Showed that contrastive objectives dramatically improve embedding quality
- Became the standard approach for sentence embedding pre-training

**Relevance**: The contrastive component in our ProjectionBiEncoder loss is inspired by this line of work. InfoNCE + in-batch negatives is the standard contrastive recipe.

### 3.6 Modern Developments (2022–2025)

#### SetFit: Efficient Few-Shot Learning Without Prompts
**Tunstall et al., 2022** | arXiv

Showed that contrastive fine-tuning of sentence transformers can achieve near-GPT-3 performance with just 8 examples per class. Relevant because it demonstrates the power of the BiEncoder approach even in low-data regimes.

#### GTE, E5, BGE Embedding Models (2023–2024)
New generation of text embedding models trained with contrastive learning at scale:
- GTE (Alibaba), E5 (Microsoft), BGE (BAAI)
- Trained on billions of text pairs
- State-of-the-art on retrieval and similarity benchmarks (MTEB)
- Can be used as drop-in replacements for BERT in our BiEncoder architecture

**Relevance**: These models could serve as better base encoders for any of our transformer-based variants (BiEncoder, ProjectionBiEncoder, LateInteraction, PolyEncoder, DynQuery, SpanClass), potentially yielding significant improvements without changing the model structure.

---

## 4. Bibliography

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*.

2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

3. Yin, W., Hay, J., & Roth, D. (2019). Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. *EMNLP 2019*.

4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

5. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR 2020*.

6. Humeau, S., Shuster, K., Lachaux, M.-A., & Weston, J. (2020). Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. *ICLR 2020*.

7. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

8. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP 2021*.

9. Tunstall, L., Reimers, N., Jo, U. E. S., et al. (2022). Efficient Few-Shot Learning Without Prompts. *arXiv:2209.11055*.

10. Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2023). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. *arXiv:2311.08526*.

11. Zaratiana, U., Tomeh, N., Holat, P., & Charnois, T. (2024). DyREx: Dynamic Query Representation for Extractive Question Answering. *arXiv*.
