"""
SpanClass Model for Zero-Shot Text Classification.

Inspired by GLiNER (Zaratiana et al., 2023) and the PhD thesis on span-based NER:
    - Instead of using a single pooled text representation, extracts candidate
      spans from the text and scores each span against each label.
    - Spans are represented by concatenating the start and end token embeddings,
      then projecting through an FFN.
    - A learned scorer selects the top-K most relevant spans.
    - Each label's final score is an attention-weighted aggregation over
      span-label similarities.

Why SpanClass?
    - Captures sub-sentence evidence: a label might match a specific phrase
      rather than the entire text.
    - Top-K selection acts as an information bottleneck, focusing on the
      most relevant text regions.
    - Attention-weighted aggregation lets multiple spans contribute to a
      label's score with learned importance weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class SpanClassModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    Span-attentive model for zero-shot classification.
    Extracts top-K spans from text and scores them against labels
    via attention-weighted aggregation.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        max_span_width: int = 5,
        top_k_spans: int = 8,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.max_span_width = max_span_width
        self.top_k_spans = top_k_spans

        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_dim = self.shared_encoder.config.hidden_size

        # Span representation: FFN(h_start || h_end) -> D
        self.span_ffn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Span relevance scorer
        self.span_scorer = nn.Linear(hidden_dim, 1)

    def encode_tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode strings into token-level embeddings.

        Returns:
            token_embs: [N, seq_len, D]
            token_mask: [N, seq_len]
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = self.shared_encoder(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]

    def encode_mean_pool(self, texts: list[str]) -> torch.Tensor:
        """Encode strings with mean pooling. Returns [N, D]."""
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = self.shared_encoder(**inputs)
        att_mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1)
        return pooled

    def _enumerate_spans(
        self,
        token_embs: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enumerate all valid spans up to max_span_width for a single text.

        Args:
            token_embs: [seq_len, D]
            token_mask: [seq_len]

        Returns:
            span_reprs:  [num_spans, D] — FFN(h_start || h_end)
            span_scores: [num_spans] — relevance scores
        """
        seq_len = int(token_mask.sum().item())
        max_w = min(self.max_span_width, seq_len)

        start_vecs = []
        end_vecs = []

        for width in range(1, max_w + 1):
            for start in range(seq_len - width + 1):
                end = start + width - 1
                start_vecs.append(token_embs[start])
                end_vecs.append(token_embs[end])

        if not start_vecs:
            # Edge case: no valid spans (shouldn't happen with seq_len >= 1)
            D = token_embs.size(-1)
            device = token_embs.device
            return torch.zeros(1, D, device=device), torch.zeros(1, device=device)

        starts = torch.stack(start_vecs)  # [num_spans, D]
        ends = torch.stack(end_vecs)      # [num_spans, D]

        # Span representation: FFN(h_start || h_end)
        concat = torch.cat([starts, ends], dim=-1)  # [num_spans, 2D]
        span_reprs = self.span_ffn(concat)           # [num_spans, D]

        # Relevance scores
        span_scores = self.span_scorer(span_reprs).squeeze(-1)  # [num_spans]

        return span_reprs, span_scores

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: extract spans, select top-K, score against labels.

        Returns:
            scores: [B, max_num_labels] — sigmoid probabilities per label
            mask:   [B, max_num_labels] — True where a real label exists
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Encode text tokens
        text_token_embs, text_token_mask = self.encode_tokens(texts)

        # Encode all labels with mean pooling
        all_labels = [label for labels in batch_labels for label in labels]
        all_label_embs = self.encode_mean_pool(all_labels)

        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels

        scores = torch.zeros(B, max_num_label, device=device)
        mask = torch.zeros(B, max_num_label, dtype=torch.bool, device=device)

        current = 0
        for i, count in enumerate(label_counts):
            actual = min(count, max_num_label)
            if actual == 0:
                current += count
                continue

            # Enumerate spans for this text
            span_reprs, span_relevance = self._enumerate_spans(
                text_token_embs[i], text_token_mask[i]
            )

            # Top-K selection (clamp if fewer spans than top_k)
            k = min(self.top_k_spans, span_reprs.size(0))
            top_indices = span_relevance.topk(k).indices
            top_spans = span_reprs[top_indices]       # [K, D]
            top_relevance = span_relevance[top_indices]  # [K]

            # Label embeddings for this text
            label_embs = all_label_embs[current : current + actual]  # [actual, D]

            # Span-label similarity: [actual, K]
            sim = label_embs @ top_spans.T

            # Attention weights from span relevance scores: [K]
            attn_weights = F.softmax(top_relevance, dim=0)  # [K]

            # Weighted aggregation: [actual, K] @ [K] -> [actual]
            label_scores = sim @ attn_weights

            scores[i, :actual] = label_scores
            mask[i, :actual] = True
            current += count

        scores = torch.sigmoid(scores)
        return scores, mask

    def compute_loss(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked Binary Cross Entropy loss."""
        loss = F.binary_cross_entropy(scores, targets, reduction="none")
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum()

    @torch.no_grad()
    def predict(
        self,
        texts: list[str],
        labels: list[list[str]],
    ) -> list[dict]:
        """Inference: return label scores for each text."""
        scores, mask = self.forward(texts, labels)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(labels[i]):
                if j < self.max_num_labels and mask[i, j]:
                    text_result[label] = round(scores[i, j].item(), 4)
            results.append({"text": text, "scores": text_result})
        return results
