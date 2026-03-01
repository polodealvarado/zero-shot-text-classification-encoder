"""
DynQuery Model for Zero-Shot Text Classification.

Inspired by DyREx (Zaratiana et al., 2024):
    - Uses cross-attention to dynamically refine label representations
      conditioned on the input text's token-level context.
    - Labels attend over text tokens via nn.MultiheadAttention, producing
      text-aware label embeddings that capture relevant evidence.
    - Scoring is dot product between refined label embeddings and the
      pooled text representation, followed by sigmoid.

Why DynQuery?
    - Unlike BiEncoder, label representations are not static — they adapt
      to each input text via cross-attention.
    - Unlike CrossEncoder, text and labels are encoded independently first,
      keeping the architecture modular and efficient.
    - The cross-attention mechanism allows labels to "query" the text for
      relevant evidence, similar to how DyREx dynamically extracts relations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class DynQueryModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    Dynamic Query model for zero-shot classification.
    Labels attend over text tokens via cross-attention to produce
    text-conditioned label representations for scoring.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.num_attention_heads = num_attention_heads

        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_dim = self.shared_encoder.config.hidden_size

        # Cross-attention: labels (Q) attend over text tokens (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

    def encode_tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode strings into token-level embeddings (no pooling).

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

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode text tokens, encode labels, cross-attend, score.

        Returns:
            scores: [B, max_num_labels] — sigmoid probabilities per label
            mask:   [B, max_num_labels] — True where a real label exists
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Encode text at token level
        text_token_embs, text_token_mask = self.encode_tokens(texts)

        # Encode all labels with mean pooling
        all_labels = [label for labels in batch_labels for label in labels]
        all_label_embs = self.encode_mean_pool(all_labels)

        # Also get pooled text embeddings for final scoring
        att_mask = text_token_mask.unsqueeze(-1)
        text_pooled = (text_token_embs * att_mask).sum(1) / att_mask.sum(1)  # [B, D]

        # Compute scores per text
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

            # Gather label embeddings for this text: [actual, D]
            label_embs = all_label_embs[current : current + actual]

            # Cross-attention: labels query the text tokens
            # Q: [1, actual, D], K/V: [1, seq_len, D]
            text_kv = text_token_embs[i].unsqueeze(0)  # [1, seq_len, D]
            label_q = label_embs.unsqueeze(0)  # [1, actual, D]

            # Key padding mask: True means IGNORE (PyTorch convention)
            key_padding = ~text_token_mask[i].bool().unsqueeze(0)  # [1, seq_len]

            refined_labels, _ = self.cross_attention(
                query=label_q,
                key=text_kv,
                value=text_kv,
                key_padding_mask=key_padding,
            )  # [1, actual, D]

            refined_labels = refined_labels.squeeze(0)  # [actual, D]

            # Score: dot product between refined labels and pooled text
            text_vec = text_pooled[i]  # [D]
            label_scores = refined_labels @ text_vec  # [actual]

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
