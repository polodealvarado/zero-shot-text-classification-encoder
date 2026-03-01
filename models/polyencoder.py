"""
PolyEncoder Model for Zero-Shot Text Classification.

Inspired by Poly-encoders (Humeau et al., 2020):
    - Middle ground between BiEncoder (fast but low interaction) and
      CrossEncoder (slow but high interaction).
    - Uses m learnable "poly-codes" that attend over text token embeddings
      to extract m context vectors (multiple "views" of the text).
    - Each candidate label then attends over these m vectors to produce
      a label-conditioned text representation.
    - This allows the text representation to adapt to each label candidate,
      unlike BiEncoder which uses a fixed text embedding.

Why PolyEncoder?
    - More expressive than BiEncoder: text representation adapts to each label.
    - More efficient than CrossEncoder: text tokens are encoded once,
      then only lightweight attention is done per label.
    - The m poly-codes capture m different aspects of the text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class PolyEncoderModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    Poly-encoder for zero-shot classification.
    Uses learnable poly-codes for multi-view text representation
    conditioned on each candidate label.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        num_poly_codes: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.num_poly_codes = num_poly_codes

        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_dim = self.shared_encoder.config.hidden_size

        # m learnable poly-codes: [m, D]
        self.poly_codes = nn.Parameter(torch.randn(num_poly_codes, hidden_dim) * 0.02)

    def encode_tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode strings into token-level embeddings.

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

    def poly_attention(
        self,
        token_embs: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        First attention layer: poly-codes attend over text tokens.

        Args:
            token_embs: [B, seq_len, D]
            token_mask: [B, seq_len]

        Returns:
            context_vectors: [B, m, D] — m context vectors per text
        """
        B, seq_len, D = token_embs.shape
        m = self.num_poly_codes

        # poly_codes: [m, D] → expand to [B, m, D]
        codes = self.poly_codes.unsqueeze(0).expand(B, -1, -1)

        # Attention scores: [B, m, D] × [B, D, seq_len] → [B, m, seq_len]
        attn_scores = torch.bmm(codes, token_embs.transpose(1, 2))

        # Mask padding tokens (set to -inf)
        mask_expanded = token_mask.unsqueeze(1).bool()  # [B, 1, seq_len]
        attn_scores = attn_scores.masked_fill(~mask_expanded, -1e9)

        # Softmax over seq_len dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, m, seq_len]

        # Weighted sum: [B, m, seq_len] × [B, seq_len, D] → [B, m, D]
        context_vectors = torch.bmm(attn_weights, token_embs)

        return context_vectors

    def label_conditioned_repr(
        self,
        context_vectors: torch.Tensor,
        label_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Second attention layer: label attends over m context vectors.

        Args:
            context_vectors: [B, m, D] — poly-code context vectors
            label_embedding:  [B, D] — label embedding (mean-pooled)

        Returns:
            text_repr: [B, D] — label-conditioned text representation
        """
        # Attention scores: [B, D] × [B, D, m] → [B, m]
        attn_scores = torch.bmm(
            label_embedding.unsqueeze(1),  # [B, 1, D]
            context_vectors.transpose(1, 2),  # [B, D, m]
        ).squeeze(1)  # [B, m]

        # Softmax over m codes
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, m]

        # Weighted sum: [B, 1, m] × [B, m, D] → [B, D]
        text_repr = torch.bmm(
            attn_weights.unsqueeze(1), context_vectors
        ).squeeze(1)

        return text_repr

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: poly-code attention + label-conditioned scoring.

        Returns:
            scores: [B, max_num_labels] — sigmoid scores
            mask:   [B, max_num_labels] — True where real labels exist
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Encode text tokens (no pooling)
        text_token_embs, text_token_mask = self.encode_tokens(texts)

        # First attention: poly-codes → m context vectors per text
        context_vectors = self.poly_attention(text_token_embs, text_token_mask)

        # Encode all labels with mean pooling
        all_labels = [label for labels in batch_labels for label in labels]
        all_label_embs = self.encode_mean_pool(all_labels)

        # Compute scores per label
        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels

        scores = torch.zeros(B, max_num_label, device=device)
        mask = torch.zeros(B, max_num_label, dtype=torch.bool, device=device)

        current = 0
        for i, count in enumerate(label_counts):
            actual = min(count, max_num_label)
            for j in range(actual):
                label_idx = current + j
                label_emb = all_label_embs[label_idx].unsqueeze(0)  # [1, D]

                # Get context vectors for this text
                ctx = context_vectors[i].unsqueeze(0)  # [1, m, D]

                # Second attention: label-conditioned text repr
                text_repr = self.label_conditioned_repr(ctx, label_emb)  # [1, D]

                # Score: dot product
                score = (text_repr * label_emb).sum(dim=-1)  # [1]
                scores[i, j] = score.squeeze()
                mask[i, j] = True
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
