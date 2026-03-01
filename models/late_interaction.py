"""
Late Interaction Model (ColBERT-style) for Zero-Shot Text Classification.

Inspired by ColBERT (Khattab & Zaharia, 2020):
    - Instead of collapsing tokens into a single vector (mean pooling),
      we keep token-level embeddings.
    - Scoring uses MaxSim: for each text token, find the maximum cosine similarity
      with any label token, then sum across text tokens.
    - This allows fine-grained token-level interaction between text and label.

Why Late Interaction?
    - More expressive than BiEncoder: captures token-level alignment.
    - More efficient than CrossEncoder: text and label are encoded independently.
    - MaxSim is effective because it finds the best-matching token pair for each
      text token, then aggregates — like soft term matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class LateInteractionModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    ColBERT-style late interaction model for zero-shot classification.
    Uses token-level MaxSim scoring instead of single-vector dot product.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        token_projection_dim: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.token_projection_dim = token_projection_dim

        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_dim = self.shared_encoder.config.hidden_size

        # Project token embeddings to lower dimension (768 → 128)
        self.token_projection = nn.Linear(hidden_dim, token_projection_dim)

    def encode_tokens(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode strings into token-level embeddings (no pooling).

        Returns:
            token_embs: [N, seq_len, token_projection_dim] — L2-normalized
            token_mask: [N, seq_len] — attention mask (1 for real tokens, 0 for padding)
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = self.shared_encoder(**inputs)

        # Project to lower dimension
        token_embs = self.token_projection(outputs.last_hidden_state)

        # L2 normalize each token embedding
        token_embs = F.normalize(token_embs, p=2, dim=-1)

        return token_embs, inputs["attention_mask"]

    def maxsim_score(
        self,
        text_embs: torch.Tensor,
        text_mask: torch.Tensor,
        label_embs: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        MaxSim scoring between one text and one label.

        For each text token, find the max cosine similarity with any label token,
        then sum across all text tokens.

        Args:
            text_embs:  [seq_t, D]
            text_mask:  [seq_t]
            label_embs: [seq_l, D]
            label_mask: [seq_l]

        Returns:
            Scalar score.
        """
        # Similarity matrix: [seq_t, seq_l]
        sim_matrix = text_embs @ label_embs.T

        # Mask out padding label tokens (set to -inf so max ignores them)
        label_mask_expanded = label_mask.unsqueeze(0).bool()  # [1, seq_l]
        sim_matrix = sim_matrix.masked_fill(~label_mask_expanded, -1e9)

        # MaxSim: max over label tokens for each text token
        max_sim, _ = sim_matrix.max(dim=-1)  # [seq_t]

        # Mask out padding text tokens and sum
        max_sim = max_sim * text_mask.float()
        score = max_sim.sum()

        # Normalize by number of real text tokens
        score = score / text_mask.float().sum().clamp(min=1)

        return score

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with token-level MaxSim scoring.

        Returns:
            scores: [B, max_num_labels] — sigmoid of MaxSim scores
            mask:   [B, max_num_labels] — True where real labels exist
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Encode text tokens
        text_embs, text_masks = self.encode_tokens(texts)

        # Encode all label tokens
        all_labels = [label for labels in batch_labels for label in labels]
        label_embs, label_masks = self.encode_tokens(all_labels)

        # Compute MaxSim scores
        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels

        scores = torch.zeros(B, max_num_label, device=device)
        mask = torch.zeros(B, max_num_label, dtype=torch.bool, device=device)

        current = 0
        for i, count in enumerate(label_counts):
            actual = min(count, max_num_label)
            for j in range(actual):
                label_idx = current + j
                scores[i, j] = self.maxsim_score(
                    text_embs[i], text_masks[i],
                    label_embs[label_idx], label_masks[label_idx],
                )
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
