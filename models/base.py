"""
BiEncoder Model for Zero-Shot Text Classification.

Architecture:
    - A single shared transformer encoder (e.g. BERT) encodes both texts and labels
      into the same embedding space.
    - Similarity is computed via dot product between text and label embeddings.
    - Sigmoid activation produces independent probabilities per label (multi-label).

Why BiEncoder?
    - At inference time, label embeddings can be pre-computed and cached.
    - Scales linearly with the number of labels (unlike CrossEncoder which is quadratic).
    - The shared encoder learns a joint semantic space where related texts and labels
      are close together — enabling zero-shot generalization to unseen labels.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class BiEncoderModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    BiEncoder for zero-shot text classification.

    The model maps both input texts and candidate labels into a shared embedding space
    using a single transformer encoder, then computes similarity scores via dot product.
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_num_labels: int = 5):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels

        # Shared encoder: same weights encode both texts and labels
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts_or_labels: list[str]) -> torch.Tensor:
        """
        Encode a list of strings into dense embeddings using mask-aware mean pooling.

        Returns: Tensor of shape [N, D] where D is the hidden dimension.
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts_or_labels, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = self.shared_encoder(**inputs)

        # Mask-aware mean pooling
        att_mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1)

        return pooled

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode texts and labels, compute similarity scores.

        Args:
            texts: List of input texts, length B (batch size).
            batch_labels: List of lists of labels for each text.

        Returns:
            scores: [B, max_num_labels] — sigmoid probabilities per label.
            mask:   [B, max_num_labels] — boolean mask (True where a real label exists).
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Flatten and encode all labels at once
        all_labels = [label for labels in batch_labels for label in labels]
        label_embeddings = self.encode(all_labels)

        # Encode texts
        text_embeddings = self.encode(texts)

        # Reconstruct batch structure with padding
        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels

        padded_label_embeddings = torch.zeros(
            B, max_num_label, label_embeddings.size(-1), device=device
        )
        mask = torch.zeros(B, max_num_label, dtype=torch.bool, device=device)

        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                actual = min(count, max_num_label)
                padded_label_embeddings[i, :actual, :] = label_embeddings[
                    current : current + actual
                ]
                mask[i, :actual] = True
                current = end

        # Dot product similarity
        scores = torch.bmm(
            padded_label_embeddings, text_embeddings.unsqueeze(2)
        ).squeeze(2)

        # Sigmoid for independent per-label probabilities
        scores = torch.sigmoid(scores)

        return scores, mask

    def compute_loss(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked Binary Cross Entropy loss.

        Args:
            scores:  [B, max_num_labels] — sigmoid outputs from forward()
            targets: [B, max_num_labels] — 1.0 for positive labels, 0.0 for negatives
            mask:    [B, max_num_labels] — True for real label positions

        Returns:
            Scalar loss value (mean over all valid positions).
        """
        loss = F.binary_cross_entropy(scores, targets, reduction="none")
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum()

    @torch.no_grad()
    def predict(
        self,
        texts: list[str],
        labels: list[list[str]],
    ) -> list[dict]:
        """
        Inference: return label scores for each text.

        Args:
            texts: List of input texts.
            labels: List of candidate label lists (one per text).

        Returns:
            List of dicts: [{"text": ..., "scores": {"label": score, ...}}, ...]
        """
        scores, mask = self.forward(texts, labels)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(labels[i]):
                if j < self.max_num_labels and mask[i, j]:
                    text_result[label] = round(scores[i, j].item(), 4)
            results.append({"text": text, "scores": text_result})
        return results
