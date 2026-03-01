"""
ConvMatch Model for Zero-Shot Text Classification.

Inspired by the efficiency philosophy of Zaratiana's work and the insight
from "String Representation of Entities" that surface-level text patterns
carry significant signal for matching:
    - Replaces the transformer encoder entirely with multi-scale CNNs
      over pretrained word embeddings.
    - Parallel Conv1d filters with kernel sizes [2,3,4,5] capture n-gram
      patterns at different scales.
    - Global max pooling extracts the strongest activation per filter.
    - A projection layer maps to a shared similarity space.

Why ConvMatch?
    - Orders of magnitude faster than transformer-based models at inference.
    - CNNs capture local n-gram patterns that are often sufficient for
      topic/category classification.
    - The multi-scale design captures both bigram collocations and longer
      phrasal patterns.
    - Uses pretrained embeddings from BERT as initialization, combining
      the representational quality of transformers with CNN efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer


class ConvMatchModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="zero-shot-biencoder",
    repo_url="https://github.com/your-username/zero-shot-classifier",
):
    """
    Multi-scale CNN model for zero-shot classification.
    Uses parallel convolutions over pretrained word embeddings
    instead of a full transformer encoder.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        num_filters: int = 128,
        projection_dim: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.num_filters = num_filters
        self.projection_dim = projection_dim

        # Extract pretrained word embeddings from transformer, then discard it
        pretrained = AutoModel.from_pretrained(model_name)
        embedding_weights = pretrained.embeddings.word_embeddings.weight.data.clone()
        embed_dim = embedding_weights.size(1)
        del pretrained

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Pretrained embeddings (fine-tunable)
        self.word_embeddings = nn.Embedding.from_pretrained(
            embedding_weights, freeze=False
        )

        # Multi-scale convolutions: kernel sizes 2, 3, 4, 5
        self.kernel_sizes = [2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in self.kernel_sizes
        ])

        # Projection to shared similarity space
        self.projection = nn.Linear(len(self.kernel_sizes) * num_filters, projection_dim)

    def _cnn_encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts via multi-scale CNN + max pool + project + L2 normalize.

        Returns: [N, projection_dim]
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Word embeddings: [N, seq_len, embed_dim]
        embeddings = self.word_embeddings(inputs["input_ids"])

        # Transpose for Conv1d: [N, embed_dim, seq_len]
        x = embeddings.transpose(1, 2)

        # Pad to minimum length required by largest kernel
        max_k = max(self.kernel_sizes)
        if x.size(2) < max_k:
            pad_size = max_k - x.size(2)
            x = F.pad(x, (0, pad_size))

        # Parallel convolutions + ReLU + global max pool
        conv_outputs = []
        for conv in self.convs:
            # conv: [N, num_filters, seq_len - k + 1]
            c = F.relu(conv(x))
            # Global max pool over time: [N, num_filters]
            pooled = c.max(dim=2).values
            conv_outputs.append(pooled)

        # Concatenate all filter outputs: [N, 4 * num_filters]
        concat = torch.cat(conv_outputs, dim=1)

        # Project to similarity space: [N, projection_dim]
        projected = self.projection(concat)

        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized

    def forward(
        self,
        texts: list[str],
        batch_labels: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: CNN-encode texts and labels, compute dot-product scores.

        Returns:
            scores: [B, max_num_labels] — sigmoid probabilities per label
            mask:   [B, max_num_labels] — True where a real label exists
        """
        device = next(self.parameters()).device
        B = len(texts)

        # Encode texts and labels with CNN
        text_embeddings = self._cnn_encode(texts)

        all_labels = [label for labels in batch_labels for label in labels]
        label_embeddings = self._cnn_encode(all_labels)

        # Reconstruct batch with padding (same pattern as BiEncoder)
        label_counts = [len(labels) for labels in batch_labels]
        max_num_label = self.max_num_labels

        padded_label_embeddings = torch.zeros(
            B, max_num_label, self.projection_dim, device=device
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
