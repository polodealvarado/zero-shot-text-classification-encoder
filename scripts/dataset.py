"""
Dataset for zero-shot text classification training.

Key concept — Negative Sampling:
    For each text, we know which labels are correct (positives).
    But the model also needs to learn what is NOT correct (negatives).
    We randomly sample labels from OTHER examples in the dataset and add them
    as negative examples (target=0). This teaches the model to discriminate.

    Example:
        Text: "The stock market crashed"
        Positive labels: ["Finance", "Economy"]         -> target = 1
        Negative labels: ["Biology", "Sports", "Music"] -> target = 0
        Combined:        ["Finance", "Economy", "Biology", "Sports", "Music"]
        Targets:         [1, 1, 0, 0, 0]
"""

import os
import random

from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset


def resolve_latest_dataset(base_dir: str) -> str:
    """Return the most recent timestamped subdirectory inside base_dir."""
    subdirs = sorted(
        (d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))),
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No dataset runs found in {base_dir}")
    latest = os.path.join(base_dir, subdirs[0])
    print(f"Resolved latest dataset: {latest}")
    return latest


def load_and_split(
    data_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Load train/test splits from an HF DatasetDict directory.

    The global label pool is built from ALL data (train+test) so that
    negative sampling during training can use labels that only appear in test.

    Args:
        data_path: Path to HF DatasetDict directory (Arrow format).
        test_ratio: Unused — kept for API compatibility. Splits are pre-computed.
        seed: Unused — kept for API compatibility.

    Returns:
        (train_data, test_data, all_labels)
    """
    ds = load_from_disk(data_path)

    train_data = [{"text": row["text"], "labels": row["labels"]} for row in ds["train"]]
    test_data = [{"text": row["text"], "labels": row["labels"]} for row in ds["test"]]

    # Global label pool from ALL data
    all_data = train_data + test_data
    all_labels = list({label for ex in all_data for label in ex["labels"]})

    print(f"Loaded {len(all_data)} examples | Train: {len(train_data)} | Test: {len(test_data)} | Labels: {len(all_labels)}")
    return train_data, test_data, all_labels


def load_and_split_from_hub(
    repo_id: str,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Load train/test splits from an HF Hub dataset repo.

    Same return format as load_and_split() but fetches directly from Hub.
    """
    ds = load_dataset(repo_id)

    train_data = [{"text": row["text"], "labels": row["labels"]} for row in ds["train"]]
    test_data = [{"text": row["text"], "labels": row["labels"]} for row in ds["test"]]

    all_data = train_data + test_data
    all_labels = list({label for ex in all_data for label in ex["labels"]})

    print(f"Loaded {len(all_data)} examples from Hub ({repo_id}) | Train: {len(train_data)} | Test: {len(test_data)} | Labels: {len(all_labels)}")
    return train_data, test_data, all_labels


class ZeroShotDataset(Dataset):
    """
    Dataset that loads text-label pairs and applies negative sampling on-the-fly.

    Each __getitem__ returns:
        - text: the input text string
        - labels: list of label strings (positives + negatives, shuffled)
        - targets: list of floats (1.0 for positive, 0.0 for negative)
    """

    def __init__(
        self,
        data: list[dict],
        all_labels: list[str],
        max_negatives: int = 3,
        seed: int | None = None,
    ):
        """
        Args:
            data: List of {"text": ..., "labels": [...]}.
            all_labels: Global label pool (from train+test) for negative sampling.
            max_negatives: Maximum number of negative labels to sample per example.
            seed: If set, negative sampling is deterministic per item (use for test set).
        """
        self.data = data
        self.max_negatives = max_negatives
        self.all_labels = all_labels
        self.seed = seed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one training sample with positive + negative labels.

        The labels are shuffled so the model can't learn positional patterns
        (e.g., "first labels are always positive").

        When seed is set, the same negatives and shuffle order are produced
        every time for the same idx — making evaluation deterministic.
        """
        example = self.data[idx]
        text = example["text"]
        positive_labels = example["labels"]

        rng = random.Random(self.seed + idx) if self.seed is not None else random

        # Sample negatives
        positive_set = set(positive_labels)
        candidates = [l for l in self.all_labels if l not in positive_set]
        num_neg = rng.randint(1, min(self.max_negatives, len(candidates)))
        negative_labels = rng.sample(candidates, num_neg)

        # Combine and build targets
        all_labels = positive_labels + negative_labels
        targets = [1.0] * len(positive_labels) + [0.0] * len(negative_labels)

        # Shuffle to prevent positional bias
        combined = list(zip(all_labels, targets))
        rng.shuffle(combined)
        all_labels, targets = zip(*combined)

        return {
            "text": text,
            "labels": list(all_labels),
            "targets": list(targets),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for DataLoader.

    Why custom collate?
        - Each example has a different number of labels (variable length).
        - We can't use default collate which expects uniform tensor sizes.
        - Instead, we keep lists of strings and pass them to the model,
          which handles padding internally.

    Returns:
        dict with:
            - texts: list of strings (length B)
            - batch_labels: list of list of strings (length B, variable inner length)
            - targets: padded tensor [B, max_labels_in_batch]
    """
    texts = [item["text"] for item in batch]
    batch_labels = [item["labels"] for item in batch]
    targets_list = [item["targets"] for item in batch]

    return {
        "texts": texts,
        "batch_labels": batch_labels,
        "targets_list": targets_list,
    }
