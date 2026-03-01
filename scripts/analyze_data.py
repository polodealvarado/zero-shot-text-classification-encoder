"""
Dataset quality analysis for zero-shot text classification data.

Computes label distribution, vocabulary stats, text length analysis,
multi-label complexity, label co-occurrence, train/test overlap,
and stratification balance.

Usage:
    uv run python scripts/analyze_data.py
    uv run python scripts/analyze_data.py --data-path data/custom.json --save-report
"""

import json
import math
import os
import sys
from collections import Counter
from itertools import combinations

import typer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dataset import load_and_split, resolve_latest_dataset


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def compute_label_stats(data: list[dict]) -> dict:
    """Label frequency distribution and diversity metrics."""
    label_counter: Counter = Counter()
    for ex in data:
        label_counter.update(ex["labels"])

    total_annotations = sum(label_counter.values())
    unique_labels = len(label_counter)
    most_common = label_counter.most_common(10)
    least_common = (
        label_counter.most_common()[:-11:-1]
        if unique_labels > 10
        else label_counter.most_common()[::-1]
    )

    # Label frequency entropy (higher = more uniform)
    entropy = 0.0
    for count in label_counter.values():
        p = count / total_annotations
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(unique_labels) if unique_labels > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "unique_labels": unique_labels,
        "total_annotations": total_annotations,
        "top_10_labels": [(label, count) for label, count in most_common],
        "bottom_10_labels": [(label, count) for label, count in least_common],
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


def compute_vocabulary_stats(data: list[dict]) -> dict:
    """Type-Token Ratio and hapax legomena for text vocabulary."""
    all_tokens: list[str] = []
    for ex in data:
        all_tokens.extend(ex["text"].lower().split())

    total_tokens = len(all_tokens)
    token_counter = Counter(all_tokens)
    unique_tokens = len(token_counter)
    hapax = sum(1 for count in token_counter.values() if count == 1)

    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": round(ttr, 4),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax / unique_tokens, 4) if unique_tokens > 0 else 0.0,
    }


def compute_text_length_stats(data: list[dict]) -> dict:
    """Text length distribution (in words)."""
    lengths = [len(ex["text"].split()) for ex in data]

    if not lengths:
        return {"count": 0}

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    return {
        "count": n,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": round(sum(lengths) / n, 1),
        "median": lengths_sorted[n // 2],
        "p10": lengths_sorted[int(n * 0.1)],
        "p90": lengths_sorted[int(n * 0.9)],
        "std": round((sum((x - sum(lengths) / n) ** 2 for x in lengths) / n) ** 0.5, 1),
    }


def compute_multilabel_complexity(data: list[dict]) -> dict:
    """Label count per example — multi-label complexity metrics."""
    label_counts = [len(ex["labels"]) for ex in data]

    if not label_counts:
        return {"count": 0}

    counter = Counter(label_counts)
    n = len(label_counts)

    return {
        "mean_labels_per_example": round(sum(label_counts) / n, 2),
        "max_labels_per_example": max(label_counts),
        "min_labels_per_example": min(label_counts),
        "distribution": {k: v for k, v in sorted(counter.items())},
        "single_label_pct": round(counter.get(1, 0) / n * 100, 1),
        "multi_label_pct": round((n - counter.get(1, 0)) / n * 100, 1),
    }


def compute_label_cooccurrence(data: list[dict], top_n: int = 10) -> dict:
    """Most frequent label pairs that appear together."""
    pair_counter: Counter = Counter()
    for ex in data:
        labels = sorted(set(ex["labels"]))
        for pair in combinations(labels, 2):
            pair_counter[pair] += 1

    top_pairs = pair_counter.most_common(top_n)

    return {
        "total_unique_pairs": len(pair_counter),
        "top_pairs": [
            {"pair": list(pair), "count": count} for pair, count in top_pairs
        ],
    }


def compute_train_test_overlap(train_data: list[dict], test_data: list[dict]) -> dict:
    """Label overlap between train and test splits."""
    train_labels = {label for ex in train_data for label in ex["labels"]}
    test_labels = {label for ex in test_data for label in ex["labels"]}

    shared = train_labels & test_labels
    test_only = test_labels - train_labels
    train_only = train_labels - test_labels

    overlap_ratio = len(shared) / len(test_labels) if test_labels else 0.0

    return {
        "train_unique_labels": len(train_labels),
        "test_unique_labels": len(test_labels),
        "shared_labels": len(shared),
        "test_only_labels": len(test_only),
        "train_only_labels": len(train_only),
        "overlap_ratio": round(overlap_ratio, 4),
        "test_only_examples": sorted(test_only)[:10] if test_only else [],
    }


def compute_stratification_balance(
    train_data: list[dict], test_data: list[dict]
) -> dict:
    """How well the train/test split preserves label proportions."""
    train_counter: Counter = Counter()
    test_counter: Counter = Counter()

    for ex in train_data:
        train_counter.update(ex["labels"])
    for ex in test_data:
        test_counter.update(ex["labels"])

    train_total = sum(train_counter.values())
    test_total = sum(test_counter.values())

    if train_total == 0 or test_total == 0:
        return {"balance_score": 0.0}

    all_labels = set(train_counter.keys()) | set(test_counter.keys())

    # Mean absolute difference in label proportions
    diffs = []
    for label in all_labels:
        train_prop = train_counter[label] / train_total
        test_prop = test_counter[label] / test_total
        diffs.append(abs(train_prop - test_prop))

    mad = sum(diffs) / len(diffs) if diffs else 0.0
    # Balance score: 1 = perfect, 0 = worst
    balance_score = max(0.0, 1.0 - mad * 10)

    return {
        "balance_score": round(balance_score, 4),
        "mean_absolute_diff": round(mad, 6),
        "num_labels_compared": len(all_labels),
    }


# ---------------------------------------------------------------------------
# Main analysis + report
# ---------------------------------------------------------------------------


def analyze_dataset(data_path: str, test_ratio: float = 0.2) -> dict:
    """Run full analysis. Returns dict with all metrics."""
    train_data, test_data, all_labels = load_and_split(
        data_path=data_path, test_ratio=test_ratio
    )
    all_data = train_data + test_data

    return {
        "total_examples": len(all_data),
        "train_examples": len(train_data),
        "test_examples": len(test_data),
        "label_stats": compute_label_stats(all_data),
        "vocabulary_stats": compute_vocabulary_stats(all_data),
        "text_length_stats": compute_text_length_stats(all_data),
        "multilabel_complexity": compute_multilabel_complexity(all_data),
        "label_cooccurrence": compute_label_cooccurrence(all_data),
        "train_test_overlap": compute_train_test_overlap(train_data, test_data),
        "stratification_balance": compute_stratification_balance(train_data, test_data),
    }


def print_report(report: dict) -> None:
    """Print formatted report to stdout."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("DATASET QUALITY REPORT")
    print(sep)

    # Overview
    print(f"\n{'--- Overview ---':^70}")
    print(f"  Total examples:    {report['total_examples']}")
    print(f"  Train examples:    {report['train_examples']}")
    print(f"  Test examples:     {report['test_examples']}")

    # Label stats
    ls = report["label_stats"]
    print(f"\n{'--- Label Distribution ---':^70}")
    print(f"  Unique labels:        {ls['unique_labels']}")
    print(f"  Total annotations:    {ls['total_annotations']}")
    print(
        f"  Entropy:              {ls['entropy']} (normalized: {ls['normalized_entropy']})"
    )
    print(f"\n  Top 10 labels:")
    for label, count in ls["top_10_labels"]:
        bar = "#" * min(count, 40)
        print(f"    {label:<30} {count:>4}  {bar}")

    # Vocabulary
    vs = report["vocabulary_stats"]
    print(f"\n{'--- Vocabulary ---':^70}")
    print(f"  Total tokens:       {vs['total_tokens']:,}")
    print(f"  Unique tokens:      {vs['unique_tokens']:,}")
    print(f"  Type-Token Ratio:   {vs['type_token_ratio']}")
    print(
        f"  Hapax legomena:     {vs['hapax_legomena']:,} ({vs['hapax_ratio']:.1%} of vocabulary)"
    )

    # Text length
    tl = report["text_length_stats"]
    print(f"\n{'--- Text Length (words) ---':^70}")
    print(f"  Mean: {tl['mean']}  |  Median: {tl['median']}  |  Std: {tl['std']}")
    print(
        f"  Min: {tl['min']}  |  Max: {tl['max']}  |  P10: {tl['p10']}  |  P90: {tl['p90']}"
    )

    # Multi-label complexity
    mc = report["multilabel_complexity"]
    print(f"\n{'--- Multi-label Complexity ---':^70}")
    print(f"  Mean labels/example:   {mc['mean_labels_per_example']}")
    print(
        f"  Range:                 {mc['min_labels_per_example']} - {mc['max_labels_per_example']}"
    )
    print(f"  Single-label:          {mc['single_label_pct']}%")
    print(f"  Multi-label:           {mc['multi_label_pct']}%")
    print(f"  Distribution:          {mc['distribution']}")

    # Co-occurrence
    co = report["label_cooccurrence"]
    print(f"\n{'--- Label Co-occurrence ---':^70}")
    print(f"  Unique label pairs:  {co['total_unique_pairs']}")
    if co["top_pairs"]:
        print(f"  Top pairs:")
        for item in co["top_pairs"]:
            print(f"    {item['pair'][0]} + {item['pair'][1]}: {item['count']}")

    # Train/test overlap
    ov = report["train_test_overlap"]
    print(f"\n{'--- Train/Test Label Overlap ---':^70}")
    print(
        f"  Train labels: {ov['train_unique_labels']}  |  Test labels: {ov['test_unique_labels']}"
    )
    print(
        f"  Shared: {ov['shared_labels']}  |  Test-only: {ov['test_only_labels']}  |  Train-only: {ov['train_only_labels']}"
    )
    print(f"  Overlap ratio: {ov['overlap_ratio']:.1%}")
    if ov["test_only_examples"]:
        print(f"  Test-only labels (sample): {ov['test_only_examples']}")

    # Stratification balance
    sb = report["stratification_balance"]
    print(f"\n{'--- Stratification Balance ---':^70}")
    print(f"  Balance score:      {sb['balance_score']} (1.0 = perfect)")
    print(f"  Mean abs diff:      {sb['mean_absolute_diff']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    data_path: str = typer.Option(
        None, help="Path to HF DatasetDict directory (default: latest in data/)"
    ),
    test_ratio: float = typer.Option(0.2, help="Train/test split ratio"),
    save_report: bool = typer.Option(
        False, "--save-report", help="Save report as JSON"
    ),
    output_path: str = typer.Option(
        "results/data_analysis.json", help="Output path for JSON report"
    ),
) -> None:
    """Analyze dataset quality for zero-shot text classification."""
    if data_path is None:
        data_path = resolve_latest_dataset("data_dir")
    report = analyze_dataset(data_path=data_path, test_ratio=test_ratio)
    print_report(report)

    if save_report:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
