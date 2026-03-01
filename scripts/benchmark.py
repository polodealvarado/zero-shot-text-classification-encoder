"""
Benchmark script — evaluates pre-trained models without re-training.

Discovers trained models in model_output/, loads training metadata
(training time, metrics) from training_meta.json, runs inference on
the test set to measure per-sample inference time, and prints a
summary table sorted by F1.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --model-dir model_output
"""

import json
import os
import sys
import time

import typer

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ZeroShotDataset, collate_fn, load_and_split, load_and_split_from_hub, resolve_latest_dataset
from models import MODEL_REGISTRY
from scripts.train import build_targets_tensor, evaluate, suppress_hf_logging


def discover_models(base_dir: str) -> list[dict]:
    """
    Scan model_output/ for trained models with training_meta.json.

    Structure: base_dir/{encoder_label}/{model_type}/{timestamp}/training_meta.json
    For each encoder_label/model_type pair, picks the latest (most recent timestamp) run.
    """
    found = []

    if not os.path.isdir(base_dir):
        return found

    for encoder_dir in sorted(os.listdir(base_dir)):
        encoder_path = os.path.join(base_dir, encoder_dir)
        if not os.path.isdir(encoder_path):
            continue

        for model_type_dir in sorted(os.listdir(encoder_path)):
            type_path = os.path.join(encoder_path, model_type_dir)
            if not os.path.isdir(type_path):
                continue

            # Find latest timestamp subdir with training_meta.json
            runs = sorted(
                (d for d in os.listdir(type_path) if os.path.isdir(os.path.join(type_path, d))),
                reverse=True,
            )
            for run_dir in runs:
                run_path = os.path.join(type_path, run_dir)
                meta_path = os.path.join(run_path, "training_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    found.append({
                        "path": run_path,
                        "encoder_label": encoder_dir,
                        "meta": meta,
                        "model_type": meta.get("model_type", model_type_dir),
                        "encoder_name": meta.get("encoder_name", "unknown"),
                    })
                    break  # only latest run per encoder/model_type

    return found


def discover_hub_models(repo_ids: list[str]) -> list[dict]:
    """
    Download training_meta.json from HF Hub repos and return model info.

    Returns same dict format as discover_models(), with `path` set to the
    repo ID (from_pretrained accepts both local paths and Hub IDs).
    """
    from huggingface_hub import hf_hub_download

    found = []
    for repo_id in repo_ids:
        try:
            meta_path = hf_hub_download(repo_id=repo_id, filename="training_meta.json")
        except Exception as e:
            print(f"  Warning: skipping {repo_id} — could not download training_meta.json: {e}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        encoder_name = meta.get("encoder_name", "unknown")
        encoder_label = encoder_name.replace("/", "_")

        found.append({
            "path": repo_id,
            "encoder_label": encoder_label,
            "meta": meta,
            "model_type": meta.get("model_type", "unknown"),
            "encoder_name": encoder_name,
        })

    return found


@torch.no_grad()
def measure_inference_time(model, dataloader, device):
    """Measure average inference time per sample across the full test set."""
    model.eval()
    total_time = 0.0
    total_samples = 0

    for i, batch in enumerate(dataloader):
        texts = batch["texts"]
        batch_labels = batch["batch_labels"]

        # Warmup on first batch
        if i == 0:
            model(texts, batch_labels)

        start = time.time()
        model(texts, batch_labels)
        elapsed = (time.time() - start) * 1000  # ms

        total_time += elapsed
        total_samples += len(texts)

    model.train()
    return total_time / total_samples if total_samples > 0 else 0.0


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table sorted by F1, with best/worst markers."""
    sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)

    if len(sorted_results) <= 1:
        best_idx, worst_idx = 0, 0
    else:
        best_idx, worst_idx = 0, len(sorted_results) - 1

    # Column widths
    w = {"name": 45, "params": 12, "train": 10, "infer": 12, "p": 8, "r": 8, "f1": 8, "rank": 8}
    total_w = sum(w.values()) + len(w) - 1

    print(f"\n{'=' * total_w}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * total_w}")

    header = (
        f"{'Encoder / Model':<{w['name']}} "
        f"{'Params':>{w['params']}} "
        f"{'Train(s)':>{w['train']}} "
        f"{'Infer(ms/sample)':>{w['infer']}} "
        f"{'P':>{w['p']}} "
        f"{'R':>{w['r']}} "
        f"{'F1':>{w['f1']}} "
        f"{'':>{w['rank']}}"
    )
    print(header)
    print("-" * total_w)

    for i, r in enumerate(sorted_results):
        marker = ""
        if i == best_idx and len(sorted_results) > 1:
            marker = "<< BEST"
        elif i == worst_idx and len(sorted_results) > 1:
            marker = "   worst"

        name = f"{r.get('encoder_label', '?')}/{r['model_type']}"
        if len(name) > w["name"]:
            name = name[:w["name"] - 2] + ".."

        row = (
            f"{name:<{w['name']}} "
            f"{r['param_count']:>{w['params']},} "
            f"{r['train_time_s']:>{w['train']}.1f} "
            f"{r['inference_ms_per_sample']:>{w['infer']}.2f} "
            f"{r['precision']:>{w['p']}.4f} "
            f"{r['recall']:>{w['r']}.4f} "
            f"{r['f1']:>{w['f1']}.4f} "
            f"{marker}"
        )
        print(row)

    print(f"{'=' * total_w}")


def run_benchmark(config_path: str = "config.yaml", model_dir: str | None = None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["dataset"]

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # Discover models: --model-dir → local scan, else → Hub repos from config
    if model_dir is not None:
        models_found = discover_models(model_dir)
        if not models_found:
            print(f"\nNo trained models found in {model_dir}/")
            print("Train models first with: uv run python scripts/train.py")
            return
    else:
        hub_repo_ids = config.get("benchmark", {}).get("models", [])
        if hub_repo_ids:
            print(f"Fetching {len(hub_repo_ids)} model(s) from HF Hub...")
            models_found = discover_hub_models(hub_repo_ids)
        else:
            fallback_dir = config.get("training", {}).get("model_dir", "model_output")
            print(f"No benchmark.models in config, falling back to local scan: {fallback_dir}/")
            models_found = discover_models(fallback_dir)

        if not models_found:
            print("\nNo models found. Add repo IDs to benchmark.models in config.yaml")
            print("or train models first with: uv run python scripts/train.py")
            return

    # Load test data — prefer HF Hub dataset, fall back to local data_dir
    dataset_repo_id = data_cfg.get("dataset_repo_id")
    if dataset_repo_id:
        print(f"Loading dataset from HF Hub: {dataset_repo_id}")
        _, test_data, all_labels = load_and_split_from_hub(dataset_repo_id)
    else:
        data_path = resolve_latest_dataset(data_cfg["data_dir"])
        _, test_data, all_labels = load_and_split(data_path=data_path)

    test_dataset = ZeroShotDataset(
        data=test_data,
        all_labels=all_labels,
        max_negatives=data_cfg.get("max_negatives", 3),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("training", {}).get("batch_size", 8),
        shuffle=False,
        collate_fn=collate_fn,
    )

    sep = "=" * 70
    n_models = len(models_found)

    print(f"\n{sep}")
    print(f"  BENCHMARK — {n_models} model(s)")
    print(f"{sep}")
    print(f"  Device: {device}")
    print(f"  Test: {len(test_dataset)} samples")
    for m in models_found:
        print(f"  - {m['encoder_label']}/{m['model_type']} ({m['encoder_name']})")
    print(f"{sep}\n")

    results = []

    for idx, m in enumerate(models_found, 1):
        meta = m["meta"]
        model_type = m["model_type"]
        encoder_label = m.get("encoder_label", "unknown")
        print(f"  [{idx}/{n_models}] {encoder_label}/{model_type} ({m['encoder_name']})")

        # Load pre-trained model
        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            print(f"         Skipping — unknown model type: {model_type}")
            continue

        try:
            with suppress_hf_logging():
                model = model_cls.from_pretrained(m["path"])
        except Exception as e:
            print(f"         Skipping — failed to load: {e}")
            continue

        model.to(device)
        model.eval()

        # Measure per-sample inference time on test set
        inference_ms = measure_inference_time(model, test_loader, device)
        print(f"         Inference: {inference_ms:.2f} ms/sample")

        results.append({
            "model_type": model_type,
            "encoder_label": encoder_label,
            "encoder_name": m["encoder_name"],
            "path": m["path"],
            "param_count": meta.get("param_count", 0),
            "train_time_s": meta.get("train_time_s", 0),
            "inference_ms_per_sample": round(inference_ms, 4),
            "precision": meta.get("precision", 0),
            "recall": meta.get("recall", 0),
            "f1": meta.get("f1", 0),
        })

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        print("No models could be loaded.")
        return

    # Print summary table
    print_results_table(results)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"  BENCHMARK COMPLETE — {len(results)} model(s) evaluated")
    print(f"{sep}")
    best = max(results, key=lambda r: r["f1"])
    print(f"  Best F1: {best['f1']:.4f} — {best['encoder_label']}/{best['model_type']}")
    print(f"  Results saved to {output_path}")
    print(f"{sep}\n")

    return results


def main(
    config: str = typer.Option("config.yaml", help="Path to config file"),
    model_dir: str | None = typer.Option(None, help="Override model output directory"),
) -> None:
    run_benchmark(config_path=config, model_dir=model_dir)


if __name__ == "__main__":
    typer.run(main)
