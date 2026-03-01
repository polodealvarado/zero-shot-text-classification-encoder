"""
Training script for zero-shot classifier variants.

Supports all model types via the MODEL_REGISTRY:
    - biencoder (default)
    - projection_biencoder
    - late_interaction
    - polyencoder

Usage:
    uv run python scripts/train.py                          # default config
    uv run python scripts/train.py --model-type polyencoder # override model type
    uv run python scripts/train.py --config my_config.yaml  # custom config
"""

import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
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

MODEL_DESCRIPTIONS = {
    "biencoder": "Shared BERT encoder with dot-product similarity and sigmoid activation.",
    "projection_biencoder": "CLIP-inspired with projection heads, L2 norm, and learnable temperature.",
    "late_interaction": "ColBERT-style token-level MaxSim scoring for fine-grained alignment.",
    "polyencoder": "Learnable poly-codes with label-conditioned cross-attention.",
    "dynquery": "DyREx-inspired dynamic label queries via cross-attention over text tokens.",
    "spanclass": "GLiNER-inspired span-attentive classification with top-K span selection.",
    "convmatch": "Multi-scale CNN encoder over pretrained embeddings (no transformer at inference).",
}

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ZeroShotDataset, collate_fn, load_and_split, load_and_split_from_hub, resolve_latest_dataset
from models import MODEL_REGISTRY


@contextmanager
def suppress_hf_logging():
    """Temporarily suppress verbose HuggingFace weight-loading logs.

    The BERT checkpoint includes pre-training heads (MLM + NSP) that are discarded
    when loading as BertModel. The resulting UNEXPECTED keys report is harmless noise.
    """
    loggers = {}
    for name in ("huggingface_hub", "transformers"):
        logger = logging.getLogger(name)
        loggers[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in loggers.items():
            logging.getLogger(name).setLevel(level)


def build_targets_tensor(
    targets_list: list[list[float]],
    max_num_labels: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad variable-length target lists into a uniform tensor [B, max_num_labels]."""
    B = len(targets_list)
    targets = torch.zeros(B, max_num_labels, device=device)
    for i, t in enumerate(targets_list):
        length = min(len(t), max_num_labels)
        targets[i, :length] = torch.tensor(t[:length], device=device)
    return targets


@torch.no_grad()
def evaluate(model, dataloader, max_num_labels, device, threshold=0.5):
    """
    Evaluate model on a dataset and return Precision, Recall, F1.
    """
    model.eval()

    tp = 0
    fp = 0
    fn = 0

    for batch in dataloader:
        texts = batch["texts"]
        batch_labels = batch["batch_labels"]
        targets_list = batch["targets_list"]

        scores, mask = model(texts, batch_labels)
        targets = build_targets_tensor(targets_list, max_num_labels, device)

        preds = (scores > threshold).float()

        valid = mask.float()
        tp += ((preds == 1) & (targets == 1) & (valid == 1)).sum().item()
        fp += ((preds == 1) & (targets == 0) & (valid == 1)).sum().item()
        fn += ((preds == 0) & (targets == 1) & (valid == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    model.train()
    return {"precision": precision, "recall": recall, "f1": f1}


def build_model(model_type: str, train_cfg: dict, max_num_labels: int):
    """Instantiate a model from the registry using config parameters."""
    model_cls = MODEL_REGISTRY[model_type]

    # Build kwargs based on what the model's __init__ accepts
    kwargs = {
        "model_name": train_cfg["base_model"],
        "max_num_labels": max_num_labels,
    }

    if model_type == "projection_biencoder":
        kwargs["projection_dim"] = train_cfg.get("projection_dim", 256)
        kwargs["contrastive_alpha"] = train_cfg.get("contrastive_alpha", 0.1)
    elif model_type == "late_interaction":
        kwargs["token_projection_dim"] = train_cfg.get("token_projection_dim", 128)
    elif model_type == "polyencoder":
        kwargs["num_poly_codes"] = train_cfg.get("num_poly_codes", 16)
    elif model_type == "dynquery":
        kwargs["num_attention_heads"] = train_cfg.get("num_attention_heads", 4)
    elif model_type == "spanclass":
        kwargs["max_span_width"] = train_cfg.get("max_span_width", 5)
        kwargs["top_k_spans"] = train_cfg.get("top_k_spans", 8)
    elif model_type == "convmatch":
        kwargs["num_filters"] = train_cfg.get("num_filters", 128)
        kwargs["projection_dim"] = train_cfg.get("projection_dim", 256)

    return model_cls(**kwargs)


def generate_model_card(
    model_type: str,
    model_name: str,
    repo_id: str,
    param_count: int,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    train_time: float,
    metrics: dict,
    dataset_repo_id: str = "",
) -> str:
    """Generate a HuggingFace model card (README.md)."""
    description = MODEL_DESCRIPTIONS.get(model_type, "Zero-shot text classification model.")

    # Map model_type to its class name
    model_class_names = {
        "biencoder": "BiEncoderModel",
        "projection_biencoder": "ProjectionBiEncoderModel",
        "late_interaction": "LateInteractionModel",
        "polyencoder": "PolyEncoderModel",
        "dynquery": "DynQueryModel",
        "spanclass": "SpanClassModel",
        "convmatch": "ConvMatchModel",
    }
    class_name = model_class_names.get(model_type, "BiEncoderModel")
    module_map = {
        "biencoder": "base",
        "projection_biencoder": "projection",
        "late_interaction": "late_interaction",
        "polyencoder": "polyencoder",
        "dynquery": "dynquery",
        "spanclass": "spanclass",
        "convmatch": "convmatch",
    }
    module_name = module_map.get(model_type, "base")

    dataset_link = ""
    datasets_meta = ""
    if dataset_repo_id:
        dataset_link = f"\n## Dataset\n\nTrained on [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}).\n"
        datasets_meta = f"\ndatasets:\n  - {dataset_repo_id}"

    return f"""---
language:
  - en
license: mit
library_name: transformers
pipeline_tag: zero-shot-classification
tags:
  - zero-shot
  - multi-label
  - text-classification
  - pytorch
metrics:
  - precision
  - recall
  - f1
base_model: {model_name}{datasets_meta}
---

# Zero-Shot Text Classification — {model_type}

{description}

This model encodes texts and candidate labels into a shared embedding space using BERT,
enabling classification into arbitrary categories without retraining for new labels.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `{model_name}` |
| Model variant | `{model_type}` |
| Training steps | {num_steps} |
| Batch size | {batch_size} |
| Learning rate | {learning_rate} |
| Trainable params | {param_count:,} |
| Training time | {train_time:.1f}s |
{dataset_link}
## Evaluation Results

| Metric | Score |
|--------|-------|
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |

## Usage

```python
from models.{module_name} import {class_name}

model = {class_name}.from_pretrained("{repo_id}")

predictions = model.predict(
    texts=["The stock market crashed yesterday."],
    labels=[["Finance", "Sports", "Biology", "Economy"]],
)
print(predictions)
# [{{"text": "...", "scores": {{"Finance": 0.98, "Economy": 0.85, ...}}}}]
```
"""


def train(
    config_path: str = "config.yaml",
    model_type_override: str | None = None,
    encoder_name_override: str | None = None,
):
    # --- Load config ---
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["dataset"]

    # --- Reproducibility ---
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Model type: CLI override > config > default
    model_type = model_type_override or train_cfg.get("type", "biencoder")
    if model_type not in MODEL_REGISTRY:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # Derive max_num_labels from dataset params (max positives + max negatives)
    max_num_labels = data_cfg["max_labels"] + data_cfg["max_negatives"]

    # Encoder: CLI override > config default
    if encoder_name_override:
        train_cfg = {**train_cfg, "base_model": encoder_name_override}

    # Encoder label for output dir (sanitized model name)
    encoder_label = train_cfg["base_model"].replace("/", "_")

    # --- Device setup ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --- Model ---
    with suppress_hf_logging():
        model = build_model(model_type, train_cfg, max_num_labels)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type} ({train_cfg['base_model']}) | Params: {param_count:,} | Max labels: {max_num_labels}")

    # --- Data: train/test split — prefer HF Hub, fall back to local data_dir ---
    dataset_repo_id = data_cfg.get("dataset_repo_id")
    if dataset_repo_id:
        print(f"Loading dataset from HF Hub: {dataset_repo_id}")
        train_data, test_data, all_labels = load_and_split_from_hub(dataset_repo_id)
    else:
        data_path = resolve_latest_dataset(data_cfg["data_dir"])
        train_data, test_data, all_labels = load_and_split(data_path=data_path)

    train_dataset = ZeroShotDataset(
        data=train_data,
        all_labels=all_labels,
        max_negatives=data_cfg.get("max_negatives", 3),
    )
    test_dataset = ZeroShotDataset(
        data=test_data,
        all_labels=all_labels,
        max_negatives=data_cfg.get("max_negatives", 3),
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
    )

    # --- Output dir: model_output/{encoder_label}/{model_type}/{YYYYMMDD_HHMMSS} ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(train_cfg.get("model_dir", "model_output"), encoder_label, model_type, timestamp)

    # --- Training loop ---
    num_steps = train_cfg["num_steps"]
    log_every = train_cfg.get("log_every", 50)
    save_every = train_cfg.get("save_every", 500)
    eval_every = train_cfg.get("eval_every", 250)
    best_metric = train_cfg.get("best_metric", "f1")
    save_total_limit = train_cfg.get("save_total_limit", 2)

    model.train()
    step = 0
    running_loss = 0.0

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TRAINING — {model_type} ({train_cfg['base_model']})")
    print(f"{sep}")
    print(f"  Steps: {num_steps} | Batch: {train_cfg['batch_size']} | LR: {train_cfg['learning_rate']}")
    print(f"  Eval metric: {best_metric} | Eval every: {eval_every} | Save every: {save_every}")
    print(f"  Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")
    print(f"  Output: {output_dir}")
    print(f"{sep}\n")

    best_value = -1.0
    best_step = 0
    best_metrics = None
    best_model_dir = os.path.join(output_dir, "best_model")
    last_avg_loss = 0.0
    start_time = time.time()

    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break

            texts = batch["texts"]
            batch_labels = batch["batch_labels"]
            targets_list = batch["targets_list"]

            # Forward pass
            scores, mask = model(texts, batch_labels)

            # Build target tensor
            targets = build_targets_tensor(
                targets_list, max_num_labels, device
            )

            # Compute loss
            loss = model.compute_loss(scores, targets, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # Logging
            if step % log_every == 0:
                avg_loss = running_loss / log_every
                last_avg_loss = avg_loss
                elapsed = time.time() - start_time
                eta = elapsed / step * (num_steps - step)
                print(f"  Step {step:>5}/{num_steps} | Loss: {avg_loss:.4f} | {elapsed:.0f}s elapsed | ETA: {eta:.0f}s")
                running_loss = 0.0

            # Evaluate (single eval per step — checkpoint reuses the result)
            should_eval = step % eval_every == 0
            should_save = step % save_every == 0

            if should_eval or should_save:
                metrics = evaluate(model, test_loader, max_num_labels, device)
                metric_value = metrics.get(best_metric, metrics["f1"])
                is_new_best = metric_value > best_value

                if is_new_best:
                    best_value = metric_value
                    best_step = step
                    best_metrics = metrics.copy()
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    model.tokenizer.save_pretrained(best_model_dir)

                marker = " *best (saved)" if is_new_best else ""
                print(
                    f"  [eval] P: {metrics['precision']:.4f} | "
                    f"R: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1']:.4f}{marker}"
                )

            if should_save:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
                model.save_pretrained(ckpt_dir)

                # Prune: keep most recent save_total_limit checkpoints
                if save_total_limit > 0:
                    all_ckpts = sorted(
                        [
                            d for d in os.listdir(output_dir)
                            if d.startswith("checkpoint_step") and os.path.isdir(os.path.join(output_dir, d))
                        ],
                        key=lambda d: int(d.replace("checkpoint_step", "")),
                    )
                    while len(all_ckpts) > save_total_limit:
                        oldest = all_ckpts.pop(0)
                        shutil.rmtree(os.path.join(output_dir, oldest))

    train_time = time.time() - start_time

    # --- Final evaluation ---
    final_metrics = evaluate(model, test_loader, max_num_labels, device)
    final_value = final_metrics.get(best_metric, final_metrics["f1"])
    if final_value > best_value:
        best_value = final_value
        best_step = num_steps
        best_metrics = final_metrics.copy()
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        model.tokenizer.save_pretrained(best_model_dir)

    # If no eval ran during training, use final as best
    if best_metrics is None:
        best_metrics = final_metrics.copy()
        best_step = num_steps
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        model.tokenizer.save_pretrained(best_model_dir)

    # --- Save final model to output_dir ---
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)

    training_meta = {
        "model_type": model_type,
        "encoder_name": train_cfg["base_model"],
        "param_count": param_count,
        "num_steps": num_steps,
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "train_time_s": round(train_time, 2),
        "precision": round(final_metrics["precision"], 4),
        "recall": round(final_metrics["recall"], 4),
        "f1": round(final_metrics["f1"], 4),
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(training_meta, f, indent=2)

    # --- Save best model metadata ---
    best_training_meta = {
        "model_type": model_type,
        "encoder_name": train_cfg["base_model"],
        "param_count": param_count,
        "num_steps": num_steps,
        "best_step": best_step,
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "train_time_s": round(train_time, 2),
        "precision": round(best_metrics["precision"], 4),
        "recall": round(best_metrics["recall"], 4),
        "f1": round(best_metrics["f1"], 4),
    }
    with open(os.path.join(best_model_dir, "training_meta.json"), "w") as f:
        json.dump(best_training_meta, f, indent=2)

    # --- Structured summary ---
    thin = "-" * 40
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE — {model_type} ({train_cfg['base_model']})")
    print(f"{sep}")
    print(f"  Steps: {num_steps} | Time: {train_time:.1f}s | Final loss: {last_avg_loss:.4f}")
    print(f"  {thin}")
    print(f"  Best model (step {best_step}):")
    print(f"    Precision: {best_metrics['precision']:.4f}")
    print(f"    Recall:    {best_metrics['recall']:.4f}")
    print(f"    F1 Score:  {best_metrics['f1']:.4f}")
    print(f"  {thin}")
    print(f"  Final model (step {num_steps}):")
    print(f"    Precision: {final_metrics['precision']:.4f}")
    print(f"    Recall:    {final_metrics['recall']:.4f}")
    print(f"    F1 Score:  {final_metrics['f1']:.4f}")
    print(f"  {thin}")
    print(f"  Best model saved to {best_model_dir}")
    print(f"  Final model saved to {output_dir}")
    print(f"{sep}\n")

    # --- Push to HuggingFace Hub (only if new model improves on Hub model) ---
    if train_cfg.get("push_to_hub", False) and HF_TOKEN:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        private = train_cfg.get("private", False)
        dataset_repo_id = data_cfg.get("dataset_repo_id", "")

        # Derive repo_id as <username>/<model_type>
        username = api.whoami()["name"]
        repo_id = f"{username}/{model_type}"

        # Check if repo exists and compare metrics
        should_push = True
        repo_exists = api.repo_exists(repo_id)

        if repo_exists:
            print(f"HF Hub repo '{repo_id}' already exists. Comparing {best_metric}...")
            try:
                hub_meta_path = hf_hub_download(repo_id=repo_id, filename="training_meta.json")
                with open(hub_meta_path) as f:
                    hub_meta = json.load(f)

                hub_value = hub_meta.get(best_metric, 0.0)
                new_value = best_metrics[best_metric]

                if new_value > hub_value:
                    print(f"  New model is better: {best_metric} {new_value:.4f} > {hub_value:.4f} (Hub)")
                else:
                    print(f"  New model is not better: {best_metric} {new_value:.4f} <= {hub_value:.4f} (Hub)")
                    print("  Skipping Hub push.")
                    should_push = False
            except Exception as e:
                print(f"  Could not fetch Hub metadata ({e}). Pushing as first upload.")
        else:
            api.create_repo(repo_id, private=private)
            print(f"Created HF Hub repo: {repo_id} (private={private})")

        if should_push:
            # Generate model card for best model
            card_content = generate_model_card(
                model_type=model_type,
                model_name=train_cfg["base_model"],
                repo_id=repo_id,
                param_count=param_count,
                num_steps=num_steps,
                batch_size=train_cfg["batch_size"],
                learning_rate=train_cfg["learning_rate"],
                train_time=train_time,
                metrics=best_metrics,
                dataset_repo_id=dataset_repo_id,
            )
            card_path = os.path.join(best_model_dir, "README.md")
            with open(card_path, "w") as f:
                f.write(card_content)
            print(f"Model card saved to {card_path}")

            # Push best model folder to Hub
            print(f"\nPushing best model to HuggingFace Hub: {repo_id}")
            api.upload_folder(
                folder_path=best_model_dir,
                repo_id=repo_id,
            )
            print(f"Model pushed to https://huggingface.co/{repo_id}")
    elif train_cfg.get("push_to_hub", False) and not HF_TOKEN:
        print("Warning: push_to_hub is true but HF_TOKEN is not set. Skipping upload.")

    return {
        "model_type": model_type,
        "param_count": param_count,
        "train_time_s": train_time,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
    }


def main(
    config: str = typer.Option("config.yaml", help="Path to config file"),
    model_type: str | None = typer.Option(None, help="Override model type from config"),
    encoder: str | None = typer.Option(None, help="Override encoder (HF model ID)"),
) -> None:
    train(config_path=config, model_type_override=model_type, encoder_name_override=encoder)


if __name__ == "__main__":
    typer.run(main)
