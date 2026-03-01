"""
Synthetic data generation using Google Gemini 2.5 Flash.

Generates diverse text-classification examples for training the BiEncoder model.
Each example has a text and a configurable number of semantic labels. After generation, runs a dataset
quality analysis and optionally pushes the dataset to HuggingFace Hub.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import typer
import yaml
from datasets import Dataset as HFDataset, DatasetDict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# --- Configuration ---
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
MODEL_NAME = "gemini-2.5-flash"
_DEFAULT_OUTPUT_DIR = "data"
_DEFAULT_SAMPLES_PER_REQUEST = 50  # ask for 50 examples per API call

client = genai.Client(api_key=API_KEY)

PROMPT_TEMPLATE = """Generate exactly {n} diverse text classification examples.

RULES:
- Each example must have a "text" field (1-3 sentences, natural language) and a "labels" field.
- CRITICAL: Each example MUST have MINIMUM {min_labels} and MAXIMUM {max_labels} labels. Any example outside this range will be discarded.
- Vary the number of labels across the full range ({min_labels}-{max_labels}). Distribute uniformly.
- Labels should be short (1-3 words), conceptual categories (e.g. "Finance", "Machine Learning", "Climate Change").
- Cover a WIDE variety of domains: science, technology, sports, politics, health, arts, education, business, environment, history, philosophy, law, entertainment, travel, food, psychology, etc.
- Make texts realistic and varied in style: news headlines, statements, questions, descriptions.
- Do NOT repeat topics or labels across examples. Be creative and diverse.
- Batch number: {batch_id} (use this to vary your outputs)

Return ONLY a valid JSON array, no markdown, no explanation. Example format:
__EXAMPLES__
"""


def _build_prompt_examples(min_labels: int, max_labels: int) -> str:
    """Build prompt examples that match the configured label range."""
    examples = []

    # Example with min_labels
    if min_labels == 1:
        examples.append(
            '  {"text": "The stock market crashed yesterday.", "labels": ["Finance"]}'
        )
    elif min_labels == 2:
        examples.append(
            '  {"text": "The stock market crashed yesterday.", "labels": ["Finance", "Economy"]}'
        )
    else:
        labels = ["Finance", "Economy", "Markets", "Investment", "Banking"][:min_labels]
        examples.append(
            f'  {{"text": "The stock market crashed yesterday.", "labels": {json.dumps(labels)}}}'
        )

    # Example with a mid-range label count
    mid = max(min_labels, (min_labels + max_labels) // 2)
    mid_labels = ["Biology", "Environment", "Animals", "Conservation", "Ecology",
                  "Biodiversity", "Zoology", "Tropics", "Research", "Discovery"][:mid]
    examples.append(
        f'  {{"text": "A new species of bird was discovered in the Amazon.", "labels": {json.dumps(mid_labels)}}}'
    )

    # Example with max_labels
    max_labels_list = ["Economics", "Public Policy", "Ethics", "Philosophy", "Automation",
                       "Social Justice", "Labor", "Technology", "Politics", "Welfare"][:max_labels]
    examples.append(
        f'  {{"text": "The debate over universal basic income spans multiple disciplines.", "labels": {json.dumps(max_labels_list)}}}'
    )

    return "[\n" + ",\n".join(examples) + "\n]"


def _parse_json_array(text: str) -> list:
    """Parse a JSON array, repairing truncation if needed.

    LLMs sometimes return truncated JSON (hit token limit mid-object).
    Strategy: try strict parse first, then find the last complete object
    and close the array.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the last complete object by looking for "}," or "}\n]" patterns.
    # Walk backwards to find the last valid closing brace of a complete object.
    last_good = -1
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
            if depth_brace == 0 and depth_bracket == 1:
                last_good = i
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1

    if last_good > 0:
        repaired = text[:last_good + 1] + "]"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not repair JSON array", text, 0)


def generate_batch(batch_id: int, n: int = _DEFAULT_SAMPLES_PER_REQUEST, min_labels: int = 1, max_labels: int = 4) -> list:
    """Generate a batch of synthetic examples via Gemini."""
    examples = _build_prompt_examples(min_labels, max_labels)
    prompt = PROMPT_TEMPLATE.format(n=n, batch_id=batch_id, min_labels=min_labels, max_labels=max_labels).replace("__EXAMPLES__", examples)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1.0,  # high diversity
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    text = response.text.strip()
    # Clean potential markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return _parse_json_array(text)


def validate_example(ex: dict, min_labels: int = 1, max_labels: int = 4) -> bool:
    """Check that an example has the required structure."""
    return (
        isinstance(ex, dict)
        and "text" in ex
        and "labels" in ex
        and isinstance(ex["text"], str)
        and isinstance(ex["labels"], list)
        and min_labels <= len(ex["labels"]) <= max_labels
        and all(isinstance(l, str) for l in ex["labels"])
    )


def generate_dataset_card(report: dict, repo_id: str, min_labels: int = 1, max_labels: int = 4) -> str:
    """Generate a HuggingFace dataset card (README.md) from analysis report."""
    total = report["total_examples"]
    ls = report["label_stats"]
    vs = report["vocabulary_stats"]
    mc = report["multilabel_complexity"]
    tl = report["text_length_stats"]

    # Size category for HF metadata
    if total < 1000:
        size_cat = "n<1K"
    elif total < 10000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    # Top labels table
    top_labels_rows = ""
    for label, count in ls["top_10_labels"]:
        top_labels_rows += f"| {label} | {count} |\n"

    return f"""---
language:
  - en
license: mit
task_categories:
  - zero-shot-classification
  - text-classification
tags:
  - synthetic
  - gemini
  - zero-shot
  - multi-label
size_categories:
  - {size_cat}
---

# {repo_id.split('/')[-1] if '/' in repo_id else repo_id}

Synthetic dataset for zero-shot multi-label text classification, generated with Google Gemini 2.5 Flash.

## Dataset Structure

Each example contains:
- **`text`**: A natural language sentence (news headlines, statements, questions, descriptions)
- **`labels`**: A list of {min_labels}-{max_labels} semantic topic labels

```json
{{
  "text": "The stock market crashed yesterday.",
  "labels": ["Finance", "Economy"]
}}
```

## Statistics

| Metric | Value |
|--------|-------|
| Total examples | {total} |
| Unique labels | {ls['unique_labels']} |
| Total annotations | {ls['total_annotations']} |
| Avg labels/example | {mc['mean_labels_per_example']} |
| Vocabulary size | {vs['unique_tokens']:,} |
| Type-Token Ratio | {vs['type_token_ratio']} |
| Label entropy (normalized) | {ls['normalized_entropy']} |

## Top 10 Labels

| Label | Count |
|-------|-------|
{top_labels_rows}
## Text Length Distribution (words)

| Metric | Value |
|--------|-------|
| Mean | {tl['mean']} |
| Median | {tl['median']} |
| Min | {tl['min']} |
| Max | {tl['max']} |
| Std | {tl['std']} |

## Generation Methodology

- **Generator model**: Google Gemini 2.5 Flash
- **Temperature**: 1.0 (high diversity)
- **Labels per example**: {min_labels}-{max_labels}
- **Style**: News headlines, statements, questions, descriptions
- **Domains**: Science, technology, sports, politics, health, arts, education, business, environment, history, philosophy, law, entertainment, travel, food, psychology, and more
"""


def push_dataset_to_hub(ds: DatasetDict, report: dict, config_path: str = "config.yaml", min_labels: int = 1, max_labels: int = 4) -> None:
    """Push dataset to HuggingFace Hub if configured."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    if not dataset_cfg.get("push_to_hub", False):
        return

    repo_id = dataset_cfg.get("dataset_repo_id", "")
    if not repo_id:
        print("Warning: dataset.push_to_hub is true but dataset.dataset_repo_id is empty. Skipping dataset push.")
        return

    from huggingface_hub import HfApi

    private = dataset_cfg.get("private", False)

    print(f"\nPushing dataset to HuggingFace Hub: {repo_id} (private={private})")

    # Push dataset (creates repo automatically)
    ds.push_to_hub(repo_id, private=private)

    # Upload dataset card
    api = HfApi()
    card_content = generate_dataset_card(report, repo_id, min_labels=min_labels, max_labels=max_labels)
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def main(
    output_dir: str = typer.Option(None, help="Base output directory (default: from config.yaml)"),
    target_samples: int = typer.Option(None, help="Number of samples to generate (default: from config.yaml)"),
    samples_per_request: int = typer.Option(_DEFAULT_SAMPLES_PER_REQUEST, help="Samples per API request"),
    config: str = typer.Option("config.yaml", help="Path to config file"),
) -> None:
    from datetime import datetime

    # Read defaults from config
    with open(config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("dataset", {})
    if target_samples is None:
        target_samples = data_cfg.get("target_samples", 1000)
    if output_dir is None:
        output_dir = data_cfg.get("data_dir", _DEFAULT_OUTPUT_DIR)
    test_ratio = data_cfg.get("test_ratio", 0.2)
    min_labels = max(1, data_cfg.get("min_labels", 1))
    max_labels = max(min_labels, data_cfg.get("max_labels", 4))

    # Timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, timestamp)

    all_data = []
    num_requests = (target_samples // samples_per_request) + 1

    pbar = tqdm(total=target_samples, desc="Generating samples", unit="ex")

    for batch_id in range(num_requests):
        if len(all_data) >= target_samples:
            break

        remaining = target_samples - len(all_data)
        n = min(samples_per_request, remaining)

        try:
            batch = generate_batch(batch_id, n, min_labels=min_labels, max_labels=max_labels)
            valid = [ex for ex in batch if validate_example(ex, min_labels=min_labels, max_labels=max_labels)]
            if len(valid) < n:
                rejected = len(batch) - len(valid)
                bad_counts = [len(ex.get("labels", [])) for ex in batch if not validate_example(ex, min_labels=min_labels, max_labels=max_labels)]
                pbar.write(f"Batch {batch_id + 1}: {len(valid)}/{n} examples ({rejected} rejected, label counts: {bad_counts[:10]})")
            else:
                pbar.write(f"Batch {batch_id + 1}: {len(valid)}/{n} examples")
            all_data.extend(valid)
            pbar.update(len(valid))
        except Exception as e:
            pbar.write(f"Batch {batch_id + 1} ERROR: {e}")

        time.sleep(2)

    pbar.close()

    # Trim to exact target
    all_data = all_data[:target_samples]

    # Split into train/test
    rng = random.Random(42)
    indices = list(range(len(all_data)))
    rng.shuffle(indices)
    split_idx = int(len(all_data) * (1 - test_ratio))
    train_data = [all_data[i] for i in indices[:split_idx]]
    test_data = [all_data[i] for i in indices[split_idx:]]

    # Save as HF DatasetDict with train/test splits
    ds_dict = DatasetDict({
        "train": HFDataset.from_list(train_data),
        "test": HFDataset.from_list(test_data),
    })
    ds_dict.save_to_disk(output_path)

    # --- Export as JSON (if configured) ---
    export_json = data_cfg.get("export_json", False)
    if export_json:
        json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, "synthetic_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"Exported JSON: {json_path} ({len(all_data)} examples)")

    # Stats
    all_labels = set()
    for ex in all_data:
        all_labels.update(ex["labels"])

    print(f"\nDone! Saved {len(all_data)} examples to {output_path} (train: {len(train_data)}, test: {len(test_data)})")
    print(f"Unique labels: {len(all_labels)}")
    print(
        f"Avg labels per example: {sum(len(ex['labels']) for ex in all_data) / len(all_data):.1f}"
    )

    # --- Dataset quality analysis ---
    from scripts.analyze_data import analyze_dataset, print_report as print_analysis_report

    report = analyze_dataset(data_path=output_path, test_ratio=test_ratio)
    print_analysis_report(report)

    # --- Push to HuggingFace Hub (if configured) ---
    push_dataset_to_hub(ds_dict, report, config_path=config, min_labels=min_labels, max_labels=max_labels)


if __name__ == "__main__":
    typer.run(main)
