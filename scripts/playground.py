"""
Interactive Gradio playground for zero-shot text classification.

Lets you select a trained model, type texts and labels, and see predictions
with configurable threshold filtering.

Usage:
    uv run python scripts/playground.py
"""

import json
import os
import sys

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MODEL_REGISTRY
from scripts.benchmark import discover_models

MODEL_DIR = "model_output"

# Device detection (cuda > mps > cpu)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Cache: model_path -> (model, meta dict)
_model_cache: dict[str, tuple] = {}


def get_model_choices() -> list[tuple[str, str]]:
    """Return (display_name, path) pairs for the dropdown."""
    models = discover_models(MODEL_DIR)
    choices = []
    for m in models:
        label = f"{m['model_type']} ({m['encoder_label']})"
        choices.append((label, m["path"]))
    return choices


def load_model(model_path: str) -> tuple:
    """Load a model from a local path or HF Hub repo ID, with caching.

    Returns (model, meta_dict).
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    # Local model: read training_meta.json from disk
    # Hub model: download training_meta.json first
    if os.path.isdir(model_path):
        meta_path = os.path.join(model_path, "training_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        meta_file = hf_hub_download(repo_id=model_path, filename="training_meta.json")
        with open(meta_file) as f:
            meta = json.load(f)

    model_type = meta["model_type"]
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    _model_cache[model_path] = (model, meta)
    return model, meta


def format_model_info(source: str, meta: dict) -> str:
    """Build a markdown summary of the loaded model."""
    model_type = meta.get("model_type", "unknown")
    encoder = meta.get("encoder_name", "unknown")
    params = meta.get("param_count", 0)
    f1 = meta.get("f1", 0)
    precision = meta.get("precision", 0)
    recall = meta.get("recall", 0)

    return (
        f"**Loaded model:** `{source}`\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Type | {model_type} |\n"
        f"| Encoder | {encoder} |\n"
        f"| Params | {params:,} |\n"
        f"| F1 / P / R | {f1:.4f} / {precision:.4f} / {recall:.4f} |"
    )


def resolve_model_source(dropdown_value: str, hub_repo_id: str) -> str:
    """Pick Hub repo ID if provided, otherwise fall back to local dropdown."""
    hub = hub_repo_id.strip() if hub_repo_id else ""
    if hub:
        return hub
    return dropdown_value or ""


@torch.no_grad()
def predict(
    model_path: str, hub_repo_id: str,
    texts_raw: str, labels_raw: str, threshold: float,
) -> tuple[str, str]:
    """Run prediction and return (model_info, results) markdown strings."""
    source = resolve_model_source(model_path, hub_repo_id)
    if not source:
        return "", "Select a local model or enter a HF Hub repo ID."
    if not texts_raw.strip():
        return "", "Enter at least one text."
    if not labels_raw.strip():
        return "", "Enter at least one label."

    texts = [t.strip() for t in texts_raw.strip().splitlines() if t.strip()]
    labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

    try:
        model, meta = load_model(source)
    except Exception as e:
        return "", f"Failed to load model: {e}"

    model_info = format_model_info(source, meta)

    batch_labels = [labels] * len(texts)
    results = model.predict(texts, batch_labels)

    output_parts = []
    for r in results:
        text = r["text"]
        scores = r["scores"]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        lines = [f"**Text:** {text}", ""]
        lines.append("| Label | Score | Predicted |")
        lines.append("|-------|------:|:---------:|")
        for label, score in sorted_scores:
            predicted = "Yes" if score >= threshold else ""
            lines.append(f"| {label} | {score:.4f} | {predicted} |")

        output_parts.append("\n".join(lines))

    return model_info, "\n\n---\n\n".join(output_parts)


def build_ui():
    """Build and return the Gradio interface."""
    choices = get_model_choices()

    with gr.Blocks(title="Zero-Shot Classification Playground") as demo:
        gr.Markdown("# Zero-Shot Classification Playground")
        gr.Markdown(f"Device: `{DEVICE}` | Models found: **{len(choices)}**")

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    label="Local model",
                    info="Trained models from model_output/",
                )
                hub_input = gr.Textbox(
                    label="HF Hub repo ID (overrides local)",
                    placeholder="username/my-zero-shot-model",
                )
                texts_input = gr.Textbox(
                    label="Texts (one per line)",
                    placeholder="The stock market rallied today.\nThe team won the championship.",
                    lines=4,
                )
                labels_input = gr.Textbox(
                    label="Labels (comma-separated)",
                    placeholder="Finance, Sports, Technology, Politics",
                )
                threshold_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="Threshold",
                )
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                model_info = gr.Markdown()
                output = gr.Markdown(label="Results")

        predict_btn.click(
            fn=predict,
            inputs=[model_dropdown, hub_input, texts_input, labels_input, threshold_slider],
            outputs=[model_info, output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
