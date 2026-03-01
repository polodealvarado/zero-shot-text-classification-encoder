"""
LLM-as-a-Judge evaluation with adaptive difficulty using Gemini 2.5 Flash.

Gemini generates evaluation samples at progressive difficulty levels (Easy,
Medium, Hard), the model predicts on them, and Gemini judges the predictions.
Difficulty adapts based on the model's performance each round.

Usage:
    uv run python scripts/llm_judge.py
    uv run python scripts/llm_judge.py --num-rounds 10 --batch-size 3
    uv run python scripts/llm_judge.py --model username/biencoder
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager

import typer
import yaml

import torch
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

from models import MODEL_REGISTRY


@contextmanager
def suppress_hf_logging():
    """Temporarily suppress verbose HuggingFace weight-loading logs."""
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

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY is not set. Add it to .env or export it.")
    sys.exit(1)

GEMINI_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = {
    1: {
        "name": "Easy",
        "description": "Clear text with semantically distant candidate labels",
        "num_correct": "1",
        "num_candidates": "3-4",
    },
    2: {
        "name": "Medium",
        "description": "Multi-label text with some semantic proximity between candidates",
        "num_correct": "2-3",
        "num_candidates": "5-6",
    },
    3: {
        "name": "Hard",
        "description": "Ambiguous text with semantically very close candidate labels",
        "num_correct": "2-4",
        "num_candidates": "7-8",
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """You are an expert at creating evaluation samples for a zero-shot text classification model.

Generate exactly {batch_size} evaluation samples at difficulty level: {level_name} ({level_description}).

RULES:
- Each sample must have:
  - "text": A natural language sentence (1-3 sentences)
  - "ground_truth": A list of {num_correct} correct label(s) for the text
  - "candidate_labels": A list of {num_candidates} candidate labels that INCLUDES all ground_truth labels plus distractors
- Difficulty "{level_name}" means: {level_description}
- Cover diverse domains: science, technology, sports, politics, health, arts, education, business, environment, history, philosophy, law, entertainment, travel, food, psychology, etc.
- Labels should be short (1-3 words), conceptual categories
- Make texts realistic and varied in style
- Every label in ground_truth MUST appear in candidate_labels
- Shuffle the order of candidate_labels (don't put ground truth first)

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{"text": "...", "ground_truth": ["Label1"], "candidate_labels": ["Label1", "Label2", "Label3"]}}
]
"""

JUDGE_PROMPT = """You are an expert evaluator for a zero-shot text classification model.

Given:
- TEXT: The input text
- GROUND TRUTH LABELS: The correct labels
- PREDICTED SCORES: The model's predicted scores for each candidate label (higher = more confident)
- THRESHOLD: 0.5 (scores >= 0.5 are considered positive predictions)

Evaluate the model's predictions and respond with a JSON object (no markdown fences):
{{
    "correctness": <0-10 score>,
    "reasoning": "<1-2 sentences explaining the evaluation>",
    "missed_labels": "<any obvious labels the model should have predicted but didn't, or 'none'>",
    "false_positives": "<any labels incorrectly predicted as positive, or 'none'>"
}}

TEXT: {text}
GROUND TRUTH LABELS: {ground_truth}
CANDIDATE LABELS: {candidate_labels}
PREDICTED SCORES: {predicted_scores}
"""

# ---------------------------------------------------------------------------
# Model loading (unchanged)
# ---------------------------------------------------------------------------


def resolve_latest_model(base_dir: str = "model_output") -> str:
    """Find the most recent trained model run inside base_dir.

    Walks the structure: base_dir/{encoder}/{model_type}/{timestamp}/
    and returns the path to the latest timestamp directory that contains
    a config.json file.
    """
    candidates = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Model output directory not found: {base_dir}")

    for encoder in os.listdir(base_dir):
        encoder_path = os.path.join(base_dir, encoder)
        if not os.path.isdir(encoder_path):
            continue
        for model_type in os.listdir(encoder_path):
            type_path = os.path.join(encoder_path, model_type)
            if not os.path.isdir(type_path):
                continue
            for ts_dir in os.listdir(type_path):
                run_path = os.path.join(type_path, ts_dir)
                if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "config.json")):
                    candidates.append(run_path)

    if not candidates:
        raise FileNotFoundError(f"No trained model runs found in {base_dir}")

    latest = sorted(candidates)[-1]
    print(f"Resolved latest model: {latest}")
    return latest


def _is_hf_repo_id(path: str) -> bool:
    """Check if a string looks like a HuggingFace repo ID (e.g. 'username/model')."""
    return "/" in path and not os.path.exists(path)


CONFIG_KEY_TO_MODEL = {
    "num_poly_codes": "polyencoder",
    "num_attention_heads": "dynquery",
    "max_span_width": "spanclass",
    "num_filters": "convmatch",
    "projection_dim": "projection_biencoder",
    "token_projection_dim": "late_interaction",
}


def _detect_model_type(config: dict) -> str:
    """Detect model type from config.json keys. Falls back to 'biencoder'."""
    for key, model_type in CONFIG_KEY_TO_MODEL.items():
        if key in config:
            return model_type
    return "biencoder"


def load_model(model_path: str):
    """Load a trained model from a local directory or HuggingFace repo ID."""
    is_remote = _is_hf_repo_id(model_path)

    if is_remote:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_file) as f:
            config = json.load(f)
    else:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)

    model_type = _detect_model_type(config)
    model_cls = MODEL_REGISTRY[model_type]
    loaded = model_cls.from_pretrained(model_path)
    print(f"Loaded model: {model_type} from {model_path}")
    return loaded


# ---------------------------------------------------------------------------
# JSON parsing (from generate_data.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_eval_batch(level: int, batch_size: int) -> list[dict]:
    """Generate a batch of evaluation samples at a given difficulty level."""
    cfg = DIFFICULTY_LEVELS[level]
    prompt = GENERATE_PROMPT.format(
        batch_size=batch_size,
        level_name=cfg["name"],
        level_description=cfg["description"],
        num_correct=cfg["num_correct"],
        num_candidates=cfg["num_candidates"],
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,
            max_output_tokens=4096,
        ),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    samples = _parse_json_array(text)

    # Validate structure and ground_truth subset of candidate_labels
    valid = []
    for s in samples:
        if (
            isinstance(s, dict)
            and isinstance(s.get("text"), str)
            and isinstance(s.get("ground_truth"), list)
            and isinstance(s.get("candidate_labels"), list)
            and len(s["ground_truth"]) >= 1
            and len(s["candidate_labels"]) >= len(s["ground_truth"])
            and all(gt in s["candidate_labels"] for gt in s["ground_truth"])
        ):
            valid.append(s)

    return valid


# ---------------------------------------------------------------------------
# Judging (unchanged)
# ---------------------------------------------------------------------------


def judge_prediction(text: str, ground_truth: list[str], candidate_labels: list[str], scores: dict) -> dict:
    """Ask Gemini to judge a single prediction."""
    prompt = JUDGE_PROMPT.format(
        text=text,
        ground_truth=json.dumps(ground_truth),
        candidate_labels=json.dumps(candidate_labels),
        predicted_scores=json.dumps(scores),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )

    # Gemini 2.5 Flash may return thinking + response as separate parts;
    # concatenate only non-thought text parts
    parts = response.candidates[0].content.parts
    text_parts = [p.text for p in parts if p.text and not getattr(p, "thought", False)]
    text_response = "".join(text_parts).strip()

    if text_response.startswith("```"):
        text_response = text_response.split("\n", 1)[1]
        text_response = text_response.rsplit("```", 1)[0]

    try:
        return json.loads(text_response)
    except json.JSONDecodeError:
        return {
            "correctness": -1,
            "reasoning": f"Failed to parse: {text_response[:200]}",
            "missed_labels": "parse_error",
            "false_positives": "parse_error",
        }


# ---------------------------------------------------------------------------
# Main adaptive loop
# ---------------------------------------------------------------------------


def run_llm_judge(
    model: str = None,
    num_rounds: int = 6,
    batch_size: int = 5,
):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Resolve model path
    if model is None:
        model = resolve_latest_model("model_output")

    # Load model
    with suppress_hf_logging():
        loaded_model = load_model(model)
    loaded_model.to(device)
    loaded_model.eval()

    print(f"\nAdaptive LLM Judge: {num_rounds} rounds x {batch_size} samples")
    print(f"Model: {model}")
    print(f"Gemini: {GEMINI_MODEL}")
    print()

    current_level = 1
    max_level_reached = 1
    all_results = []
    all_scores = []
    round_summaries = []
    # Per-level accumulators
    level_scores: dict[int, list[int]] = {1: [], 2: [], 3: []}

    for round_num in range(1, num_rounds + 1):
        level_cfg = DIFFICULTY_LEVELS[current_level]
        level_name = level_cfg["name"]
        print(f"--- Round {round_num}/{num_rounds} | Level {current_level} ({level_name}) ---")

        # 1. Generate samples
        try:
            samples = generate_eval_batch(current_level, batch_size)
        except Exception as e:
            print(f"  Generation failed: {e}")
            round_summaries.append({
                "round": round_num, "level": current_level,
                "level_name": level_name, "num_samples": 0,
                "avg_correctness": None, "error": str(e),
            })
            continue

        if not samples:
            print("  No valid samples generated, skipping round")
            round_summaries.append({
                "round": round_num, "level": current_level,
                "level_name": level_name, "num_samples": 0,
                "avg_correctness": None, "error": "no valid samples",
            })
            continue

        round_scores = []

        for i, sample in enumerate(samples):
            text = sample["text"]
            ground_truth = sample["ground_truth"]
            candidate_labels = sample["candidate_labels"]

            # 2. Model predicts
            predictions = loaded_model.predict([text], [candidate_labels])
            scores = predictions[0]["scores"]

            # 3. Gemini judges
            try:
                judgment = judge_prediction(text, ground_truth, candidate_labels, scores)
            except Exception as e:
                judgment = {
                    "correctness": -1,
                    "reasoning": f"API error: {str(e)}",
                    "missed_labels": "error",
                    "false_positives": "error",
                }

            result = {
                "round": round_num,
                "level": current_level,
                "level_name": level_name,
                "text": text,
                "ground_truth": ground_truth,
                "candidate_labels": candidate_labels,
                "predicted_scores": scores,
                "judgment": judgment,
            }
            all_results.append(result)

            if judgment["correctness"] >= 0:
                round_scores.append(judgment["correctness"])
                all_scores.append(judgment["correctness"])
                level_scores[current_level].append(judgment["correctness"])

            tag = ""
            if judgment["correctness"] >= 8:
                tag = " (good)"
            elif judgment["correctness"] >= 5:
                tag = " (ok)"
            else:
                tag = " (poor)"
            print(f"  [{i+1}/{len(samples)}] Score: {judgment['correctness']}/10{tag}")

            time.sleep(1)

        # Round summary
        if round_scores:
            round_avg = sum(round_scores) / len(round_scores)
        else:
            round_avg = 0.0

        round_summaries.append({
            "round": round_num,
            "level": current_level,
            "level_name": level_name,
            "num_samples": len(samples),
            "avg_correctness": round(round_avg, 1),
        })

        print(f"  Round avg: {round_avg:.1f}/10")

        # 4. Adapt difficulty
        prev_level = current_level
        if round_avg >= 8 and current_level < 3:
            current_level += 1
        elif round_avg < 5 and current_level > 1:
            current_level -= 1

        if current_level > max_level_reached:
            max_level_reached = current_level

        if current_level != prev_level:
            print(f"  Difficulty: {DIFFICULTY_LEVELS[prev_level]['name']} -> {DIFFICULTY_LEVELS[current_level]['name']}")
        print()

    # --- Summary ---
    print(f"{'='*60}")
    print("LLM JUDGE SUMMARY (Adaptive Difficulty)")
    print(f"{'='*60}")

    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"  Total samples:       {len(all_scores)}")
        print(f"  Overall avg:         {overall_avg:.1f}/10")
        print(f"  Max level reached:   {max_level_reached} ({DIFFICULTY_LEVELS[max_level_reached]['name']})")
        print(f"  Perfect scores (10): {sum(1 for s in all_scores if s == 10)}")
        print(f"  Good scores (>=8):   {sum(1 for s in all_scores if s >= 8)}")
        print(f"  Poor scores (<5):    {sum(1 for s in all_scores if s < 5)}")

        # Per-level stats
        level_stats = {}
        for lvl in (1, 2, 3):
            scores_at_lvl = level_scores[lvl]
            if scores_at_lvl:
                avg = sum(scores_at_lvl) / len(scores_at_lvl)
                level_stats[DIFFICULTY_LEVELS[lvl]["name"]] = {
                    "num_samples": len(scores_at_lvl),
                    "avg_correctness": round(avg, 2),
                    "perfect_10": sum(1 for s in scores_at_lvl if s == 10),
                    "good_8_9": sum(1 for s in scores_at_lvl if 8 <= s < 10),
                    "ok_5_7": sum(1 for s in scores_at_lvl if 5 <= s < 8),
                    "poor_below_5": sum(1 for s in scores_at_lvl if s < 5),
                }
                print(f"\n  {DIFFICULTY_LEVELS[lvl]['name']} ({len(scores_at_lvl)} samples): avg {avg:.1f}/10")

        summary = {
            "model": model,
            "num_rounds": num_rounds,
            "batch_size": batch_size,
            "total_samples": len(all_scores),
            "overall_avg_correctness": round(overall_avg, 2),
            "max_level_reached": max_level_reached,
            "level_stats": level_stats,
            "score_distribution": {
                "perfect_10": sum(1 for s in all_scores if s == 10),
                "good_8_9": sum(1 for s in all_scores if 8 <= s < 10),
                "ok_5_7": sum(1 for s in all_scores if 5 <= s < 8),
                "poor_below_5": sum(1 for s in all_scores if s < 5),
            },
        }
    else:
        summary = {"error": "No valid scores obtained"}

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "summary": summary,
        "rounds": round_summaries,
        "results": all_results,
    }
    output_path = "results/llm_judge_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")


def main(
    model: str = typer.Option(None, help="Local path or HF repo ID, e.g. 'username/biencoder' (default: from config or latest in model_output/)"),
    num_rounds: int = typer.Option(None, help="Number of adaptive evaluation rounds (default: 6)"),
    batch_size: int = typer.Option(None, help="Samples generated per round (default: 5)"),
    config: str = typer.Option("config.yaml", help="Path to config file"),
) -> None:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    judge_cfg = cfg.get("llm_judge", {})

    if model is None:
        model = judge_cfg.get("model")
    if num_rounds is None:
        num_rounds = judge_cfg.get("num_rounds", 6)
    if batch_size is None:
        batch_size = judge_cfg.get("batch_size", 5)

    run_llm_judge(model=model, num_rounds=num_rounds, batch_size=batch_size)


if __name__ == "__main__":
    typer.run(main)
