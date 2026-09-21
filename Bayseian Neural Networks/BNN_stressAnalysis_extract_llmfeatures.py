"""Extract stress-related LLM features for BNN Experiment 02.

Important: feature extraction is model-independent. The CSV produced by this
script can also be reused for Monte Carlo Dropout and Deep Ensemble models so
that all downstream models receive exactly the same rows and LLM features.

The script:
* processes at most the first 2,000 rows;
* uses a fixed deductive codebook;
* appends one processed row at a time;
* resumes from an atomic JSON checkpoint after interruption;
* retries temporary Ollama failures;
* records whether extraction succeeded for each row.
"""



import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


# =========================================================
# CONFIGURATION -- edit these paths and TEXT_COLUMN if needed
# =========================================================
INPUT_CSV = Path("stress_analysis_normalized.csv")
OUTPUT_CSV = Path("stress_analysis_normalized_bnn_exp02_with_llm.csv")
CHECKPOINT_JSON = Path("stress_bnn_llm_checkpoint.json")

TEXT_COLUMN = "text"

# Fixed experiment settings
MAX_ROWS = 2000
MAX_CHARACTERS = 600

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"
REQUEST_TIMEOUT_SECONDS = 120
TEMPERATURE = 0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0

# Progress and throttling
PRINT_EVERY = 20
SLEEP_BETWEEN_CALLS_SECONDS = 0.0


CODEBOOK = {
    "T1_STRESS_INTENSITY": (
        "Mentions strong stress, pressure, overwhelm, or feeling stressed out."
    ),
    "T2_ANXIETY_WORRY": (
        "Mentions anxiety, worry, panic, fear, or rumination."
    ),
    "T3_BURNOUT_EXHAUSTION": (
        "Mentions burnout, exhaustion, fatigue, or feeling drained."
    ),
    "T4_SLEEP_PROBLEMS": (
        "Mentions insomnia, poor sleep, nightmares, or difficulty sleeping."
    ),
    "T5_WORK_STUDY_PRESSURE": (
        "Mentions work or study deadlines, workload, or performance pressure."
    ),
    "T6_RELATIONSHIP_SOCIAL_STRESS": (
        "Mentions family or relationship conflict, loneliness, or social stress."
    ),
    "T7_PHYSICAL_SYMPTOMS": (
        "Mentions physical stress symptoms such as headache, stomach problems, "
        "or a racing heart."
    ),
    "T8_COPING_STRATEGY": (
        "Mentions coping actions such as exercise, breaks, support, therapy, "
        "or planning."
    ),
}


def build_prompt(text: str) -> str:
    keys = list(CODEBOOK)
    return f"""
You are a qualitative research assistant performing deductive thematic coding.

Use only the fixed codebook below. For every theme return an integer:
0 = absent
1 = present

Return only one valid JSON object. Do not use Markdown, explanations, or extra
keys. The object must contain exactly these keys:
{json.dumps(keys, indent=2)}

Codebook:
{json.dumps(CODEBOOK, indent=2)}

Text:
{text}
""".strip()


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": TEMPERATURE},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if "response" not in payload:
        raise ValueError("Ollama response did not contain a 'response' field.")
    return str(payload["response"])


def extract_json_object(model_output: str) -> dict:
    """Parse a JSON object, allowing harmless text around the object."""
    start = model_output.find("{")
    end = model_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")

    parsed = json.loads(model_output[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output was not a JSON object.")
    return parsed


def normalize_binary(value) -> int:
    """Accept only recognizable Boolean/binary values."""
    if value in (1, True):
        return 1
    if value in (0, False):
        return 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "present"}:
        return 1
    if normalized in {"0", "false", "no", "absent"}:
        return 0
    raise ValueError(f"Unrecognized binary value: {value!r}")


def validate_themes(parsed: dict) -> dict[str, int]:
    expected_keys = set(CODEBOOK)
    returned_keys = set(parsed)
    if returned_keys != expected_keys:
        missing = sorted(expected_keys - returned_keys)
        extra = sorted(returned_keys - expected_keys)
        raise ValueError(f"Incorrect JSON keys; missing={missing}, extra={extra}")
    return {key: normalize_binary(parsed[key]) for key in CODEBOOK}


def extract_themes_with_retry(text: str) -> tuple[dict[str, int], bool, str]:
    prompt = build_prompt(text)
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_ollama(prompt)
            themes = validate_themes(extract_json_object(raw_output))
            return themes, True, ""
        except Exception as error:  # retain the row and record the failure
            last_error = f"{type(error).__name__}: {error}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    return {key: 0 for key in CODEBOOK}, False, last_error


def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        return max(0, int(checkpoint.get("next_row", 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print("[WARNING] Checkpoint is unreadable; processing will restart at row 0.")
        return 0


def save_checkpoint(path: Path, next_row: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"next_row": int(next_row)}),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def verify_resume_state(output_path: Path, checkpoint_row: int) -> int:
    """Ensure the checkpoint agrees with the number of completed output rows."""
    if not output_path.exists():
        if checkpoint_row > 0:
            print("[WARNING] Output is missing; resetting checkpoint to row 0.")
        return 0

    completed_rows = len(pd.read_csv(output_path, usecols=[TEXT_COLUMN]))
    if completed_rows != checkpoint_row:
        raise RuntimeError(
            "Checkpoint/output mismatch: "
            f"checkpoint next_row={checkpoint_row}, output rows={completed_rows}. "
            "Resolve this mismatch before resuming to avoid duplicated or missing rows."
        )
    return checkpoint_row


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_JSON.parent.mkdir(parents=True, exist_ok=True)

    complete_data = pd.read_csv(INPUT_CSV, low_memory=False)
    data = complete_data.head(MAX_ROWS).copy().reset_index(drop=True)
    processing_limit = len(data)

    if TEXT_COLUMN not in data.columns:
        raise KeyError(
            f"TEXT_COLUMN '{TEXT_COLUMN}' was not found. "
            f"Available columns: {data.columns.tolist()}"
        )

    start_row = verify_resume_state(
        OUTPUT_CSV,
        load_checkpoint(CHECKPOINT_JSON),
    )
    if start_row == 0 and not OUTPUT_CSV.exists():
        save_checkpoint(CHECKPOINT_JSON, 0)

    if start_row >= processing_limit:
        print(f"Nothing to do: {processing_limit} rows are already complete.")
        print(f"Output: {OUTPUT_CSV.resolve()}")
        return

    print("=" * 68)
    print("Stress LLM feature extraction — BNN Experiment 02")
    print(f"Input CSV          : {INPUT_CSV.resolve()}")
    print(f"Rows in input      : {len(complete_data)}")
    print(f"Rows to process    : {processing_limit} (maximum {MAX_ROWS})")
    print(f"Text column        : {TEXT_COLUMN}")
    print(f"Ollama model       : {MODEL_NAME}")
    print(f"Output CSV         : {OUTPUT_CSV.resolve()}")
    print(f"Resume row         : {start_row}")
    print("=" * 68)

    write_header = not OUTPUT_CSV.exists()
    for row_number in range(start_row, processing_limit):
        raw_text = data.at[row_number, TEXT_COLUMN]
        text = "" if pd.isna(raw_text) else str(raw_text).strip()
        text = text[:MAX_CHARACTERS]

        themes, success, error_message = extract_themes_with_retry(text)
        if not success:
            print(f"[Row {row_number}] Extraction failed: {error_message}")

        output_row = data.iloc[row_number].to_dict()
        output_row.update(themes)
        output_row["extraction_success"] = int(success)
        output_row["extraction_error"] = error_message

        pd.DataFrame([output_row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8",
        )
        write_header = False

        # Advance only after the completed row has been appended.
        save_checkpoint(CHECKPOINT_JSON, row_number + 1)

        completed = row_number + 1
        if completed % PRINT_EVERY == 0 or completed == processing_limit:
            print(f"Processed {completed}/{processing_limit}")

        if SLEEP_BETWEEN_CALLS_SECONDS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

    print("Completed stress LLM feature extraction for BNN Experiment 02.")
    print(f"Saved: {OUTPUT_CSV.resolve()}")
    print(
        "Review extraction_success before training; rows marked 0 contain "
        "fallback zero theme values after all retries failed."
    )


if __name__ == "__main__":
    main()
