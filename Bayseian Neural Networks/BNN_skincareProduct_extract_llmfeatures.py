
"""
Skincare Product Reviews
LLM-Derived Thematic Feature Extraction

resence-polarity-intensity implementation.

Five skincare-related themes are encoded using:
    Presence:  0 = absent, 1 = present
    Polarity: -1 = negative, 0 = neutral/mixed, 1 = positive
    Intensity: 0 = absent, 1 = mild, 2 = moderate, 3 = strong


Five predefined themes generate 15 thematic features

"""

import os
import json
import time
import hashlib
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2"

INPUT_FILENAME = "skincare_2000_seed42.csv"

INPUT_PATH = os.path.join(
    ROOT, "data_subsets", INPUT_FILENAME
)

OUTPUT_DIR = os.path.join(
    ROOT, "data_subsets", "skincare_llm_15features"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "skincare_2000_seed42_llm_15features.csv"
)

CACHE_PATH = os.path.join(
    OUTPUT_DIR, "llm_cache.jsonl"
)

ERROR_PATH = os.path.join(
    OUTPUT_DIR, "extraction_errors.jsonl"
)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR, "checkpoint_partial.csv"
)

TEXT_COLUMN = "review_text"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:latest"

MAX_ROWS = 2000
MAX_CHARS = 1000

TEMPERATURE = 0.0
TIMEOUT = 180
MAX_RETRIES = 3
SAVE_EVERY = 25

# ============================================================
# FIXED DEDUCTIVE THEMATIC CODEBOOK
# ============================================================

THEMES = {
    "sentiment":
        "Overall positive or negative experience with the "
        "skincare product.",

    "product_quality":
        "Perceived product quality, effectiveness, texture, "
        "consistency or performance.",

    "value_for_money":
        "Perceived price fairness, affordability or value "
        "relative to product performance.",

    "skin_suitability":
        "Suitability for the reviewer's skin, including "
        "beneficial effects, irritation or adverse reactions.",

    "ingredient_concern":
        "Explicit discussion of product ingredients, "
        "including concerns, safety or perceived benefits."
}

DIMENSIONS = (
    "present",
    "polarity",
    "intensity"
)

FEATURE_COLUMNS = [
    f"llm_{theme}_{dimension}"
    for theme in THEMES
    for dimension in DIMENSIONS
]

# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a qualitative researcher performing deductive
thematic coding of skincare product reviews.

Use ONLY the fixed thematic codebook below.

For each theme, extract three numerical features.

PRESENCE:
0 = theme absent
1 = theme present

POLARITY:
-1 = negative
 0 = neutral, mixed or unclear
 1 = positive

INTENSITY:
0 = theme absent
1 = mild
2 = moderate
3 = strong

CODING RULES:

1. Use only information supported by the review text.
2. Do not infer product properties that are not mentioned.
3. Evaluate each theme independently.
4. If a theme is absent:
   present = 0
   polarity = 0
   intensity = 0
5. If a theme is present:
   present = 1
   intensity must be 1, 2 or 3.
6. Neutral or mixed sentiment should have polarity = 0.
7. For ingredient_concern, negative polarity indicates
   an unfavourable ingredient-related experience or concern;
   positive polarity indicates a favourable ingredient-related
   experience or assessment.
8. Do not infer ingredient safety from ingredient names alone.
9. Return valid JSON only.
10. Do not include Markdown or explanations.

FIXED CODEBOOK:
{json.dumps(THEMES, indent=2)}

REQUIRED JSON STRUCTURE:

{{
  "themes": {{
    "sentiment": {{
      "present": 0,
      "polarity": 0,
      "intensity": 0
    }},
    "product_quality": {{
      "present": 0,
      "polarity": 0,
      "intensity": 0
    }},
    "value_for_money": {{
      "present": 0,
      "polarity": 0,
      "intensity": 0
    }},
    "skin_suitability": {{
      "present": 0,
      "polarity": 0,
      "intensity": 0
    }},
    "ingredient_concern": {{
      "present": 0,
      "polarity": 0,
      "intensity": 0
    }}
  }}
}}

Return exactly these five themes.
""".strip()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def append_jsonl(path, record):

    with open(path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


def text_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_cache(path):

    cache = {}

    if os.path.exists(path):

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                try:
                    record = json.loads(line)
                    cache[record["key"]] = record["value"]

                except (ValueError, KeyError):
                    continue

    return cache


def zero_schema():

    return {
        "themes": {
            theme: {
                "present": 0,
                "polarity": 0,
                "intensity": 0
            }
            for theme in THEMES
        }
    }


# ============================================================
# VALIDATE EXTRACTED FEATURES
# ============================================================

def validate_output(parsed):

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    if set(parsed.keys()) != {"themes"}:
        raise ValueError(
            "Unexpected top-level JSON keys."
        )

    themes = parsed["themes"]

    if not isinstance(themes, dict):
        raise ValueError(
            "The 'themes' value must be a JSON object."
        )

    if set(themes.keys()) != set(THEMES.keys()):
        raise ValueError(
            "Missing or unexpected theme names."
        )

    validated = {"themes": {}}

    for theme in THEMES:

        values = themes[theme]

        if not isinstance(values, dict):
            raise ValueError(
                f"Invalid theme structure: {theme}"
            )

        required = {
            "present",
            "polarity",
            "intensity"
        }

        if set(values.keys()) != required:
            raise ValueError(
                f"Invalid feature keys for {theme}"
            )

        present = values["present"]
        polarity = values["polarity"]
        intensity = values["intensity"]

        for value in (
            present,
            polarity,
            intensity
        ):
            if type(value) is not int:
                raise ValueError(
                    f"Non-integer value for {theme}"
                )

        if present not in (0, 1):
            raise ValueError(
                f"Invalid presence for {theme}"
            )

        if polarity not in (-1, 0, 1):
            raise ValueError(
                f"Invalid polarity for {theme}"
            )

        if intensity not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid intensity for {theme}"
            )

        if present == 0:

            if polarity != 0 or intensity != 0:
                raise ValueError(
                    f"Inconsistent absent theme: {theme}"
                )

        if present == 1 and intensity == 0:
            raise ValueError(
                f"Present theme has zero intensity: {theme}"
            )

        validated["themes"][theme] = {
            "present": present,
            "polarity": polarity,
            "intensity": intensity
        }

    return validated


# ============================================================
# FLATTEN FEATURES INTO DATASET COLUMNS
# ============================================================

def flatten_features(codes):

    features = {}

    for theme, values in codes["themes"].items():

        features[
            f"llm_{theme}_present"
        ] = values["present"]

        features[
            f"llm_{theme}_polarity"
        ] = values["polarity"]

        features[
            f"llm_{theme}_intensity"
        ] = values["intensity"]

    return features


# ============================================================
# OLLAMA REQUEST
# ============================================================

def call_ollama(text):

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"REVIEW TEXT:\n{text}"
            }
        ],
        "options": {
            "temperature": TEMPERATURE
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    content = result["message"]["content"]

    return json.loads(content)


# ============================================================
# EXTRACT FEATURES WITH RETRY
# ============================================================

def extract_features(text):

    if not text.strip():
        return zero_schema(), True, "empty_text"

    last_error = ""

    for attempt in range(MAX_RETRIES):

        try:

            parsed = call_ollama(text)

            validated = validate_output(parsed)

            return validated, True, ""

        except Exception as error:

            last_error = (
                f"{type(error).__name__}: {error}"
            )

            if attempt < MAX_RETRIES - 1:

                time.sleep(
                    2 * (attempt + 1)
                )

    return zero_schema(), False, last_error


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD INPUT DATASET
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    # --------------------------------------------------------
    # VERIFY TEXT COLUMN
    # --------------------------------------------------------

    if TEXT_COLUMN not in df.columns:

        raise ValueError(
            f"Required text column '{TEXT_COLUMN}' "
            "not found in dataset."
        )

    # --------------------------------------------------------
    # SELECT ROWS
    # --------------------------------------------------------

    df = df.head(
        MAX_ROWS
    ).copy().reset_index(drop=True)

    cache = load_cache(CACHE_PATH)

    # --------------------------------------------------------
    # RESUME FROM CHECKPOINT
    # --------------------------------------------------------

    if os.path.exists(CHECKPOINT_PATH):

        previous = pd.read_csv(
            CHECKPOINT_PATH
        )

        if len(previous) > len(df):
            raise ValueError(
                "Checkpoint contains too many rows."
            )

        if len(previous) > 0:

            expected = list(
                range(len(previous))
            )

            actual = previous[
                "row_index"
            ].tolist()

            if actual != expected:
                raise ValueError(
                    "Checkpoint row indices do not match."
                )

        results = previous.to_dict("records")

        start = len(results)

    else:

        results = []

        start = 0

    # --------------------------------------------------------
    # EXTRACTION SUMMARY
    # --------------------------------------------------------

    print("=" * 65)
    print("SKINCARE LLM FEATURE EXTRACTION")
    print("=" * 65)

    print("Input:", INPUT_PATH)
    print("Text column:", TEXT_COLUMN)
    print("LLM:", OLLAMA_MODEL)
    print("Themes:", len(THEMES))
    print("Features:", len(FEATURE_COLUMNS))
    print("Rows:", len(df))
    print("Starting row:", start)

    # --------------------------------------------------------
    # PROCESS EACH REVIEW
    # --------------------------------------------------------

    for i in tqdm(
        range(start, len(df)),
        initial=start,
        total=len(df),
        desc="LLM thematic coding"
    ):

        raw_text = df.at[
            i,
            TEXT_COLUMN
        ]

        text = (
            ""
            if pd.isna(raw_text)
            else str(raw_text).strip()
        )

        text = text[:MAX_CHARS]

        key = text_hash(text)

        if key in cache:

            codes = cache[key]

            success = True

            error = ""

        else:

            codes, success, error = extract_features(text)

            if success:

                append_jsonl(
                    CACHE_PATH,
                    {
                        "key": key,
                        "value": codes
                    }
                )

                cache[key] = codes

        # ----------------------------------------------------
        # LOG FAILURES
        # ----------------------------------------------------

        if not success:

            append_jsonl(
                ERROR_PATH,
                {
                    "row_index": i,
                    "error": error
                }
            )

        # ----------------------------------------------------
        # STORE FEATURES
        # ----------------------------------------------------

        features = flatten_features(codes)

        results.append({
            "row_index": i,
            **features,
            "extraction_success": int(success),
            "extraction_error": error
        })

        # ----------------------------------------------------
        # SAVE CHECKPOINT
        # ----------------------------------------------------

        if (i + 1) % SAVE_EVERY == 0:

            pd.DataFrame(results).to_csv(
                CHECKPOINT_PATH,
                index=False
            )

    # ========================================================
    # CREATE ENRICHED DATASET
    # ========================================================

    features_df = pd.DataFrame(results)

    # Exclude row_index from duplicated columns
    feature_values = features_df[
        FEATURE_COLUMNS + [
            "extraction_success",
            "extraction_error"
        ]
    ].copy()

    enriched_df = pd.concat(
        [
            df.reset_index(drop=True),
            feature_values.reset_index(drop=True)
        ],
        axis=1
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    enriched_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    features_df.to_csv(
        CHECKPOINT_PATH,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    failures = int(
        (features_df["extraction_success"] == 0).sum()
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETED")
    print("=" * 65)

    print("Output:", OUTPUT_PATH)
    print("Rows:", len(enriched_df))
    print("Themes:", len(THEMES))
    print("LLM-derived features:", len(FEATURE_COLUMNS))
    print("Failed extractions:", failures)

    print("\nExtracted feature columns:")

    for feature in FEATURE_COLUMNS:
        print(feature)


if __name__ == "__main__":
    main()
