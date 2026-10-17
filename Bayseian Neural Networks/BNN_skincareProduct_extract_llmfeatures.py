
import os
import json
import re
import pandas as pd
import numpy as np
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

OUTPUT_FILENAME = INPUT_FILENAME.replace(
    ".csv", "_llm.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT, "data_subsets", OUTPUT_FILENAME
)

RAW_LOG_PATH = os.path.join(
    ROOT, "data_subsets", "llm_raw_outputs_log.jsonl"
)

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3.1:latest"

# ============================================================
# TEXT COLUMN
# Only review_text will be used for LLM feature extraction
# ============================================================

TEXT_CANDIDATES = [
    "review_text"
]

MAX_CHARS = 1000

# ============================================================
# LLM-DERIVED FEATURES
# ============================================================

FEATURES = [
    "llm_sentiment",
    "llm_product_quality",
    "llm_value_for_money",
    "llm_skin_suitability",
    "llm_ingredient_concern"
]

# ============================================================
# PROMPT
# ============================================================

def prompt_for(text):

    return f"""
You are a skincare product evaluation expert.

Based ONLY on the text below, estimate the following scores:

- llm_sentiment:
  0 = very negative
  1 = very positive

- llm_product_quality:
  0 = very poor quality
  5 = excellent quality

- llm_value_for_money:
  0 = extremely overpriced
  5 = excellent value

- llm_skin_suitability:
  0 = likely harmful or unsuitable
  5 = highly suitable and beneficial

- llm_ingredient_concern:
  0 = no concerning ingredients
  5 = highly concerning ingredients

Return ONLY valid JSON.
No explanation.

TEXT:
{text}
""".strip()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pick_text_column(df):

    # Only the review_text column is permitted
    if "review_text" not in df.columns:

        raise ValueError(
            "Column 'review_text' not found in dataset."
        )

    return "review_text"


def extract_json(text):

    match = re.search(
        r"\{.*\}",
        text.strip(),
        flags=re.S
    )

    if not match:

        raise ValueError(
            "No JSON found in LLM response."
        )

    return json.loads(
        match.group(0)
    )


def coerce_features(d):

    out = {}

    for k in FEATURES:

        v = d.get(k, 0)

        try:
            v = float(v)

        except Exception:
            v = 0.0

        out[k] = v

    # Clip sentiment to 0-1
    out["llm_sentiment"] = float(
        np.clip(
            out["llm_sentiment"],
            0.0,
            1.0
        )
    )

    # Clip remaining features to 0-5
    for k in FEATURES:

        if k != "llm_sentiment":

            out[k] = float(
                np.clip(
                    out[k],
                    0.0,
                    5.0
                )
            )

    return out


def call_ollama(prompt):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json().get(
        "response", ""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded:", df.shape)

    # --------------------------------------------------------
    # SELECT TEXT COLUMN
    # --------------------------------------------------------

    text_col = pick_text_column(df)

    print("Using text column:", text_col)

    # --------------------------------------------------------
    # RESET RAW LOG
    # --------------------------------------------------------

    if os.path.exists(RAW_LOG_PATH):

        os.remove(RAW_LOG_PATH)

    # --------------------------------------------------------
    # INITIALISE FEATURE STORAGE
    # --------------------------------------------------------

    features_dict = {
        k: [] for k in FEATURES
    }

    failures = 0

    # --------------------------------------------------------
    # LLM FEATURE EXTRACTION
    # --------------------------------------------------------

    with open(
        RAW_LOG_PATH,
        "w",
        encoding="utf-8"
    ) as raw_file:

        for i, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            # Use only review_text
            text = str(
                row.get(text_col, "")
            ).replace(
                "\n", " "
            ).strip()

            # Limit text length
            if len(text) > MAX_CHARS:

                text = text[:MAX_CHARS]

            # Generate prompt
            prompt = prompt_for(text)

            try:

                # Call Llama 3.1 through Ollama
                response = call_ollama(prompt)

                # Save raw LLM response
                raw_file.write(
                    json.dumps(
                        {
                            "i": int(i),
                            "response": response
                        },
                        ensure_ascii=False
                    ) + "\n"
                )

                # Extract JSON and convert scores
                parsed = coerce_features(
                    extract_json(response)
                )

                # Store extracted feature values
                for k in FEATURES:

                    features_dict[k].append(
                        parsed[k]
                    )

            except Exception as e:

                failures += 1

                # Log extraction error
                raw_file.write(
                    json.dumps(
                        {
                            "i": int(i),
                            "error": str(e)
                        },
                        ensure_ascii=False
                    ) + "\n"
                )

                # Preserve original failure handling
                for k in FEATURES:

                    features_dict[k].append(
                        0.0
                    )

    # ========================================================
    # ADD LLM FEATURES TO DATASET
    # ========================================================

    for k in FEATURES:

        df[k] = features_dict[k]

    # ========================================================
    # SAVE ENRICHED DATASET
    # ========================================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # EXTRACTION SUMMARY
    # ========================================================

    print("\n==========================================")
    print("Extraction Completed")
    print("Output saved to:", OUTPUT_PATH)
    print(
        "Failures:",
        failures,
        "out of",
        len(df)
    )
    print("==========================================")

    print("\nFeature Summary:")

    print(
        df[FEATURES]
        .describe()
        .T[["min", "max", "mean"]]
    )

    print("\nNon-zero counts:")

    print(
        (df[FEATURES] != 0).sum()
    )


if __name__ == "__main__":
    main()
