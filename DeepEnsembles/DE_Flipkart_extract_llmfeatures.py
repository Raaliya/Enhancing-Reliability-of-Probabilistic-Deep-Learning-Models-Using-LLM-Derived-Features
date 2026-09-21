import os
import json
import time
import argparse
import pandas as pd
import requests

MODEL_NAME = "qwen2.5:7b-instruct"

CODEBOOK = {
    "T1_PRODUCT_QUALITY": "Mentions product quality, durability, performance.",
    "T2_VALUE_FOR_MONEY": "Mentions price fairness or value.",
    "T3_DELIVERY_EXPERIENCE": "Mentions delivery speed, packaging, logistics.",
    "T4_CUSTOMER_SERVICE": "Mentions seller or customer support interaction.",
    "T5_DEFECT_DAMAGE": "Mentions defective, damaged, broken product.",
    "T6_RETURN_REFUND": "Mentions return or refund issues.",
    "T7_RECOMMENDATION": "Explicit recommendation or not recommending.",
    "T8_EMOTIONAL_INTENSITY": "Strong emotional tone (very positive or negative)."
}

def build_prompt(review_text):
    return f"""
You are a qualitative research assistant performing deductive thematic coding.

Use the fixed codebook below.
For each theme return:
0 = Absent
1 = Present

Return ONLY valid JSON with these exact keys.

Codebook:
{json.dumps(CODEBOOK, indent=2)}

Review:
{review_text}
"""

def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]

def parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end+1])
    raise ValueError("Invalid JSON output")

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["next_row"]
    return 0

def save_checkpoint(path, row):
    with open(path, "w") as f:
        json.dump({"next_row": row}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--checkpoint", default="flipkart_thematic_checkpoint.json")
    parser.add_argument("--text_col", default="review_text")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    start_row = load_checkpoint(args.checkpoint)

    print(f"Resuming from row: {start_row}")

    write_header = not os.path.exists(args.output_csv)

    for i in range(start_row, len(df)):
        text = str(df.loc[i, args.text_col])[:1000]

        prompt = build_prompt(text)

        try:
            response = call_ollama(prompt)
            themes = parse_json(response)

            # Ensure all keys present
            for k in CODEBOOK:
                if k not in themes:
                    themes[k] = 0

        except Exception as e:
            print(f"Row {i} failed: {e}")
            themes = {k: 0 for k in CODEBOOK}

        row_data = df.loc[i].to_dict()
        row_data.update(themes)

        pd.DataFrame([row_data]).to_csv(
            args.output_csv,
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False

        save_checkpoint(args.checkpoint, i + 1)

        if i % 20 == 0:
            print(f"Processed {i}/{len(df)}")

    print("Thematic coding completed.")

if __name__ == "__main__":
    main()
    
