import os, json, re, hashlib, time
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_CSV = r"women_clothing_reviews_normalized.csv"   # <-- your normalized women dataset
OUT_DIR = "outputs_exp02_women_clothing_ollama_2000"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
OUT_PATH   = os.path.join(OUT_DIR, "women_clothing_llm_features_2000.csv")
BAD_PATH   = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")
CKPT_PATH  = os.path.join(OUT_DIR, "checkpoint_partial.csv")

OLLAMA_BASE  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

N_ROWS = 2000
RANDOM_STATE = 42
MAX_CHARS = 1000
TEMPERATURE = 0.0
NUM_PREDICT = 220
TIMEOUT = (10, 600)   # connect, read
PRINT_EVERY = 10
SAVE_EVERY = 25       # checkpoint every N rows

THEMES = [
    "fit_sizing",
    "material_quality",
    "comfort",
    "style_appearance",
    "value_price",
    "delivery_packaging",
    "recommendation_intent",
    "return_exchange",
]

SYSTEM_INSTRUCTIONS = f"""
You are a qualitative researcher performing DEDUCTIVE thematic coding using a fixed codebook.

CRITICAL RULES:
- Only code what is explicitly stated. Do NOT guess.
- If a theme is not mentioned: present=0, polarity=0, intensity=0.
- Polarity: -1 negative, 0 neutral/mixed/unclear, +1 positive.
- Intensity: 0 none, 1 mild, 2 moderate, 3 strong.
- Output MUST be valid JSON ONLY. No extra text. No markdown.

THEMES:
{", ".join(THEMES)}

REQUIRED JSON SCHEMA:
{{
  "overall_sentiment": -1|0|1,
  "confidence": 0.0-1.0,
  "themes": {{
    "{THEMES[0]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[1]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[2]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[3]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[4]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[5]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[6]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[7]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}}
  }}
}}
""".strip()

# =========================
# HELPERS
# =========================
def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                obj=json.loads(line)
                cache[obj["key"]] = obj["value"]
    return cache

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_json_block(txt):
    txt = (txt or "").strip()
    if not txt:
        return None
    if txt.startswith("{") and txt.endswith("}"):
        return txt
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return m.group(0) if m else None

def clamp_int(v, lo, hi, default=0):
    try:
        v=int(v); return max(lo, min(hi, v))
    except: return default

def clamp_float(v, lo=0.0, hi=1.0, default=0.5):
    try:
        v=float(v); return max(lo, min(hi, v))
    except: return default

def zero_schema():
    return {
        "overall_sentiment": 0,
        "confidence": 0.0,
        "themes": {t: {"present":0, "polarity":0, "intensity":0} for t in THEMES}
    }

def normalize(parsed):
    out = {
        "overall_sentiment": clamp_int(parsed.get("overall_sentiment",0), -1, 1, 0),
        "confidence": clamp_float(parsed.get("confidence",0.5), 0.0, 1.0, 0.5),
        "themes": {}
    }
    th = parsed.get("themes", {})
    if not isinstance(th, dict):
        th = {}
    for tname in THEMES:
        t = th.get(tname, {})
        if not isinstance(t, dict):
            t = {}
        present = clamp_int(t.get("present",0), 0, 1, 0)
        polarity = clamp_int(t.get("polarity",0), -1, 1, 0)
        intensity = clamp_int(t.get("intensity",0), 0, 3, 0)
        if present == 0:
            polarity = 0
            intensity = 0
        out["themes"][tname] = {"present":present, "polarity":polarity, "intensity":intensity}
    return out

def flatten(codes):
    d = {
        "llm_overall_sentiment": float(codes["overall_sentiment"]),
        "llm_confidence": float(codes["confidence"])
    }
    for theme, v in codes["themes"].items():
        d[f"llm_{theme}_present"] = float(v["present"])
        d[f"llm_{theme}_polarity"] = float(v["polarity"])
        d[f"llm_{theme}_intensity"] = float(v["intensity"])
    return d

def row_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def call_ollama(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content","")

def strict_code(text):
    # if empty text -> return zeros (still counts as a row)
    if not text.strip():
        return zero_schema()

    user_msg = (
        "Return ONLY valid JSON for the schema. No prose, no markdown, no backticks.\n\n"
        f"TEXT:\n{text}"
    )
    raw1 = call_ollama([
        {"role":"system","content":SYSTEM_INSTRUCTIONS},
        {"role":"user","content":user_msg}
    ])
    jb = extract_json_block(raw1)
    if jb:
        try:
            return normalize(json.loads(jb))
        except:
            pass

    # repair pass
    repair_msg = (
        "Fix this into STRICT VALID JSON matching the required schema. Return JSON only.\n\n"
        f"TEXT TO FIX:\n{raw1}"
    )
    raw2 = call_ollama([
        {"role":"system","content":SYSTEM_INSTRUCTIONS},
        {"role":"user","content":repair_msg}
    ])
    jb2 = extract_json_block(raw2)
    if jb2:
        try:
            return normalize(json.loads(jb2))
        except:
            pass

    append_jsonl(BAD_PATH, {"error":"json_failed", "raw1":raw1, "raw2":raw2})
    return zero_schema()

def pick_col(df, options):
    colmap = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in colmap:
            return colmap[opt.lower()]
    return None

def to_text(title, review):
    t = "" if pd.isna(title) else str(title)
    r = "" if pd.isna(review) else str(review)
    s = f"TITLE: {t}\nREVIEW: {r}".strip()
    s = s[:MAX_CHARS] + ("…" if len(s) > MAX_CHARS else "")
    return s

# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(BASE_CSV)

    title_col = pick_col(df, ["Title"])
    review_col = pick_col(df, ["Review Text", "Review", "Text", "Review Text "])  # flexible

    if review_col is None:
        raise ValueError(f"Could not find review text column. Columns: {df.columns.tolist()}")

    # FIXED sample FIRST to guarantee 2000 rows output
    if len(df) < N_ROWS:
        print(f"[WARN] Only {len(df)} rows available, using all.")
        df = df.copy()
    else:
        df = df.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    cache = load_cache(CACHE_PATH)
    print("Rows to code:", len(df))
    print("Cache entries:", len(cache))
    print("Using model:", OLLAMA_MODEL)
    print("Using base :", OLLAMA_BASE)
    print("Title col  :", title_col)
    print("Review col :", review_col)

    rows_out = []

    #for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="LLM thematic coding")):
        #title = getattr(row, title_col) if title_col else ""
        #review = getattr(row, review_col)
    for i in tqdm(range(len(df)), total=len(df), desc="LLM thematic coding"):
        title = df.loc[i, title_col] if title_col else ""
        review = df.loc[i, review_col]

        text_in = to_text(title, review)
        key = row_hash(text_in)

        if key in cache:
            codes = cache[key]
        else:
            # retry a few times manually on timeouts
            for attempt in range(3):
                try:
                    codes = strict_code(text_in)
                    break
                except Exception as e:
                    if attempt == 2:
                        append_jsonl(BAD_PATH, {"error":"request_failed", "exception":str(e), "text":text_in})
                        codes = zero_schema()
                    time.sleep(2)

            append_jsonl(CACHE_PATH, {"key": key, "value": codes})
            cache[key] = codes

        rows_out.append({"row_index": i, **flatten(codes)})

        if (i+1) % PRINT_EVERY == 0:
            print(f"Processed {i+1}/{len(df)}")

        if (i+1) % SAVE_EVERY == 0:
            pd.DataFrame(rows_out).to_csv(CKPT_PATH, index=False)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_PATH, index=False)

    # sanity check
    llm_cols = [c for c in out_df.columns if c.startswith("llm_")]
    nonzero = int((out_df[llm_cols] != 0).sum().sum())
    rows_any = int((out_df[llm_cols].sum(axis=1) != 0).sum())

    print("\nDONE")
    print("Saved:", OUT_PATH)
    print("Sanity check:")
    print("- Nonzero cells:", nonzero)
    print("- Rows with any nonzero:", rows_any)
    if os.path.exists(BAD_PATH):
        print("Debug log:", BAD_PATH)

if __name__ == "__main__":
    main()
