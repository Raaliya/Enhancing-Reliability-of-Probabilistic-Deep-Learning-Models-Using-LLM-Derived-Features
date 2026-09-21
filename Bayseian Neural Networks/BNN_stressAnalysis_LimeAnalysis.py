import os
import html
import webbrowser
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
WORK_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime"

DATA_PATH = os.path.join(WORK_DIR, "stress_llm_features_2000.csv")

MODEL_PATH = os.path.join(WORK_DIR, "bnn_stress_original_plus_llm_model.pt")
SCALER_PATH = os.path.join(WORK_DIR, "bnn_stress_original_plus_llm_scaler.joblib")

OUTPUT_HTML = os.path.join(
    WORK_DIR,
    "bnn_stress_lime_dashboard_top15_llm_original_features.html"
)

# =========================================================
# SETTINGS
# =========================================================
TARGET_COL = "confidence"
INSTANCE_INDEX = 10
MC_SAMPLES = 100
MAX_ROWS_FOR_DASHBOARD = 500

TOP_N_LLM_FEATURES = 15

# =========================================================
# ORIGINAL STRESS DATASET FEATURES TO ALWAYS DISPLAY
# =========================================================
ORIGINAL_FEATURES_REQUIRED = [
    "lex_liwc_anger",
    "lex_liwc_negative emo",
    "lex_liwc_positivemo",
    "lex_liwc_negate",
    "lex_liwc_Tone",
    "lex_liwc_Authentic",
]

# =========================================================
# COLUMNS TO DROP FROM DISPLAY
# =========================================================
DROP_FROM_DISPLAY = [
    "source_index",
    "text",
    "overall_sentiment",
    "confidence",
    "parse_error",
    "runtime_error",
]

ALWAYS_EXCLUDE = DROP_FROM_DISPLAY + [TARGET_COL]

# =========================================================
# CHECK FILE PATHS
# =========================================================
if not os.path.exists(DATA_PATH):
    alt_path = os.path.join(WORK_DIR, "stress_llm_features_2000")
    if os.path.exists(alt_path):
        DATA_PATH = alt_path
    else:
        raise FileNotFoundError(f"Stress dataset not found:\n{DATA_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Stress model not found:\n{MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Stress scaler not found:\n{SCALER_PATH}")

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset:", DATA_PATH)
df_original = pd.read_csv(DATA_PATH, low_memory=False)
print("Original dataset shape:", df_original.shape)

if MAX_ROWS_FOR_DASHBOARD is not None:
    df_original = df_original.head(MAX_ROWS_FOR_DASHBOARD).copy()
    print(f"Using first {MAX_ROWS_FOR_DASHBOARD} rows for dashboard background.")

if TARGET_COL not in df_original.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found in dataset. "
        "For the Stress Analysis dataset, the target is expected to be 'confidence'."
    )

# =========================================================
# LOAD SCALER AND GET MODEL FEATURE ORDER
# =========================================================
scaler = joblib.load(SCALER_PATH)

if hasattr(scaler, "feature_names_in_"):
    model_feature_names = list(scaler.feature_names_in_)
else:
    raise ValueError(
        "The scaler does not contain feature_names_in_. "
        "Please confirm that scaler_stress.joblib is the correct Stress scaler."
    )

print("\nScaler/model feature count:", len(model_feature_names))

# =========================================================
# PREPARE TARGET
# =========================================================
df = df_original.copy()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()

y = df[TARGET_COL].copy()

if INSTANCE_INDEX >= len(df):
    raise ValueError(
        f"INSTANCE_INDEX={INSTANCE_INDEX} is out of range. "
        f"Dataset contains only {len(df)} rows after cleaning."
    )

# =========================================================
# CHECK ORIGINAL FEATURES
# =========================================================
missing_from_dataset = [
    c for c in ORIGINAL_FEATURES_REQUIRED
    if c not in df.columns
]

if missing_from_dataset:
    raise ValueError(
        "The following original LIWC features are missing from the dataset:\n"
        f"{missing_from_dataset}\n\n"
        "Please check the exact column names in stress_llm_features_2000.csv."
    )

missing_from_model = [
    c for c in ORIGINAL_FEATURES_REQUIRED
    if c not in model_feature_names
]

if missing_from_model:
    raise ValueError(
        "The following original LIWC features exist in the dataset but are NOT "
        "included in the trained Stress model/scaler:\n"
        f"{missing_from_model}\n\n"
        "LIME can only explain features used by the trained model. "
        "Please retrain the Stress BNN model using these LIWC features together "
        "with the LLM features, then rerun this script."
    )

# =========================================================
# PREPARE MODEL INPUT FEATURES
# =========================================================
missing_model_features = [
    c for c in model_feature_names
    if c not in df.columns
]

if missing_model_features:
    raise ValueError(
        "The following features required by the Stress scaler/model are missing "
        f"from the dataset:\n{missing_model_features}"
    )

X_model_raw = df[model_feature_names].copy()

for col in X_model_raw.columns:
    X_model_raw[col] = pd.to_numeric(X_model_raw[col], errors="coerce")

for col in X_model_raw.columns:
    if X_model_raw[col].isna().sum() > 0:
        median_value = X_model_raw[col].median()

        if pd.isna(median_value):
            X_model_raw[col] = X_model_raw[col].fillna(0)
        else:
            X_model_raw[col] = X_model_raw[col].fillna(median_value)

X_model_scaled = scaler.transform(X_model_raw)

# =========================================================
# LOAD MODEL
# =========================================================
state_dict = torch.load(MODEL_PATH, map_location=device)

if "net.0.weight" not in state_dict:
    raise ValueError(
        "The model architecture does not match the expected Sequential BNN format. "
        "Expected key 'net.0.weight' was not found."
    )

input_dim = state_dict["net.0.weight"].shape[1]
hidden_1 = state_dict["net.0.weight"].shape[0]
hidden_2 = state_dict["net.3.weight"].shape[0]

if input_dim != len(model_feature_names):
    raise ValueError(
        f"Model input dimension is {input_dim}, but scaler expects "
        f"{len(model_feature_names)} features."
    )

class BNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_2, 1)
        )

    def forward(self, x):
        return self.net(x)

model = BNNRegressor(
    input_dim=input_dim,
    hidden_1=hidden_1,
    hidden_2=hidden_2
).to(device)

model.load_state_dict(state_dict)
model.eval()

print("\nStress BNN model loaded successfully.")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def is_llm_feature(feature_name: str) -> bool:
    """
    Detects LLM-derived Stress thematic features.
    Excludes original LIWC features and dropped columns.
    """
    if feature_name in ORIGINAL_FEATURES_REQUIRED:
        return False

    if feature_name in DROP_FROM_DISPLAY:
        return False

    if feature_name.startswith("lex_liwc"):
        return False

    if feature_name.startswith("llm_"):
        return True

    if (
        feature_name.endswith("_present")
        or feature_name.endswith("_polarity")
        or feature_name.endswith("_intensity")
    ):
        return True

    return False

def extract_base_feature(rule: str, columns: list[str]) -> str:
    sorted_cols = sorted(columns, key=len, reverse=True)
    rule_lower = rule.lower()

    for col in sorted_cols:
        if col.lower() in rule_lower:
            return col

    return rule

def format_value(value):
    if isinstance(value, (np.floating, float)):
        return f"{value:.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)

def build_probability_bar(label: str, prob: float, color: str) -> str:
    return f"""
    <div class="prob-row">
        <div class="prob-label">{html.escape(label)}</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{prob * 100:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{prob:.2f}</div>
    </div>
    """

# =========================================================
# PREDICTION FUNCTION FOR UNCERTAINTY SUMMARY
# =========================================================
def mc_predict_single(full_scaled_row: np.ndarray, n_samples: int = 100) -> np.ndarray:
    x_tensor = torch.tensor(
        full_scaled_row.reshape(1, -1),
        dtype=torch.float32
    ).to(device)

    preds = []

    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x_tensor).cpu().numpy().flatten()[0]
            preds.append(pred)

    model.eval()
    return np.array(preds, dtype=float)

# =========================================================
# STEP 1: SELECT BEST 15 LLM FEATURES USING FULL LOCAL LIME
# =========================================================
all_display_features = [
    c for c in model_feature_names
    if c not in ALWAYS_EXCLUDE
]

llm_candidate_features = [
    c for c in all_display_features
    if is_llm_feature(c)
]

if len(llm_candidate_features) == 0:
    raise ValueError(
        "No LLM-derived Stress features were detected. "
        "Please check the Stress LLM column names."
    )

full_selection_features = ORIGINAL_FEATURES_REQUIRED + llm_candidate_features

full_selection_features = [
    c for c in full_selection_features
    if c in model_feature_names
]

full_selection_positions = [
    model_feature_names.index(c) for c in full_selection_features
]

base_instance_full_raw = X_model_raw.iloc[INSTANCE_INDEX].values.astype(float)

def predict_fn_full_selection(display_raw_array: np.ndarray) -> np.ndarray:
    display_raw_array = np.asarray(display_raw_array, dtype=float)

    if display_raw_array.ndim == 1:
        display_raw_array = display_raw_array.reshape(1, -1)

    full_raw = np.tile(base_instance_full_raw, (display_raw_array.shape[0], 1))

    for i, pos in enumerate(full_selection_positions):
        full_raw[:, pos] = display_raw_array[:, i]

    full_raw_df = pd.DataFrame(full_raw, columns=model_feature_names)
    full_scaled = scaler.transform(full_raw_df)

    x_tensor = torch.tensor(full_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy().flatten()

    return preds

X_lime_full_raw = X_model_raw[full_selection_features].copy()

full_explainer = LimeTabularExplainer(
    training_data=X_lime_full_raw.values,
    feature_names=full_selection_features,
    mode="regression",
    discretize_continuous=True,
    random_state=42
)

print("\nGenerating full LIME explanation to select top 15 LLM features...")

full_exp = full_explainer.explain_instance(
    data_row=X_lime_full_raw.iloc[INSTANCE_INDEX].values,
    predict_fn=predict_fn_full_selection,
    num_features=len(full_selection_features)
)

full_lime_rules = full_exp.as_list()

llm_contributions = {}

for rule, contribution in full_lime_rules:
    base_feature = extract_base_feature(rule, full_selection_features)

    if base_feature in llm_candidate_features:
        llm_contributions[base_feature] = abs(float(contribution))

top_llm_features = sorted(
    llm_contributions,
    key=llm_contributions.get,
    reverse=True
)[:TOP_N_LLM_FEATURES]

if len(top_llm_features) < TOP_N_LLM_FEATURES:
    remaining = [
        c for c in llm_candidate_features
        if c not in top_llm_features
    ]

    top_llm_features.extend(
        remaining[:TOP_N_LLM_FEATURES - len(top_llm_features)]
    )

compact_display_features = []

for c in ORIGINAL_FEATURES_REQUIRED + top_llm_features:
    if c in model_feature_names and c not in compact_display_features:
        compact_display_features.append(c)

NUM_FEATURES_FINAL = len(compact_display_features)

print("\nOriginal Stress LIWC features included:")
for c in ORIGINAL_FEATURES_REQUIRED:
    print(" -", c)

print(f"\nTop {TOP_N_LLM_FEATURES} LLM features selected:")
for c in top_llm_features:
    print(" -", c)

print("\nFinal compact display feature count:", NUM_FEATURES_FINAL)

# =========================================================
# STEP 2: FINAL COMPACT LIME EXPLANATION
# =========================================================
compact_display_positions = [
    model_feature_names.index(c) for c in compact_display_features
]

def predict_fn_compact(display_raw_array: np.ndarray) -> np.ndarray:
    display_raw_array = np.asarray(display_raw_array, dtype=float)

    if display_raw_array.ndim == 1:
        display_raw_array = display_raw_array.reshape(1, -1)

    full_raw = np.tile(base_instance_full_raw, (display_raw_array.shape[0], 1))

    for i, pos in enumerate(compact_display_positions):
        full_raw[:, pos] = display_raw_array[:, i]

    full_raw_df = pd.DataFrame(full_raw, columns=model_feature_names)
    full_scaled = scaler.transform(full_raw_df)

    x_tensor = torch.tensor(full_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy().flatten()

    return preds

X_lime_compact_raw = X_model_raw[compact_display_features].copy()
instance_raw_display = X_lime_compact_raw.iloc[INSTANCE_INDEX].copy()

compact_explainer = LimeTabularExplainer(
    training_data=X_lime_compact_raw.values,
    feature_names=compact_display_features,
    mode="regression",
    discretize_continuous=True,
    random_state=42
)

print("\nGenerating compact Stress LIME explanation...")

exp = compact_explainer.explain_instance(
    data_row=X_lime_compact_raw.iloc[INSTANCE_INDEX].values,
    predict_fn=predict_fn_compact,
    num_features=NUM_FEATURES_FINAL
)

# =========================================================
# PREDICTION SUMMARY
# =========================================================
mc_preds = mc_predict_single(
    X_model_scaled[INSTANCE_INDEX],
    n_samples=MC_SAMPLES
)

pred_mean = float(np.mean(mc_preds))
pred_std = float(np.std(mc_preds, ddof=1)) if len(mc_preds) > 1 else 0.0
pred_low = float(np.percentile(mc_preds, 2.5))
pred_high = float(np.percentile(mc_preds, 97.5))
actual_value = float(y.iloc[INSTANCE_INDEX])

p_lt_030 = float(np.mean(mc_preds < 0.30))
p_030_to_060 = float(np.mean((mc_preds >= 0.30) & (mc_preds < 0.60)))
p_ge_060 = float(np.mean(mc_preds >= 0.60))

# =========================================================
# FEATURE-VALUE TABLE
# =========================================================
lime_rules = exp.as_list()

table_rows = []
used_features = set()

for rule, contribution in lime_rules:
    base_feature = extract_base_feature(rule, compact_display_features)

    if base_feature in instance_raw_display.index and base_feature not in used_features:
        raw_value = instance_raw_display[base_feature]

        table_rows.append({
            "feature": base_feature,
            "value": format_value(raw_value),
            "contribution": float(contribution)
        })

        used_features.add(base_feature)

if not table_rows:
    for feat in compact_display_features:
        table_rows.append({
            "feature": feat,
            "value": format_value(instance_raw_display[feat]),
            "contribution": 0.0
        })

# =========================================================
# HTML PANELS
# =========================================================
summary_html = f"""
<div class="card">
    <h2>Prediction Summary</h2>

    <div class="summary-grid">
        <div><span class="k">Predicted value</span><span class="v">{pred_mean:.3f}</span></div>
        <div><span class="k">Actual value</span><span class="v">{actual_value:.3f}</span></div>
        <div><span class="k">Predictive std</span><span class="v">{pred_std:.3f}</span></div>
        <div><span class="k">95% interval</span><span class="v">[{pred_low:.3f}, {pred_high:.3f}]</span></div>
    </div>

    <h3>Prediction distribution by confidence band</h3>

    {build_probability_bar("Confidence < 0.30", p_lt_030, "#1f77b4")}
    {build_probability_bar("0.30 ≤ Confidence < 0.60", p_030_to_060, "#ffbf00")}
    {build_probability_bar("Confidence ≥ 0.60", p_ge_060, "#ff7f0e")}
</div>
"""

lime_html_raw = exp.as_html()

lime_panel = f"""
<div class="card">
    <h2>LIME Explanation</h2>
    <p class="note-top">
        Displaying original Stress LIWC features and the top {TOP_N_LLM_FEATURES} LLM features by local LIME contribution.
    </p>
    <iframe class="lime-frame" srcdoc="{html.escape(lime_html_raw)}"></iframe>
</div>
"""

table_html_rows = ""

for row in table_rows:
    sign_class = "pos" if row["contribution"] >= 0 else "neg"

    table_html_rows += f"""
    <tr>
        <td>{html.escape(row["feature"])}</td>
        <td>{html.escape(row["value"])}</td>
        <td class="{sign_class}">{row["contribution"]:+.3f}</td>
    </tr>
    """

feature_table_html = f"""
<div class="card">
    <h2>Feature-Value Table</h2>

    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>LIME<br>Contribution</th>
            </tr>
        </thead>
        <tbody>
            {table_html_rows}
        </tbody>
    </table>

    <p class="note">
        The table shows raw feature values for the selected instance.
    </p>
</div>
"""

# =========================================================
# FULL HTML DASHBOARD
# =========================================================
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BNN Stress Analysis LIME Dashboard</title>

<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 0;
        padding: 8px;
        background: #f7f7f7;
        color: #222;
    }}

    h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
        font-weight: 800;
    }}

    .subtitle {{
        margin-bottom: 18px;
        color: #555;
        font-size: 14px;
    }}

    .layout {{
        display: grid;
        grid-template-columns: 1.05fr 1.45fr 1.05fr;
        gap: 14px;
        align-items: start;
    }}

    .card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        padding: 18px;
    }}

    .card h2 {{
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 20px;
        font-weight: 800;
    }}

    .card h3 {{
        margin-top: 18px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 800;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
    }}

    .summary-grid div {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 10px 12px;
        background: #fafafa;
        border-radius: 8px;
    }}

    .k {{
        color: #555;
        font-weight: 600;
    }}

    .v {{
        color: #111;
        font-weight: 700;
    }}

    .prob-row {{
        display: grid;
        grid-template-columns: 155px 1fr 38px;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
    }}

    .prob-label {{
        font-size: 14px;
    }}

    .prob-bar-wrap {{
        height: 18px;
        background: #e9e9e9;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #ddd;
    }}

    .prob-bar {{
        height: 100%;
        border-radius: 10px;
    }}

    .prob-value {{
        text-align: right;
        font-weight: 700;
        font-size: 14px;
    }}

    .lime-frame {{
        width: 100%;
        height: 720px;
        border: none;
        border-radius: 8px;
        background: white;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    th, td {{
        border-bottom: 1px solid #e6e6e6;
        padding: 8px 7px;
        text-align: left;
        vertical-align: top;
    }}

    th {{
        background: #fafafa;
        font-weight: 800;
    }}

    td:nth-child(1) {{
        word-break: break-word;
    }}

    td:nth-child(2) {{
        text-align: right;
        white-space: nowrap;
    }}

    td:nth-child(3) {{
        text-align: left;
        white-space: nowrap;
    }}

    .pos {{
        color: #ff7f0e;
        font-weight: 800;
    }}

    .neg {{
        color: #1f77b4;
        font-weight: 800;
    }}

    .note {{
        margin-top: 14px;
        color: #666;
        font-size: 12px;
    }}

    .note-top {{
        margin-top: -4px;
        margin-bottom: 10px;
        color: #666;
        font-size: 12px;
    }}

    @media screen and (max-width: 1200px) {{
        .layout {{
            grid-template-columns: 1fr;
        }}

        .lime-frame {{
            height: 720px;
        }}
    }}
</style>
</head>

<body>
    <h1>BNN Stress Analysis – Single-Instance Explanation Dashboard</h1>

    <div class="subtitle">
        Instance index: {INSTANCE_INDEX} &nbsp;|&nbsp;
        Predicted mean: {pred_mean:.3f} &nbsp;|&nbsp;
        Actual: {actual_value:.3f} &nbsp;|&nbsp;
        Displayed features: {NUM_FEATURES_FINAL}
    </div>

    <div class="layout">
        {summary_html}
        {lime_panel}
        {feature_table_html}
    </div>
</body>
</html>
"""

# =========================================================
# SAVE HTML
# =========================================================
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(full_html)

print("\nStress compact LIME dashboard saved to:")
print(OUTPUT_HTML)

webbrowser.open("file:///" + OUTPUT_HTML.replace("\\", "/"))

