import os
import pandas as pd
import pickle

# --- Paths ---
DATA_DIR = os.path.join("data", "cicddos2019")  # Adjust if needed
OUTPUT_FILE_DASHBOARD = "balanced_test_data.csv"
OUTPUT_FILE_LABELLED  = "balanced_test_data_with_labels.csv"
PREPROCESSOR_FILE = "preprocessor.pkl"

# --- Parameters ---
NUM_BENIGN = 10
NUM_ATTACK = 10

# --- Load preprocessor to get feature columns ---
with open(PREPROCESSOR_FILE, "rb") as f:
    preprocessor = pickle.load(f)
feature_cols = preprocessor["features"]

# --- Load all parquet files ---
parquet_files = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".parquet"):
            parquet_files.append(os.path.join(root, file))

if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

dfs = [pd.read_parquet(pf) for pf in parquet_files]
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    raise KeyError("'Label' column not found in dataset.")

# --- Separate benign and attack ---
benign_df = df[df["Label"] == "Benign"]
attack_df = df[df["Label"] != "Benign"]

# --- Sample ---
benign_sample = benign_df.sample(min(NUM_BENIGN, len(benign_df)), random_state=42)
attack_sample = attack_df.sample(min(NUM_ATTACK, len(attack_df)), random_state=42)

# --- Combine and shuffle ---
sample_df = pd.concat([benign_sample, attack_sample]).sample(frac=1, random_state=42)

# --- Debug print to confirm attack labels sampled ---
print("[DEBUG] Sample label counts:")
print(sample_df["Label"].value_counts())

# --- Save version WITH labels (for accuracy evaluation) ---
sample_df.to_csv(OUTPUT_FILE_LABELLED, index=False)
print(f"[INFO] Labeled balanced sample saved to {OUTPUT_FILE_LABELLED}")

# --- Save version WITHOUT labels (for dashboard input) ---
sample_df[feature_cols].to_csv(OUTPUT_FILE_DASHBOARD, index=False)
print(f"[INFO] Feature-only balanced sample saved to {OUTPUT_FILE_DASHBOARD}")

print(f"[INFO] Rows: {len(sample_df)} | Benign: {len(benign_sample)} | Attack: {len(attack_sample)}")
print("[INFO] Use the LABELLED file for accuracy testing, and the other for your dashboard.")
