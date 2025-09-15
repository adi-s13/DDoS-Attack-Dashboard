import zipfile
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "archive.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "cicddos2019")

# 1. Unzip dataset if not already extracted
if not os.path.exists(EXTRACT_PATH):
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("[INFO] Extraction complete.")
else:
    print("[INFO] Dataset already extracted.")

# 2. Load all Parquet files
print("[INFO] Loading Parquet files...")
df_list = []
for root, _, files in os.walk(EXTRACT_PATH):
    for file in files:
        if file.endswith(".parquet"):
            file_path = os.path.join(root, file)
            df_list.append(pd.read_parquet(file_path))

if not df_list:
    raise FileNotFoundError("[ERROR] No parquet files found in data folder.")

df = pd.concat(df_list, ignore_index=True)
print(f"[INFO] Dataset shape before cleaning: {df.shape}")

# 3. Clean column names
df.columns = df.columns.str.strip()    # Remove spaces from all column names

# 4. Handle missing values
df = df.dropna()
print(f"[INFO] Dataset shape after dropping NaNs: {df.shape}")

# 5. Encode labels
if 'Label' not in df.columns:
    raise KeyError("[ERROR] 'Label' column not found in dataset.")

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
print(f"[INFO] Label classes: {list(label_encoder.classes_)}")

# 6. Separate features & labels
X = df.drop(columns=['Label'])
y = df['Label']

# 7. Keep only numeric columns for model input
X_numeric = X.select_dtypes(include=[np.number])
print(f"[INFO] Number of features used for scaling: {X_numeric.shape[1]}")

# 8. Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 9. Save preprocessor
preprocessor = {
    'scaler': scaler,
    'label_encoder': label_encoder,
    'features': X_numeric.columns.tolist(),
    'label_classes': list(label_encoder.classes_)
}

PICKLE_PATH = os.path.join(PROJECT_DIR, "preprocessor.pkl")
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(preprocessor, f)

print(f"[INFO] Preprocessor saved to {PICKLE_PATH}")
print("[INFO] Feature columns saved in preprocessor:")
for feat in preprocessor['features']:
    print(f"  - {feat}")
