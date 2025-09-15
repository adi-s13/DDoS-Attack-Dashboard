import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# --- Paths ---
TEST_FILE = 'balanced_test_data_with_labels.csv'  # Use the one with labels
PREPROCESSOR_FILE = 'preprocessor.pkl'
MODEL_FILE = 'model.pkl'

# --- Load Preprocessor and Model ---
with open(PREPROCESSOR_FILE, 'rb') as f:
    preprocessor = pickle.load(f)

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

# --- Load Test Data ---
df = pd.read_csv(TEST_FILE)

if 'Label' not in df.columns:
    raise ValueError("❌ The CSV file does not contain the 'Label' column. Cannot calculate accuracy.")

# --- Prepare Features and True Labels ---
X = df[preprocessor['features']]
y_true = preprocessor['label_encoder'].transform(df['Label'])

# --- Scale Features ---
X_scaled = preprocessor['scaler'].transform(X)

# --- Model Prediction ---
y_pred = model.predict(X_scaled)

# --- Accuracy ---
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Model Accuracy on test data: {accuracy * 100:.2f}%")

# --- Get only the classes present in this test data ---
present_labels = np.unique(y_true)
present_target_names = preprocessor['label_encoder'].classes_[present_labels]

# --- Classification Report (only for present classes) ---
print("\n📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=present_labels,
    target_names=present_target_names
))
