import pickle
import pandas as pd

# Load preprocessor & model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your generated test CSV
df = pd.read_csv('balanced_test_data.csv')

# Preview the shape and contents
print("CSV shape:", df.shape)
print(df.head())

# Apply same preprocessing
X = df[preprocessor['features']]
X_scaled = preprocessor['scaler'].transform(X)

# Predict
preds = model.predict(X_scaled)

print("Unique predictions:", set(preds))
print("Prediction counts:", pd.Series(preds).value_counts())
