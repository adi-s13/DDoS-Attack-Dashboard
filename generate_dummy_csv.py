import pickle
import pandas as pd
import numpy as np

# Load feature names from preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

features = preprocessor['features']

# Generate dummy data (10 rows)
rows = 10
data = {}
for feat in features:
    data[feat] = np.random.randint(1, 1000, size=rows)  # random numbers

df = pd.DataFrame(data)

# Save CSV
df.to_csv('dummy_test_data.csv', index=False)
print("✅ Dummy test data saved as dummy_test_data.csv")
