DDoS Attack Detection Dashboard
A Streamlit-based dashboard for detecting DDoS attacks in network traffic using a pre-trained scikit-learn model and a saved preprocessing pipeline. The app validates required columns, scales inputs consistently, predicts attack types, and visualizes counts and breakdowns.
Features
Interactive Streamlit UI to upload CSVs of network flow features and see predictions.
Consistent preprocessing with a stored scaler, label encoder, and ordered feature list.
Normal vs attack counters and per-attack-type bar chart.
Helper scripts to generate balanced samples from CIC-DDoS2019, evaluate accuracy, and sanity-check predictions.
Repository structure
app/ddos_dashboard.py — Main Streamlit app.
scripts/train_model.py — Demo inference UI that loads model + preprocessor.
scripts/evaluate_model_accuracy.py — Accuracy and classification report on labeled CSV.
scripts/balanced_sample_generator.py — Build balanced test CSVs from CIC-DDoS2019 parquet files.
scripts/test_model_predictions.py — Sanity-check predictions on feature-only CSVs.
scripts/print_required_features.py — Print the exact feature list expected by the preprocessor.
scripts/generate_dummy_csv.py — Generate tiny synthetic CSV for shape-only checks.
models/ — Place model.pkl and preprocessor.pkl here (ignored by Git).
data/ — Place CIC-DDoS2019 locally (ignored by Git).
Requirements
Python 3.9+ and packages: streamlit, scikit-learn, pandas, numpy.
Install minimal set:
pip install streamlit scikit-learn pandas numpy
Model and preprocessor
Place these files in models/ (not committed):
model.pkl — trained classifier used for inference.
preprocessor.pkl — dict with keys:
scaler — fitted StandardScaler or similar.
label_encoder — fitted LabelEncoder for decoding class IDs.
features — list of feature column names in required order.
To print the required features list:
python scripts/print_required_features.py
Input data format
The uploaded CSV must include all columns listed in preprocessor['features'], in that order. The app checks for missing columns and stops with an error if any are missing.
Example feature names come from CIC-DDoS2019 (Protocol, Flow Duration, Total Fwd Packets, packet length stats, IAT stats, flags, etc.). Use the scripts to generate valid samples.
Run the app
streamlit run app/ddos_dashboard.py
Then upload a CSV with required features. The app will:
Validate required columns.
Scale features with the stored scaler.
Predict labels and display decoded class names.
Show counts and a bar chart of detected attack types.
Generate samples (optional)
Requires CIC-DDoS2019 parquet files under data/cicddos2019 locally. Then run:
python scripts/balanced_sample_generator.py
Outputs:
balanced_test_data_with_labels.csv — for accuracy evaluation.
balanced_test_data.csv — feature-only for the app.
Evaluate accuracy (optional)
With the labeled CSV:
python scripts/evaluate_model_accuracy.py
Prints overall accuracy and a per-class classification report for present classes.
Sanity-check predictions (optional)
To test predictions on a feature-only CSV:
python scripts/test_model_predictions.py
Notes
Do not commit models/ or data/. Use .gitignore to keep these local.
If label classes differ between training and inference, ensure preprocessor label_encoder matches the trained model’s classes to avoid decoding errors.
If column mismatches occur, align the CSV columns with preprocessor['features'] or regenerate the sample using the provided scripts.
Acknowledgements
CIC-DDoS2019 dataset for network traffic features.
Streamlit, scikit-learn, pandas, and numpy for the application stack.
