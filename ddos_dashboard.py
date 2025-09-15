import streamlit as st
import pickle
import pandas as pd

# ----------------------
# Load Preprocessor & Model
# ----------------------
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)  # dict with scaler, label_encoder, features

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="DDoS Attack Detection Dashboard", layout="wide")
st.title("🚨 DDoS Attack Detection Dashboard")
st.write("Upload network traffic data and detect possible DDoS attacks.")

# ----------------------
# File Upload Section
# ----------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # ----------------------
    # Preprocess Data
    # ----------------------
    try:
        feature_cols = preprocessor["features"]
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Missing required columns in uploaded data: {missing_cols}")
            st.stop()

        # Select only required features in correct order
        X = df[feature_cols]

        # Apply stored scaler
        X_processed = preprocessor["scaler"].transform(X)

    except Exception as e:
        st.error(f"Error while preprocessing: {e}")
        st.stop()

    # ----------------------
    # Make Predictions & Decode
    # ----------------------
    try:
        predictions_numeric = model.predict(X_processed)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Decode numeric predictions into string class names
    pred_labels = preprocessor['label_encoder'].inverse_transform(predictions_numeric)
    df["Prediction"] = pred_labels

    # Debug check for prediction values
    st.write("DEBUG - unique decoded prediction labels:", df["Prediction"].unique())

    # ----------------------
    # Display Results
    # ----------------------
    st.write("### 🔍 Detection Results:")
    st.dataframe(df)

    # ---- Count and break down attacks ----
    normal_count = (df["Prediction"] == "Benign").sum()
    attack_count = (df["Prediction"] != "Benign").sum()

    col1, col2 = st.columns(2)
    col1.metric("🛡️ Normal Traffic", int(normal_count))
    col2.metric("🚨 Detected Attacks", int(attack_count))

    attack_type_counts = df[df["Prediction"] != "Benign"]["Prediction"].value_counts()

    if not attack_type_counts.empty:
        st.write("### 📊 Attack Types Detected:")
        st.dataframe(attack_type_counts)
        st.bar_chart(attack_type_counts)
    else:
        st.info("No attacks detected in this batch.")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and scikit-learn")

else:
    st.info("⬆️ Upload a CSV file with all required features (generated or from the dataset).")
