import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import shap

# Page configuration
st.set_page_config(page_title="Mental Health Predictor", layout="centered")

# Background styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f3ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and selected features
model = joblib.load("final_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# Do not force device on model to avoid NotImplementedError
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# SHAP explainer
explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(np.zeros((1, len(selected_features)))))

# Title and instructions
st.title("Mental Health Prediction Using Genetic Algorithm-Based Feature Selection")
st.write("Designed by Jacqueline Chiazor for CS 548 Project")
st.write("Enter a short tweet-style message. The model will predict if it reflects a depressed mental state.")

# Session state for predictions
if 'recent_predictions' not in st.session_state:
    st.session_state.recent_predictions = []

# Single Prediction
user_input = st.text_area(" Type your message here:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        try:
            # ✅ Use CPU for encoding to avoid device errors
            X_input = embedder.encode([user_input], convert_to_numpy=True, device='cpu')
            X_selected = X_input[:, selected_features]

            prediction = model.predict(X_selected)[0]
            proba = model.predict_proba(X_selected)[0]
            confidence = round(np.max(proba) * 100, 2)

            label = "🟥 Depressed" if prediction == 1 else "🟩 Not Depressed"
            st.subheader(f"Prediction: {label}")
            st.write(f"**Confidence:** {confidence}%")

            if prediction == 1:
                st.warning("💡 You're not alone. If you're feeling distressed, consider reaching out for support.")
                with st.expander("Mental Health Resources"):
                    st.markdown("""
                    - **Crisis Text Line**: Text HOME to 741741 (US)
                    - **National Suicide Prevention Lifeline**: 1-800-273-TALK (8255)
                    - [Find more support](https://www.mentalhealth.gov/get-help)
                    """)

            shap_values = explainer.shap_values(X_selected)
            shap_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_selected[0]
            )

            st.subheader("SHAP Feature Impact")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_explanation, show=False, ax=ax)
            st.pyplot(fig)

            top_indices = np.argsort(model.coef_[0])[::-1][:5]
            st.write("Top influencing embedding dimensions:")
            st.write(top_indices.tolist())

            st.session_state.recent_predictions.append({
                "message": user_input,
                "prediction": "Depressed" if prediction == 1 else "Not Depressed",
                "confidence": confidence
            })

        except Exception as e:
            st.error(f"Error: {e}")

# Show recent predictions
if st.session_state.recent_predictions:
    st.subheader("Recent Predictions")
    st.table(pd.DataFrame(st.session_state.recent_predictions[::-1]))

# Batch prediction
st.markdown("---")
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type="csv")

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)

        if 'text' not in df_input.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            st.success(f"Loaded {len(df_input)} messages.")

            max_rows = len(df_input)
            sample_size = st.number_input(
                label="How many messages to sample for prediction?",
                min_value=1,
                max_value=max_rows,
                value=min(100, max_rows),
                step=1
            )

            df_input = df_input.sample(n=sample_size, random_state=42).reset_index(drop=True)

            # ✅ Force CPU during batch encoding too
            encoded = embedder.encode(df_input['text'].astype(str).tolist(), convert_to_numpy=True, device='cpu')
            encoded_selected = encoded[:, selected_features]

            predictions = model.predict(encoded_selected)
            probabilities = model.predict_proba(encoded_selected)

            df_input['Prediction'] = ['Depressed' if p == 1 else 'Not Depressed' for p in predictions]
            df_input['Confidence (%)'] = np.max(probabilities, axis=1) * 100

            st.write("Results:")
            st.dataframe(df_input)

            st.subheader("Prediction Summary")
            fig, ax = plt.subplots()
            df_input['Prediction'].value_counts().plot(kind='bar', color=['crimson', 'seagreen'], ax=ax)
            ax.set_title("Predicted Label Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            csv = df_input.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Results", data=csv, file_name='predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error: {e}")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts mental health status (Depressed / Not Depressed) "
    "based on short messages using BERT embeddings and Genetic Algorithm–selected features.\n\n"
    "**Model**: Logistic Regression\n"
    "**Features**: 200 selected from 384 BERT dimensions\n"
    "**Dataset**: Tweets from users with and without depression"
)

# Model evaluation
with st.expander("Model Evaluation"):
    st.write("F1 Score: 0.84")
    st.write("Accuracy: 85%")
    st.write("Selected Features: 200 out of 384")
