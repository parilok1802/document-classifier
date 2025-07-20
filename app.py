# app.py

import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App UI
st.set_page_config(page_title="Text Classifier", layout="centered")
st.title("📄 Document Type Classifier")

sample_texts = [
    "This page contains the profit and loss summary for the company.",
    "The balance sheet includes assets, liabilities and equity details."
]

st.subheader("📝 Sample Texts")
for i, txt in enumerate(sample_texts, 1):
    st.markdown(f"**Sample {i}:** {txt}")

if st.button("🔮 Predict"):
    vec = vectorizer.transform(sample_texts)
    preds = model.predict(vec)
    st.subheader("✅ Predictions")
    for txt, pred in zip(sample_texts, preds):
        st.markdown(f"**📄 Text:** {txt[:60]}...\n👉 Predicted Class: `{pred}`")
