"""
streamlit_app.py
-----------------
Interactive Streamlit web app for Fake News Detection.

Run with:
    streamlit run streamlit_app.py
"""

import pickle
import os

import streamlit as st

from preprocessing import preprocess_text

MODELS_DIR = "models"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def main():
    st.title("📰 Fake News Detection")
    st.write(
        "Paste a news article below and the model will predict whether it's "
        "**Fake** or **Real**, using TF-IDF + Machine Learning trained on the "
        "ISOT Fake News Dataset."
    )

    if not (
        os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))
        and os.path.exists(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    ):
        st.error(
            "⚠️ Trained model not found. Run `python train.py` first to "
            "generate models/best_model.pkl and models/tfidf_vectorizer.pkl."
        )
        return

    model, vectorizer = load_artifacts()

    news_text = st.text_area(
        "Enter news article text:",
        height=220,
        placeholder="Paste the full news article here...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_clicked = st.button("🔍 Predict", type="primary")

    if predict_clicked:
        if not news_text.strip():
            st.warning("Please enter some news text first.")
        else:
            with st.spinner("Analyzing..."):
                clean = preprocess_text(news_text)
                vec = vectorizer.transform([clean])
                pred = model.predict(vec)[0]

                confidence = None
                if hasattr(model, "predict_proba"):
                    confidence = max(model.predict_proba(vec)[0])
                elif hasattr(model, "decision_function"):
                    score = model.decision_function(vec)[0]
                    confidence = 1 / (1 + pow(2.718281828, -score))

            if pred == 1:
                st.success("✅ This looks like **REAL** news.")
            else:
                st.error("🚫 This looks like **FAKE** news.")

            if confidence is not None:
                st.progress(min(confidence, 1.0))
                st.caption(f"Confidence: {confidence:.2%}")

    st.divider()
    st.caption(
        "Built with Python, Scikit-learn, NLTK & Streamlit • "
        "Dataset: ISOT Fake News Dataset (Kaggle)"
    )


if __name__ == "__main__":
    main()
