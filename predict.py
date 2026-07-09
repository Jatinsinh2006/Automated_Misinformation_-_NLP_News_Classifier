"""
predict.py
----------
Load the trained model + TF-IDF vectorizer and predict whether a given
news article is Fake or Real.

Usage:
    python predict.py "Some news article text here..."

Or import and use directly:
    from predict import predict_news
    predict_news("Some news article text here...")
"""

import sys
import pickle
import os

from preprocessing import preprocess_text

MODELS_DIR = "models"


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_news(text: str) -> dict:
    model, vectorizer = load_artifacts()

    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]
    label = "Real" if pred == 1 else "Fake"

    result = {"label": label}

    # Confidence score if the model supports probability/decision scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        result["confidence"] = float(max(proba))
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        result["confidence"] = float(1 / (1 + pow(2.718281828, -score)))  # sigmoid

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "news article text"')
        sys.exit(1)

    article_text = " ".join(sys.argv[1:])
    result = predict_news(article_text)
    print(f"Prediction : {result['label']}")
    if "confidence" in result:
        print(f"Confidence : {result['confidence']:.2%}")
