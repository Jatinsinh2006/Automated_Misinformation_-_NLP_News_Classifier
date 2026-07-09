"""
train.py
--------
End-to-end training pipeline for the ISOT Fake News Detection dataset.

Usage:
    python train.py

Expects:
    data/Fake.csv
    data/True.csv

Produces:
    data/processed_news.csv
    models/best_model.pkl
    models/tfidf_vectorizer.pkl
    reports/confusion_matrix.png
    reports/model_accuracy.png
    reports/classification_report.txt
"""

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import preprocess_dataframe

DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data():
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")

    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        raise FileNotFoundError(
            "Fake.csv / True.csv not found in data/. "
            "Download the ISOT Fake News Dataset from Kaggle and place both "
            "files inside the data/ folder."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # ISOT dataset usually has 'title' and 'text' columns — combine them
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    df = df[["text", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def build_and_train():
    print("Loading data...")
    df = load_data()
    print(f"Total records: {len(df)}")

    print("Preprocessing text (this can take a few minutes)...")
    df = preprocess_dataframe(df, text_column="text", new_column="clean_text")
    df.to_csv(os.path.join(DATA_DIR, "processed_news.csv"), index=False)

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": LinearSVC(),
    }

    results = {}
    best_model_name = None
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
        }

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-score : {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")

    # ---- Save classification report for best model ----
    best_preds = best_model.predict(X_test)
    report_text = classification_report(
        y_test, best_preds, target_names=["Fake", "Real"]
    )
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(report_text)

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"]
    )
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ---- Model accuracy comparison plot ----
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}",
                  ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_accuracy.png"))
    plt.close()

    # ---- Save best model + vectorizer ----
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nSaved:")
    print(f"  models/best_model.pkl  ({best_model_name})")
    print("  models/tfidf_vectorizer.pkl")
    print("  reports/confusion_matrix.png")
    print("  reports/model_accuracy.png")
    print("  reports/classification_report.txt")

    return results, best_model_name


if __name__ == "__main__":
    build_and_train()
