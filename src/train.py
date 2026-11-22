"""
Model Training and NLP Evaluation Pipeline for Sentiment Analysis.
Compares Multinomial Naive Bayes, Logistic Regression, and Calibrated LinearSVC.
Saves evaluation metrics, confusion matrices, and serialized model pipelines.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.preprocessor import clean_text
except ImportError:
    from preprocessor import clean_text

def run_training():
    print("=" * 60)
    print("NLP Sentiment Analysis Training & Benchmarking")
    print("=" * 60)

    # 1. Load dataset
    data_path = os.path.join("data", "sentiment_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    # 2. Text Preprocessing
    print("\n[Step 1/5] Preprocessing text data with NLP pipeline...")
    df["cleaned_text"] = df["text"].apply(lambda t: clean_text(t, remove_stopwords=False))

    X = df["cleaned_text"]
    y = df["sentiment"]
    labels = sorted(list(y.unique()))  # ['Negative', 'Neutral', 'Positive']
    print(f"Class labels: {labels}")
    print("Class distribution:")
    for lbl, count in y.value_counts().items():
        print(f"  - {lbl}: {count} ({count/len(y)*100:.1f}%)")

    # 3. Stratified Train-Test Split
    print("\n[Step 2/5] Performing 80/20 stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Feature Extraction (TF-IDF with Unigram + Bigram)
    print("\n[Step 3/5] Building TF-IDF Vectorizer (unigram + bigram, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} features")

    # 5. Define Candidate Classifiers
    # Note: LinearSVC is wrapped in CalibratedClassifierCV to provide well-calibrated class probabilities
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.5, max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000),
            cv=3
        )
    }

    # 6. Train, Evaluate & Benchmark
    print("\n[Step 4/5] Training and benchmarking NLP classifiers...")
    results = {}
    best_model_name = None
    best_f1 = -1.0

    for name, clf in candidate_models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_vec, y_train)

        # 5-fold cross validation on training set
        cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5, scoring="accuracy")
        
        # Test set evaluation
        y_pred = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

        print(f"Test Accuracy: {acc:.4f} | Weighted F1: {f1:.4f} | 5-Fold CV Mean: {cv_scores.mean():.4f}")

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": cm,
            "classification_report": report
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nBest Performing Model: {best_model_name} (Weighted F1: {best_f1:.4f})")

    # 7. Serialize Artifacts
    print("\n[Step 5/5] Saving model pipelines and evaluation metrics...")
    os.makedirs("models", exist_ok=True)

    # Save complete end-to-end best pipeline (raw cleaned text -> prediction)
    best_clf = candidate_models[best_model_name]
    best_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_clf)
    ])
    joblib.dump(best_pipeline, os.path.join("models", "best_sentiment_pipeline.joblib"))

    # Save all models & vectorizer so users can dynamically switch in Streamlit
    joblib.dump({
        "vectorizer": vectorizer,
        "models": candidate_models,
        "labels": labels,
        "best_model_name": best_model_name
    }, os.path.join("models", "vectorizer_and_models.joblib"))

    # Save metrics JSON for charts and app display
    metrics_summary = {
        "best_model": best_model_name,
        "labels": labels,
        "vocab_size": len(vectorizer.vocabulary_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": results
    }

    with open(os.path.join("models", "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\nModel artifacts saved successfully:")
    print("  - models/best_sentiment_pipeline.joblib")
    print("  - models/vectorizer_and_models.joblib")
    print("  - models/model_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
