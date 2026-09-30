from __future__ import annotations

import re
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .dataset import load_training_data
from .preprocessing import clean_text

_POLARITY = {
    "amazing": 2.5, "awesome": 2.5, "best": 2, "beautiful": 2, "brilliant": 2.5,
    "delightful": 2, "excellent": 2.5, "fantastic": 2.5, "friendly": 1.5, "glad": 1.5,
    "good": 1.5, "great": 2, "happy": 2, "helpful": 1.5, "impressive": 2,
    "love": 2.5, "lovely": 2, "perfect": 2.5, "polite": 1, "quick": 1,
    "recommend": 1.5, "smooth": 1.5, "thank": 1, "thanks": 1, "wonderful": 2.5,
    "enjoyable": 1.5, "comfortable": 1, "on-time": 1, "easy": 1, "win": 1,
    "bad": -1.8, "awful": -2.5, "annoyed": -1.8, "boring": -1.5, "broken": -1.8,
    "cancelled": -1.5, "cancel": -1.5, "confusing": -1.5, "delay": -1.5,
    "delayed": -1.8, "disappointed": -2, "disgusting": -2.5, "fail": -2,
    "frustrating": -2, "hate": -2.5, "horrible": -2.5, "late": -1.5,
    "lost": -1.5, "missed": -1.5, "poor": -2, "problem": -1.5, "rude": -2,
    "slow": -1.3, "terrible": -2.5, "unhelpful": -2, "unacceptable": -2.5,
    "uncomfortable": -1.5, "unfriendly": -1.8, "useless": -2, "wait": -1,
    "worst": -2.5, "wrong": -1.5, "mess": -1.5, "overpriced": -1.5,
}
_NEGATIONS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "isnt", "wasnt", "werent", "arent", "dont", "doesnt", "didnt", "cant", "couldnt", "wont", "wouldnt"}
_BOOSTERS = {"very": 1.5, "really": 1.5, "extremely": 2, "so": 1.3, "incredibly": 2}


def analyze_sentiment(text: str) -> dict[str, str | float]:
    total = 0.0
    magnitude = 0.0
    for sentence in re.split(r"[.!?;]+", text):
        tokens = clean_text(sentence).split()
        for index, token in enumerate(tokens):
            value = _POLARITY.get(token, 0.0)
            if not value:
                continue
            previous = tokens[max(0, index - 3):index]
            if tokens[max(0, index - 2):index] == ["not", "only"]:
                previous = [word for word in previous if word != "not"]
            if any(word in _NEGATIONS for word in previous):
                value *= -1
            for word in previous:
                value *= _BOOSTERS.get(word, 1.0)
            total += value
            magnitude += abs(value)

    score = total / magnitude if magnitude else 0.0
    sentiment = "positive" if score > 0.12 else "negative" if score < -0.12 else "neutral"
    return {"sentiment": sentiment, "score": round(score, 2), "confidence": round(abs(score) * 100)}


def train_model(dataset_path: str | None = None) -> dict[str, Any]:
    df = load_training_data(dataset_path)
    X = df["text"].apply(clean_text)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    X_all_vec = vectorizer.fit_transform(X)
    model.fit(X_all_vec, y)
    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy,
        "report": report,
    }


def predict_text(model_bundle: dict[str, Any], text: str) -> str:
    return str(analyze_sentiment(text)["sentiment"])


def evaluate_model(dataset_path: str | None = None) -> tuple[float, str]:
    bundle = train_model(dataset_path)
    return bundle["accuracy"], bundle["report"]
