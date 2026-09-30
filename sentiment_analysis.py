from __future__ import annotations

import argparse

from src.sentiment_analysis.train import analyze_sentiment, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run sentiment analysis")
    parser.add_argument("--dataset", help="Optional dataset CSV path")
    parser.add_argument("--predict", help="Predict the sentiment of this text")
    parser.add_argument("--evaluate", action="store_true", help="Print evaluation metrics")
    args = parser.parse_args()

    if args.predict:
        print(analyze_sentiment(args.predict)["sentiment"])
    else:
        bundle = train_model(args.dataset)
        if args.evaluate:
            print(f"Accuracy: {bundle['accuracy']:.4f}")
            print(bundle["report"])
        else:
            print(f"Evaluation accuracy: {bundle['accuracy']:.4f}")


if __name__ == "__main__":
    main()
