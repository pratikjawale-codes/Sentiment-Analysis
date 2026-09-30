# Sentiment Analysis

A local sentiment analyzer with a negation-aware word lexicon, a review-ready web interface, and an optional TF-IDF training/evaluation pipeline.

## Structure

- `app.py` - Flask web app
- `sentiment_analysis.py` - CLI entry point
- `src/sentiment_analysis/config.py` - paths and config
- `src/sentiment_analysis/dataset.py` - dataset loader with sample fallback
- `src/sentiment_analysis/preprocessing.py` - text cleaning
- `src/sentiment_analysis/train.py` - sentiment scoring and optional model evaluation
- `data/sample_tweets.csv` - local demo dataset

## Run

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Then open:

`http://127.0.0.1:5000`

Run evaluation or a prediction from the terminal:

```bash
.venv\Scripts\python.exe sentiment_analysis.py --evaluate
.venv\Scripts\python.exe sentiment_analysis.py --predict "The crew was helpful"
```

## Behavior

- Predictions run locally without a trained model, external data, or downloads.
- The analyzer handles common positive/negative terms, emphasis, and negation (for example, “not good”).
- `--evaluate` optionally evaluates TF-IDF + Logistic Regression on the bundled sample or a supplied CSV. The tiny sample is for demonstration, not meaningful benchmarking.

## API

`POST /predict`

```json
{ "text": "I love this airline and the crew was amazing." }
```

Response:

```json
{ "sentiment": "positive", "score": 1.0, "confidence": 100, "text": "I love this airline and the crew was amazing." }
```