import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOPWORDS = ENGLISH_STOP_WORDS - {"no", "nor", "not", "never", "only"}
_STOPWORDS -= {"very", "really", "extremely", "incredibly"}


def clean_text(text: str) -> str:
    plain = re.sub(r"https?://\S+|@\w+", " ", str(text).lower())
    plain = re.sub(r"\b(?:is|was|were|are|do|does|did|can|could|will|would)n['’]t\b", " not", plain)
    plain = re.sub(r"\bcannot\b", " can not", plain)
    tokens = re.findall(r"[a-z]+", plain)
    return " ".join(token for token in tokens if token not in _STOPWORDS)
