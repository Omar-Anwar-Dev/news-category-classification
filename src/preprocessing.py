"""Text preprocessing pipeline used at training and inference time.

Pipeline order (mandated by the assignment spec):
    1. lowercase
    2. noise removal (URL, HTML tag, punctuation, digits)
    3. NLTK word_tokenize
    4. stopword removal (English)
    5. WordNet lemmatisation (noun then verb pass)

Plus a small builder for three numeric features computed from the *raw*
text (before cleaning): word_count, char_count, punct_count. They are
stacked alongside TF-IDF in the classical feature pipeline.

The same `clean_text()` function is used at training and inference time,
so any change here ripples through every model. The 6 unit tests in
``tests/test_preprocessing.py`` lock the contract.
"""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

__all__ = ["clean_text", "build_numeric_features"]


# Compiled regex patterns — built once, reused per call.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_PUNCT_RE = re.compile(r"[^\w\s]")
_DIGIT_RE = re.compile(r"\d+")


def _ensure_nltk_resources() -> None:
    """Idempotent NLTK corpus download.

    CI and Colab pre-warm these via a workflow step / setup cell, but the
    function still runs for safety on first import in fresh environments.
    """
    needed = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for resource, path in needed.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()
_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=16384)
def _lemma(word: str) -> str:
    """Two-pass lemma: noun first (handles 'cars'→'car'), then verb (handles 'running'→'run')."""
    return _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(word, pos="n"), pos="v")


def clean_text(text: str) -> str:
    """Apply the full preprocessing pipeline to a raw string.

    Returns the cleaned, space-joined token sequence. Empty / whitespace-only
    input always returns an empty string.
    """
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    tokens = [_lemma(t) for t in tokens]
    return " ".join(tokens).strip()


def build_numeric_features(text: str) -> dict[str, int]:
    """Compute three count-based numeric features from *raw* text.

    Counted on the original string before cleaning, because punctuation
    is meaningful (e.g. exclamation marks in tabloid headlines) and would
    otherwise be erased by `clean_text`.
    """
    if not text:
        return {"word_count": 0, "char_count": 0, "punct_count": 0}
    return {
        "word_count": len(text.split()),
        "char_count": len(text),
        "punct_count": len(_PUNCT_RE.findall(text)),
    }
