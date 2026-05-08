"""Six unit tests locking the S1-T6 preprocessing contract.

If any of these break, every downstream model and the Gradio inference
path are also affected — `clean_text` runs at training AND inference.
"""

from src.preprocessing import build_numeric_features, clean_text


def test_full_pipeline_order_lowercase_then_punct_then_lemma() -> None:
    """The full ordered pipeline produces lemmatised, lowercased, punctuation-free output."""
    out = clean_text("Apple ANNOUNCES new PHONES, with cameras!")
    tokens = out.split()
    # lowercased and lemmatised
    assert "apple" in tokens
    assert "announce" in tokens  # 'announces' -> 'announce' (verb pass)
    assert "phone" in tokens  # 'phones' -> 'phone' (noun pass; WordNet knows 'phones')
    assert "camera" in tokens  # 'cameras' -> 'camera' (noun pass)
    # stopword removed
    assert "with" not in tokens
    # punctuation gone
    assert "," not in out
    assert "!" not in out


def test_empty_and_whitespace_only_input_return_empty_string() -> None:
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("\n\t") == ""


def test_url_stripping_removes_http_and_www_urls() -> None:
    out = clean_text("Read more at https://example.com/news/article today.")
    # URL parts gone
    assert "https" not in out
    assert "example" not in out
    assert "com" not in out
    # surrounding content survives
    assert "read" in out.split()
    assert "today" in out.split()


def test_lemma_forms_normalise_inflections_to_base() -> None:
    out_tokens = clean_text("running cars dogs barked").split()
    assert "run" in out_tokens  # running -> run (verb pass)
    assert "car" in out_tokens  # cars -> car (noun pass)
    assert "dog" in out_tokens  # dogs -> dog (noun pass)
    assert "bark" in out_tokens  # barked -> bark (verb pass)


def test_stopword_removal_drops_english_function_words() -> None:
    out_tokens = clean_text("the quick brown fox jumps over the lazy dog").split()
    assert "the" not in out_tokens
    assert "over" not in out_tokens
    # content words survive (lemmatised)
    assert "quick" in out_tokens
    assert "brown" in out_tokens
    assert "fox" in out_tokens


def test_numeric_feature_counts_match_raw_text() -> None:
    raw = "Hello, world! How are you?"
    feats = build_numeric_features(raw)
    # raw whitespace-split words
    assert feats["word_count"] == 5
    # raw character count (including spaces and punctuation)
    assert feats["char_count"] == len(raw)
    # punctuation marks: ',' '!' '?' = 3
    assert feats["punct_count"] == 3
    # empty input -> all zeros
    assert build_numeric_features("") == {"word_count": 0, "char_count": 0, "punct_count": 0}
