"""Text preprocessing for BM25 keyword search."""

from __future__ import annotations

import re

from rag_engine.config.constants import STOPWORDS


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
