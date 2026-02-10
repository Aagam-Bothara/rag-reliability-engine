"""Overlap window utilities for chunking."""

from __future__ import annotations


def compute_overlap_text(prev_chunk_text: str, overlap_pct: float) -> str:
    """Return the trailing portion of prev_chunk_text to prepend to the next chunk."""
    if overlap_pct <= 0 or not prev_chunk_text:
        return ""
    words = prev_chunk_text.split()
    overlap_count = max(1, int(len(words) * overlap_pct))
    return " ".join(words[-overlap_count:])
