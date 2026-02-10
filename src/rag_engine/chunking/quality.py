"""Post-chunking quality checks: near-duplicate detection and garbage filtering."""

from __future__ import annotations

import re

from datasketch import MinHash, MinHashLSH

from rag_engine.config.constants import (
    MIN_CHUNK_LENGTH,
    MAX_REPETITION_RATIO,
    MINHASH_NUM_PERM,
    NEAR_DUP_SIMILARITY_THRESHOLD,
)
from rag_engine.models.domain import Chunk
from rag_engine.observability.logger import get_logger

logger = get_logger("chunk_quality")


def filter_garbage_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Remove chunks that are too short, mostly non-alphabetic, or highly repetitive."""
    filtered = []
    removed = 0
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) < MIN_CHUNK_LENGTH:
            removed += 1
            continue
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.3:
            removed += 1
            continue
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < MAX_REPETITION_RATIO:
                removed += 1
                continue
        filtered.append(chunk)
    if removed:
        logger.info("garbage_filtered", removed=removed, remaining=len(filtered))
    return filtered


def detect_near_duplicates(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Detect near-duplicate chunk pairs using MinHash LSH. Returns list of (chunk_id_a, chunk_id_b)."""
    if len(chunks) < 2:
        return []

    lsh = MinHashLSH(threshold=NEAR_DUP_SIMILARITY_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    minhashes: dict[str, MinHash] = {}

    for chunk in chunks:
        mh = MinHash(num_perm=MINHASH_NUM_PERM)
        words = set(re.findall(r"\w+", chunk.text.lower()))
        for word in words:
            mh.update(word.encode("utf-8"))
        minhashes[chunk.chunk_id] = mh
        try:
            lsh.insert(chunk.chunk_id, mh)
        except ValueError:
            pass  # Duplicate key, skip

    duplicates: list[tuple[str, str]] = []
    seen = set()
    for chunk_id, mh in minhashes.items():
        candidates = lsh.query(mh)
        for candidate in candidates:
            if candidate != chunk_id:
                pair = tuple(sorted([chunk_id, candidate]))
                if pair not in seen:
                    seen.add(pair)
                    duplicates.append(pair)

    if duplicates:
        logger.warning("near_duplicates_found", count=len(duplicates))
    return duplicates


def compute_coverage(chunks: list[Chunk], original_text: str) -> float:
    """Measure what fraction of the original text is represented in chunks."""
    if not original_text:
        return 0.0
    original_words = set(re.findall(r"\w+", original_text.lower()))
    chunk_words: set[str] = set()
    for chunk in chunks:
        chunk_words.update(re.findall(r"\w+", chunk.text.lower()))
    if not original_words:
        return 0.0
    coverage = len(original_words & chunk_words) / len(original_words)
    logger.info("chunk_coverage", coverage=round(coverage, 4))
    return coverage
