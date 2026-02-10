"""Reciprocal Rank Fusion for merging retrieval results."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked result lists using RRF.

    Args:
        result_lists: Each list contains (chunk_id, score) tuples sorted by score descending.
        k: RRF constant (higher = more weight to lower-ranked results).

    Returns:
        Merged (chunk_id, rrf_score) tuples sorted by RRF score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for result_list in result_lists:
        for rank, (chunk_id, _) in enumerate(result_list):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
