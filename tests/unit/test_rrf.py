"""Tests for Reciprocal Rank Fusion."""

from rag_engine.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_single_list():
    results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = reciprocal_rank_fusion([results], k=60)
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"
    assert fused[2][0] == "c"


def test_rrf_two_lists_overlap():
    list1 = [("a", 0.9), ("b", 0.8)]
    list2 = [("b", 0.95), ("a", 0.7)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # Both a and b appear in both lists, but b is rank 1 in list2, a is rank 1 in list1
    # Scores should be close
    ids = [cid for cid, _ in fused]
    assert set(ids) == {"a", "b"}


def test_rrf_disjoint_lists():
    list1 = [("a", 0.9)]
    list2 = [("b", 0.9)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    ids = [cid for cid, _ in fused]
    assert set(ids) == {"a", "b"}
    # Both have same RRF score (both rank 0 in their respective lists)
    assert fused[0][1] == fused[1][1]


def test_rrf_empty():
    fused = reciprocal_rank_fusion([[]], k=60)
    assert fused == []


def test_rrf_preserves_rank_order():
    list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    list2 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # a is rank 1 in both, b is rank 2 in both, c is rank 3 in both
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"
    assert fused[2][0] == "c"
