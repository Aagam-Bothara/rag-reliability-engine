"""Tests for retrieval quality and confidence scoring."""

from rag_engine.config.settings import Settings
from rag_engine.scoring.confidence import ConfidenceScorer
from rag_engine.scoring.retrieval_quality import RetrievalQualityScorer


def test_rq_scorer_empty_candidates():
    settings = Settings(openai_api_key="x", google_api_key="x")
    scorer = RetrievalQualityScorer(settings)
    score, reasons = scorer.score([])
    assert score == 0.0
    assert "NO_RESULTS" in reasons


def test_rq_scorer_high_quality(sample_candidates):
    settings = Settings(openai_api_key="x", google_api_key="x")
    scorer = RetrievalQualityScorer(settings)
    score, reasons = scorer.score(sample_candidates)
    assert 0.0 <= score <= 1.0


def test_rq_scorer_single_candidate(sample_candidates):
    settings = Settings(openai_api_key="x", google_api_key="x")
    scorer = RetrievalQualityScorer(settings)
    score, reasons = scorer.score(sample_candidates[:1])
    assert 0.0 <= score <= 1.0


def test_confidence_scorer():
    settings = Settings(openai_api_key="x", google_api_key="x")
    scorer = ConfidenceScorer(settings)

    # High quality case
    conf = scorer.score(rq=0.8, groundedness=0.9, contradiction_rate=0.1)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5

    # Low quality case
    conf_low = scorer.score(rq=0.2, groundedness=0.3, contradiction_rate=0.8)
    assert conf_low < conf


def test_confidence_scorer_bounds():
    settings = Settings(openai_api_key="x", google_api_key="x")
    scorer = ConfidenceScorer(settings)

    # Should never exceed 1.0
    conf = scorer.score(rq=1.0, groundedness=1.0, contradiction_rate=0.0)
    assert conf <= 1.0

    # Should never go below 0.0
    conf = scorer.score(rq=0.0, groundedness=0.0, contradiction_rate=1.0)
    assert conf >= 0.0
