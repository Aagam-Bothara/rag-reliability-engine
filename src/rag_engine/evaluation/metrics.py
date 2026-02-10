"""Evaluation metric computation for the RAG Reliability Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby


@dataclass
class EvalCaseResult:
    """Result of running a single evaluation case."""

    case_id: str
    query: str
    category: str
    mode: str
    expected_decision: str
    acceptable_decisions: list[str]
    actual_decision: str
    expected_answer_contains: list[str]
    actual_answer: str
    confidence: float
    retrieval_quality: float
    latency_ms: float
    reasons: list[str]
    decision_correct: bool
    keywords_found: list[str]
    keywords_missing: list[str]
    error: str | None = None


def compute_metrics(results: list[EvalCaseResult]) -> dict:
    """Compute all evaluation metrics from raw results.

    Returns a dict with overall accuracy, abstain rates, answer quality,
    averages, and error count.
    """
    total = len(results)
    if total == 0:
        return _empty_metrics()

    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    # Decision accuracy (only count non-error cases)
    decision_correct = sum(r.decision_correct for r in valid)
    decision_accuracy = decision_correct / len(valid) if valid else 0.0

    # Abstain rates
    abstain_count = sum(1 for r in valid if r.actual_decision == "abstain")
    abstain_rate = abstain_count / len(valid) if valid else 0.0

    # Correct abstain rate: of expected-abstain cases, how many actually abstained
    expected_abstain = [r for r in valid if r.expected_decision == "abstain"]
    correct_abstain = sum(1 for r in expected_abstain if r.actual_decision == "abstain")
    correct_abstain_rate = correct_abstain / len(expected_abstain) if expected_abstain else 0.0

    # False abstain rate: of expected-answer cases, how many abstained
    expected_answer = [r for r in valid if r.expected_decision == "answer"]
    false_abstain = sum(1 for r in expected_answer if r.actual_decision == "abstain")
    false_abstain_rate = false_abstain / len(expected_answer) if expected_answer else 0.0

    # Answer quality: for non-abstain results with expected keywords,
    # what fraction had ALL keywords found
    answerable_with_keywords = [
        r for r in valid
        if r.actual_decision != "abstain"
        and r.expected_decision != "abstain"
        and r.expected_answer_contains
    ]
    if answerable_with_keywords:
        all_kw_found = sum(1 for r in answerable_with_keywords if not r.keywords_missing)
        answer_quality = all_kw_found / len(answerable_with_keywords)
    else:
        answer_quality = 0.0

    # Averages
    avg_confidence = sum(r.confidence for r in valid) / len(valid) if valid else 0.0
    avg_latency = sum(r.latency_ms for r in valid) / len(valid) if valid else 0.0

    return {
        "total_cases": total,
        "valid_cases": len(valid),
        "decision_accuracy": decision_accuracy,
        "abstain_rate": abstain_rate,
        "correct_abstain_rate": correct_abstain_rate,
        "false_abstain_rate": false_abstain_rate,
        "answer_quality": answer_quality,
        "avg_confidence": avg_confidence,
        "avg_latency_ms": avg_latency,
        "error_count": len(errors),
    }


def build_confusion_matrix(results: list[EvalCaseResult]) -> dict[str, dict[str, int]]:
    """Build 3x3 confusion matrix for decision predictions.

    Rows are expected decisions, columns are actual decisions.
    """
    labels = ["answer", "clarify", "abstain"]
    matrix: dict[str, dict[str, int]] = {exp: {act: 0 for act in labels} for exp in labels}

    for r in results:
        if r.error is not None:
            continue
        exp = r.expected_decision
        act = r.actual_decision
        if exp in labels and act in labels:
            matrix[exp][act] += 1

    return matrix


def compute_category_metrics(results: list[EvalCaseResult]) -> dict[str, dict]:
    """Compute per-category breakdowns of key metrics."""
    valid = [r for r in results if r.error is None]
    if not valid:
        return {}

    categories: dict[str, dict] = {}
    sorted_results = sorted(valid, key=lambda r: r.category)

    for cat, group in groupby(sorted_results, key=lambda r: r.category):
        cat_results = list(group)
        n = len(cat_results)
        categories[cat] = {
            "count": n,
            "decision_accuracy": sum(r.decision_correct for r in cat_results) / n,
            "avg_confidence": sum(r.confidence for r in cat_results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in cat_results) / n,
            "abstain_rate": sum(1 for r in cat_results if r.actual_decision == "abstain") / n,
        }

    return categories


def _empty_metrics() -> dict:
    return {
        "total_cases": 0,
        "valid_cases": 0,
        "decision_accuracy": 0.0,
        "abstain_rate": 0.0,
        "correct_abstain_rate": 0.0,
        "false_abstain_rate": 0.0,
        "answer_quality": 0.0,
        "avg_confidence": 0.0,
        "avg_latency_ms": 0.0,
        "error_count": 0,
    }
