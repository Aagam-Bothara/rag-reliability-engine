"""Run the evaluation harness against a live RAG Reliability Engine server.

Usage:
    1. Start the server:   python -m rag_engine.main
    2. Seed data:          python scripts/seed_data.py
    3. Run evaluation:     python scripts/run_eval.py [--base-url URL] [--output PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add src to path (matching seed_data.py pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_engine.evaluation.metrics import (
    EvalCaseResult,
    build_confusion_matrix,
    compute_category_metrics,
    compute_metrics,
)
from rag_engine.evaluation.runner import DEFAULT_BASE_URL, run_evaluation


def print_header(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def print_summary(metrics: dict) -> None:
    print_header("EVALUATION SUMMARY")
    print(f"  Total cases:          {metrics['total_cases']}")
    print(f"  Valid cases:          {metrics['valid_cases']}")
    print(f"  Errors:               {metrics['error_count']}")
    print(f"  Decision accuracy:    {metrics['decision_accuracy']:.1%}")
    print(f"  Abstain rate:         {metrics['abstain_rate']:.1%}")
    print(f"  Correct abstain rate: {metrics['correct_abstain_rate']:.1%}")
    print(f"  False abstain rate:   {metrics['false_abstain_rate']:.1%}")
    print(f"  Answer quality:       {metrics['answer_quality']:.1%}")
    print(f"  Avg confidence:       {metrics['avg_confidence']:.4f}")
    print(f"  Avg latency:          {metrics['avg_latency_ms']:.0f} ms")


def print_category_breakdown(by_category: dict) -> None:
    print_header("PER-CATEGORY BREAKDOWN")
    header = (
        f"  {'Category':<16} {'Count':>5} {'Accuracy':>10} "
        f"{'Confidence':>12} {'Latency':>10} {'Abstain':>10}"
    )
    print(header)
    print(f"  {'-' * 63}")
    for cat, m in sorted(by_category.items()):
        print(
            f"  {cat:<16} {m['count']:>5} "
            f"{m['decision_accuracy']:>9.1%} "
            f"{m['avg_confidence']:>11.4f} "
            f"{m['avg_latency_ms']:>8.0f}ms "
            f"{m['abstain_rate']:>9.1%}"
        )


def print_confusion_matrix(matrix: dict) -> None:
    print_header("CONFUSION MATRIX (expected \\ actual)")
    labels = ["answer", "clarify", "abstain"]
    header = f"  {'':>12}" + "".join(f"{label:>10}" for label in labels)
    print(header)
    print(f"  {'-' * 42}")
    for exp in labels:
        row = f"  {exp:>12}"
        for act in labels:
            row += f"{matrix[exp][act]:>10}"
        print(row)


def print_case_details(results: list[EvalCaseResult]) -> None:
    print_header("INDIVIDUAL CASE RESULTS")
    for r in results:
        if r.error:
            status = "ERROR"
        elif r.decision_correct:
            status = "PASS"
        else:
            status = "FAIL"

        print(
            f"  [{status:>5}] {r.case_id:<16} | "
            f"expected={r.expected_decision:<8} actual={r.actual_decision:<8} | "
            f"conf={r.confidence:.3f} | rq={r.retrieval_quality:.3f}"
        )
        if r.keywords_missing:
            print(f"         missing keywords: {r.keywords_missing}")
        if r.error:
            print(f"         error: {r.error}")


def save_results(results: list[EvalCaseResult], metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nRaw results saved to {output_path}")


async def main(base_url: str, output_path: Path) -> None:
    print(f"Running evaluation against {base_url} ...")
    print(f"Dataset: tests/fixtures/eval_dataset.json")

    results = await run_evaluation(base_url=base_url)

    metrics = compute_metrics(results)
    confusion = build_confusion_matrix(results)
    by_category = compute_category_metrics(results)

    # Add to metrics for saving
    metrics["confusion_matrix"] = confusion
    metrics["by_category"] = by_category

    print_summary(metrics)
    print_category_breakdown(by_category)
    print_confusion_matrix(confusion)
    print_case_details(results)

    save_results(results, metrics, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running server (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--output",
        default="data/eval_results.json",
        help="Path to save raw results JSON (default: data/eval_results.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.base_url, Path(args.output)))
