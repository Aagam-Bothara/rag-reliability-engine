"""Evaluation runner: loads dataset, queries the live API, collects results."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from rag_engine.evaluation.metrics import EvalCaseResult

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONCURRENCY = 3

DATASET_PATH = (
    Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "eval_dataset.json"
)


def load_dataset(path: Path | None = None) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    p = path or DATASET_PATH
    with open(p) as f:
        return json.load(f)


async def run_single_case(
    client: httpx.AsyncClient,
    case: dict,
    semaphore: asyncio.Semaphore,
) -> EvalCaseResult:
    """Run a single evaluation case against the API."""
    async with semaphore:
        expected = case["expected_decision"]
        acceptable = case.get("acceptable_decisions", [expected])
        expected_keywords = case.get("expected_answer_contains", [])

        try:
            response = await client.post(
                "/query",
                json={"query": case["query"], "mode": case.get("mode", "normal")},
            )
            response.raise_for_status()
            data = response.json()

            actual_decision = data["decision"]
            actual_answer = data.get("answer", "")
            confidence = data.get("confidence", 0.0)
            debug = data.get("debug", {})
            rq = debug.get("retrieval_quality", 0.0)
            latency = debug.get("latency_ms", 0.0)
            reasons = data.get("reasons", [])

            # Check keywords (case-insensitive substring match)
            answer_lower = actual_answer.lower()
            found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

            return EvalCaseResult(
                case_id=case["id"],
                query=case["query"],
                category=case["category"],
                mode=case.get("mode", "normal"),
                expected_decision=expected,
                acceptable_decisions=acceptable,
                actual_decision=actual_decision,
                expected_answer_contains=expected_keywords,
                actual_answer=actual_answer,
                confidence=confidence,
                retrieval_quality=rq,
                latency_ms=latency,
                reasons=reasons,
                decision_correct=actual_decision in acceptable,
                keywords_found=found,
                keywords_missing=missing,
            )
        except Exception as e:
            return EvalCaseResult(
                case_id=case["id"],
                query=case["query"],
                category=case["category"],
                mode=case.get("mode", "normal"),
                expected_decision=expected,
                acceptable_decisions=acceptable,
                actual_decision="error",
                expected_answer_contains=expected_keywords,
                actual_answer="",
                confidence=0.0,
                retrieval_quality=0.0,
                latency_ms=0.0,
                reasons=[],
                decision_correct=False,
                keywords_found=[],
                keywords_missing=expected_keywords,
                error=str(e),
            )


async def run_evaluation(
    base_url: str = DEFAULT_BASE_URL,
    dataset_path: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[EvalCaseResult]:
    """Run the full evaluation suite against a live server.

    Loads the eval dataset, sends each query to the API, and returns
    a list of EvalCaseResult objects with all metrics captured.
    """
    dataset = load_dataset(dataset_path)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
    ) as client:
        # Verify server is up
        try:
            health = await client.get("/health")
            health.raise_for_status()
            health_data = health.json()
            print(
                f"Server healthy: {health_data.get('doc_count', '?')} docs, "
                f"{health_data.get('chunk_count', '?')} chunks"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach server at {base_url}/health — is the server running? Error: {e}"
            ) from e

        # Run all cases with bounded concurrency
        tasks = [run_single_case(client, case, semaphore) for case in dataset]
        results = await asyncio.gather(*tasks)

    return list(results)
