"""Query normalization, language detection, and intent extraction."""

from __future__ import annotations

import re
import unicodedata

from langdetect import detect

from rag_engine.models.domain import ProcessedQuery
from rag_engine.observability.logger import get_logger

logger = get_logger("query_understanding")


class QueryUnderstanding:
    async def process(self, raw_query: str) -> ProcessedQuery:
        # 1. Normalize
        normalized = self._normalize(raw_query)

        # 2. Language detection
        try:
            language = detect(normalized)
        except Exception:
            language = "en"

        # 3. Intent classification (simple heuristic)
        intent = self._classify_intent(normalized)

        # 4. Constraint extraction
        constraints = self._extract_constraints(normalized)

        logger.info(
            "query_processed",
            language=language,
            intent=intent,
            constraints=constraints,
        )

        return ProcessedQuery(
            normalized=normalized,
            language=language,
            intent=intent,
            constraints=constraints,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _classify_intent(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        if any(w in q for w in ["how to", "how do", "how can", "steps to"]):
            return "how_to"
        if any(w in q for w in ["what is", "what are", "define", "explain"]):
            return "factual"
        if any(w in q for w in ["why", "reason", "cause"]):
            return "causal"
        if any(w in q for w in ["list", "enumerate", "name all"]):
            return "list"
        return "general"

    @staticmethod
    def _extract_constraints(query: str) -> dict:
        constraints: dict = {}
        # Time range patterns
        year_match = re.findall(r"\b(20\d{2})\b", query)
        if year_match:
            constraints["years"] = year_match
        # "after/before/since" patterns
        time_match = re.search(r"(after|before|since|until)\s+(\w+\s?\d{0,4})", query, re.I)
        if time_match:
            constraints["time_filter"] = {
                "type": time_match.group(1).lower(),
                "value": time_match.group(2).strip(),
            }
        return constraints
