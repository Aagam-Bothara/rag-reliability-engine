"""Verification decision maker: combine groundedness, contradiction, and consistency signals."""

from __future__ import annotations

from rag_engine.config.settings import Settings
from rag_engine.models.domain import VerificationResult
from rag_engine.scoring.reason_codes import ReasonCode


class VerificationDecisionMaker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def decide(
        self,
        groundedness: float,
        contradiction_rate: float,
        self_consistency: float | None,
        mode: str = "normal",
    ) -> VerificationResult:
        # Select thresholds based on mode
        if mode == "strict":
            ground_pass = self._settings.strict_groundedness_pass_threshold
            contra_pass = self._settings.strict_contradiction_pass_threshold
        else:
            ground_pass = self._settings.groundedness_pass_threshold
            contra_pass = self._settings.contradiction_pass_threshold

        ground_warn = self._settings.groundedness_warn_threshold
        contra_warn = self._settings.contradiction_warn_threshold

        reason_codes: list[str] = []

        if groundedness < ground_warn:
            reason_codes.append(ReasonCode.LOW_GROUNDEDNESS)
        if contradiction_rate > contra_warn:
            reason_codes.append(ReasonCode.CONTRADICTION_FOUND)
        if self_consistency is not None and self_consistency < 0.4:
            reason_codes.append(ReasonCode.SELF_INCONSISTENCY)

        # Decision logic
        if groundedness >= ground_pass and contradiction_rate <= contra_pass:
            decision = "pass"
        elif groundedness >= ground_warn and contradiction_rate <= contra_warn:
            decision = "warn"
        else:
            decision = "abstain"

        return VerificationResult(
            groundedness_score=groundedness,
            contradiction_rate=contradiction_rate,
            self_consistency_score=self_consistency,
            decision=decision,
            reason_codes=reason_codes,
        )
