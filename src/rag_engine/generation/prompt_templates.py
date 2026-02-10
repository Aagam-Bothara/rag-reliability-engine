"""All prompt templates for the RAG system."""

ANSWER_GENERATION_SYSTEM = """You are a precise, factual assistant. Answer questions using ONLY the provided evidence.
Rules:
- Cite evidence using [1], [2], etc. markers matching the evidence numbers.
- If the evidence doesn't contain enough information, say so clearly.
- Never make up information not present in the evidence.
- Be concise and direct."""

ANSWER_GENERATION_PROMPT = """Question: {query}

Evidence:
{evidence_block}

{decomposition_context}

Provide a clear, well-cited answer based on the evidence above."""

ANSWER_GENERATION_STRICT_SYSTEM = """You are a precise, factual assistant operating in STRICT mode.
Rules:
- ONLY state facts that are DIRECTLY and EXPLICITLY supported by the evidence.
- Cite every claim with [1], [2], etc.
- If ANY doubt exists about whether the evidence supports a claim, do NOT include it.
- If evidence is insufficient, state exactly what information is missing.
- Never infer, extrapolate, or generalize beyond the evidence."""

QUERY_DECOMPOSITION_PROMPT = """Break the following complex question into simpler, independent sub-questions that can be answered individually.
Return a JSON object with:
- "sub_questions": list of simple questions (max 5)
- "synthesis_instruction": how to combine the sub-answers into a final answer

If the question is already simple, return it as the only sub-question.

Question: {query}"""

GROUNDEDNESS_CHECK_PROMPT = """Evaluate how well the following answer is grounded in the provided evidence.

Answer: {answer}

Evidence:
{evidence_block}

For each claim in the answer, determine if it is directly supported by the evidence.
Return a JSON object:
- "score": float between 0.0 (not grounded) and 1.0 (fully grounded)
- "unsupported_claims": list of claims not supported by evidence"""

CONTRADICTION_DETECTION_PROMPT = """Analyze the following passages for contradictions.

{passages}

Identify any factual contradictions between the passages.
Return a JSON object:
- "contradictions": list of {{"passage_a": int, "passage_b": int, "description": str}}
- "contradiction_rate": float between 0.0 (no contradictions) and 1.0 (many contradictions)"""

ANSWER_CONTRADICTION_PROMPT = """Does the following answer contradict any of the evidence?

Answer: {answer}

Evidence:
{evidence_block}

Return a JSON object:
- "contradictions": list of {{"claim": str, "evidence_num": int, "description": str}}
- "contradiction_rate": float between 0.0 and 1.0"""

QUERY_REWRITE_PROMPT = """The following query didn't retrieve good results. Generate 3 alternative versions of this query that might retrieve better results. Use synonyms, rephrasings, and different angles.

Original query: {query}

Return a JSON object:
- "rewrites": list of 3 alternative query strings"""

SELF_CONSISTENCY_PROMPT = """Answer the following question briefly and directly based on the evidence.

Question: {query}

Evidence:
{evidence_block}

Provide a concise answer (1-3 sentences)."""


def format_evidence_block(chunks: list, max_chunks: int = 10) -> str:
    """Format chunks as a numbered evidence block for prompts."""
    lines = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.text if hasattr(chunk, "text") else chunk.chunk.text
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def format_decomposition_context(sub_questions: list[str] | None, synthesis: str | None) -> str:
    """Format decomposition plan for the generation prompt."""
    if not sub_questions or len(sub_questions) <= 1:
        return ""
    lines = ["Consider these aspects:"]
    for i, sq in enumerate(sub_questions, 1):
        lines.append(f"  {i}. {sq}")
    if synthesis:
        lines.append(f"\nSynthesis approach: {synthesis}")
    return "\n".join(lines)
