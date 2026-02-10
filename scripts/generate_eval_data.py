"""Generate a labeled evaluation dataset from seed documents using Gemini.

This script reads the seed documents, uses Gemini to auto-generate
evaluation questions across 5 categories, and writes the result to
tests/fixtures/eval_dataset.json.

Usage:
    python scripts/generate_eval_data.py

Requires RAG_GOOGLE_API_KEY in .env or environment.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google import genai
from google.genai import types
from rag_engine.config.settings import Settings

# ── Seed document content (same as scripts/seed_data.py) ──────────────────

SEED_DOCS = {
    "ai_overview": """# Artificial Intelligence Overview

## What is AI?
Artificial Intelligence (AI) is the simulation of human intelligence in machines. These machines are programmed to think like humans and mimic their actions.

## Types of AI
There are three main types of AI:
1. **Narrow AI** - designed for specific tasks (e.g., Siri, chess engines)
2. **General AI** - hypothetical AI with human-level intelligence
3. **Super AI** - hypothetical AI surpassing human intelligence

## Machine Learning
Machine learning is a subset of AI that enables systems to learn from data. Key approaches include:
- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Deep Learning
Deep learning uses neural networks with many layers. It has been particularly successful in:
- Image recognition
- Natural language processing
- Speech recognition
""",
    "rag_systems": """# Retrieval-Augmented Generation (RAG)

## What is RAG?
RAG is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating answers.

## Components of RAG
A typical RAG system includes:
1. **Document Store** - stores the knowledge base
2. **Retriever** - finds relevant documents for a query
3. **Generator** - produces answers using retrieved context

## Benefits of RAG
- Reduces hallucination by grounding answers in evidence
- Can be updated without retraining
- Provides citations for generated answers
- Works with domain-specific knowledge

## Challenges
- Retrieval quality directly impacts answer quality
- Latency can be high with large knowledge bases
- Chunk size and overlap affect performance
""",
}

# ── Domains for unanswerable question generation ──────────────────────────

UNANSWERABLE_DOMAINS = [
    "medicine and pharmacology",
    "finance and stock markets",
    "sports and athletics",
    "geography and world capitals",
    "chemistry and chemical reactions",
    "history and historical events",
    "cooking and recipes",
    "astronomy and space exploration",
    "law and legal systems",
    "music theory and composition",
    "marine biology and oceanography",
    "automotive engineering",
    "fashion and textile design",
    "architecture and urban planning",
    "agriculture and farming",
]

# ── Prompts ───────────────────────────────────────────────────────────────

FACTUAL_PROMPT = """You are generating evaluation test cases for a RAG system. The RAG system's knowledge base contains ONLY the following documents. Generate questions that CAN be answered from this content.

DOCUMENTS:
{documents}

Generate exactly {count} factual questions. For each question:
- The question must be answerable SOLELY from the documents above
- Include 1-3 expected keywords that MUST appear in a correct answer (use lowercase, partial words OK for matching like "retriev" to match "retrieval"/"retrieve"/"retriever")
- Vary the question style: what/how/why/which/explain/describe/list
- Cover different sections and topics evenly

Return valid JSON — an array of objects with these fields:
- "query": the question string
- "expected_answer_contains": array of expected keyword strings

Return ONLY the JSON array, no other text."""

MULTI_HOP_PROMPT = """You are generating evaluation test cases for a RAG system. The knowledge base contains ONLY these documents:

DOCUMENTS:
{documents}

Generate exactly {count} multi-hop questions that require synthesizing information from MULTIPLE sections or both documents to answer properly. These questions should need 2+ pieces of information combined.

For each question:
- It must require info from at least 2 different sections to answer well
- Include 1-3 expected keywords (lowercase, partial words OK)
- Make them genuinely require synthesis, not just listing facts

Return valid JSON — an array of objects:
- "query": the question string
- "expected_answer_contains": array of keyword strings

Return ONLY the JSON array, no other text."""

UNANSWERABLE_PROMPT = """Generate exactly {count} well-formed questions about the following domains that would be IMPOSSIBLE to answer using a knowledge base that only contains information about Artificial Intelligence and RAG (Retrieval-Augmented Generation) systems.

Domains: {domains}

Requirements:
- Generate 1 question per domain listed
- Questions should be specific and well-formed (not vague)
- Questions must have ZERO topical overlap with AI, machine learning, deep learning, or RAG
- Vary question types: factual, how-to, explanatory, comparative

Return valid JSON — an array of objects:
- "query": the question string
- "domain": which domain it belongs to

Return ONLY the JSON array, no other text."""

ADVERSARIAL_PROMPT = """You are generating adversarial evaluation test cases for a RAG system. The knowledge base contains ONLY information about:
1. AI basics (definition, narrow/general/super AI types)
2. Machine learning (supervised, unsupervised, reinforcement learning)
3. Deep learning (neural networks, image recognition, NLP, speech recognition)
4. RAG systems (components: document store/retriever/generator, benefits, challenges)

Generate exactly {count} adversarial questions that are RELATED to AI/ML/RAG topics but ask about specific details NOT covered in the knowledge base. These should be tricky — they sound like they should be answerable but actually aren't.

Examples of what's NOT in the KB:
- Transformer architecture, attention mechanisms, BERT, GPT
- Specific algorithms (gradient descent, backpropagation, k-means)
- Specific frameworks (PyTorch, TensorFlow, LangChain)
- Training procedures, hyperparameter tuning
- Specific metrics (BLEU, ROUGE, perplexity)
- Vector databases, embedding models, tokenization
- Prompt engineering, fine-tuning, RLHF
- AI ethics, bias, regulation

For each question:
- It should SEEM relevant to someone who knows the KB topics
- But the specific information requested is NOT present in the KB
- A well-calibrated system should abstain or warn

Return valid JSON — an array of objects:
- "query": the question string
- "why_adversarial": brief explanation of why this isn't answerable from the KB

Return ONLY the JSON array, no other text."""


async def generate_with_gemini(client: genai.Client, model: str, prompt: str) -> str:
    """Call Gemini and return the text response."""
    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=8192,
    )
    response = await client.aio.models.generate_content(
        model=model, contents=prompt, config=config
    )
    return response.text or ""


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from Gemini response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


async def generate_factual(client: genai.Client, model: str, count: int = 20) -> list[dict]:
    """Generate factual questions from seed documents."""
    docs_text = "\n\n---\n\n".join(SEED_DOCS.values())
    prompt = FACTUAL_PROMPT.format(documents=docs_text, count=count)
    raw = await generate_with_gemini(client, model, prompt)
    items = parse_json_response(raw)

    cases = []
    for i, item in enumerate(items[:count], 1):
        cases.append({
            "id": f"factual_{i:02d}",
            "query": item["query"],
            "mode": "normal",
            "expected_decision": "answer",
            "acceptable_decisions": ["answer"],
            "expected_answer_contains": item["expected_answer_contains"],
            "category": "factual",
        })
    return cases


async def generate_multi_hop(client: genai.Client, model: str, count: int = 10) -> list[dict]:
    """Generate multi-hop questions requiring synthesis across sections."""
    docs_text = "\n\n---\n\n".join(SEED_DOCS.values())
    prompt = MULTI_HOP_PROMPT.format(documents=docs_text, count=count)
    raw = await generate_with_gemini(client, model, prompt)
    items = parse_json_response(raw)

    cases = []
    for i, item in enumerate(items[:count], 1):
        cases.append({
            "id": f"multi_hop_{i:02d}",
            "query": item["query"],
            "mode": "normal",
            "expected_decision": "answer",
            "acceptable_decisions": ["answer"],
            "expected_answer_contains": item["expected_answer_contains"],
            "category": "multi-hop",
        })
    return cases


async def generate_unanswerable(
    client: genai.Client, model: str, count: int = 15
) -> list[dict]:
    """Generate unanswerable questions across diverse domains."""
    domains = UNANSWERABLE_DOMAINS[:count]
    prompt = UNANSWERABLE_PROMPT.format(count=len(domains), domains=", ".join(domains))
    raw = await generate_with_gemini(client, model, prompt)
    items = parse_json_response(raw)

    cases = []
    for i, item in enumerate(items[:count], 1):
        cases.append({
            "id": f"unanswerable_{i:02d}",
            "query": item["query"],
            "mode": "normal",
            "expected_decision": "abstain",
            "acceptable_decisions": ["abstain"],
            "expected_answer_contains": [],
            "category": "unanswerable",
        })
    return cases


async def generate_adversarial(
    client: genai.Client, model: str, count: int = 15
) -> list[dict]:
    """Generate adversarial near-miss questions about related but uncovered topics."""
    prompt = ADVERSARIAL_PROMPT.format(count=count)
    raw = await generate_with_gemini(client, model, prompt)
    items = parse_json_response(raw)

    cases = []
    for i, item in enumerate(items[:count], 1):
        cases.append({
            "id": f"adversarial_{i:02d}",
            "query": item["query"],
            "mode": "normal",
            "expected_decision": "abstain",
            "acceptable_decisions": ["abstain", "clarify"],
            "expected_answer_contains": [],
            "category": "adversarial",
        })
    return cases


def generate_strict_variants(answerable_cases: list[dict], count: int = 10) -> list[dict]:
    """Take answerable cases and create strict-mode variants."""
    cases = []
    for i, source in enumerate(answerable_cases[:count], 1):
        cases.append({
            "id": f"strict_{i:02d}",
            "query": source["query"],
            "mode": "strict",
            "expected_decision": "answer",
            "acceptable_decisions": ["answer", "clarify"],
            "expected_answer_contains": source["expected_answer_contains"],
            "category": "strict-mode",
        })
    return cases


def generate_strict_unanswerable(unanswerable_cases: list[dict], count: int = 5) -> list[dict]:
    """Take unanswerable cases and verify they also abstain in strict mode."""
    cases = []
    for i, source in enumerate(unanswerable_cases[:count], 1):
        cases.append({
            "id": f"strict_abstain_{i:02d}",
            "query": source["query"],
            "mode": "strict",
            "expected_decision": "abstain",
            "acceptable_decisions": ["abstain"],
            "expected_answer_contains": [],
            "category": "strict-mode",
        })
    return cases


async def main() -> None:
    settings = Settings()
    client = genai.Client(api_key=settings.google_api_key)
    model = settings.gemini_model

    print("Generating evaluation dataset using Gemini...\n")

    # Layer 1: Answerable questions from seed docs
    print("  [1/4] Generating factual questions (20)...")
    factual = await generate_factual(client, model, count=20)
    print(f"         Got {len(factual)} factual cases")

    print("  [2/4] Generating multi-hop questions (10)...")
    multi_hop = await generate_multi_hop(client, model, count=10)
    print(f"         Got {len(multi_hop)} multi-hop cases")

    # Layer 2: Unanswerable questions across diverse domains
    print("  [3/4] Generating unanswerable questions (15)...")
    unanswerable = await generate_unanswerable(client, model, count=15)
    print(f"         Got {len(unanswerable)} unanswerable cases")

    # Layer 3: Adversarial near-miss questions
    print("  [4/4] Generating adversarial questions (15)...")
    adversarial = await generate_adversarial(client, model, count=15)
    print(f"         Got {len(adversarial)} adversarial cases")

    # Strict-mode variants (mix of answerable + unanswerable)
    print("\n  Generating strict-mode variants...")
    all_answerable = factual + multi_hop
    strict_answer = generate_strict_variants(all_answerable, count=10)
    strict_abstain = generate_strict_unanswerable(unanswerable, count=5)
    strict = strict_answer + strict_abstain
    print(f"         Got {len(strict)} strict-mode cases")

    # Combine all
    dataset = factual + multi_hop + unanswerable + adversarial + strict

    # Write output
    output_path = Path(__file__).parent.parent / "tests" / "fixtures" / "eval_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"  DATASET GENERATED: {len(dataset)} total cases")
    print(f"{'=' * 50}")
    print(f"  Factual:        {len(factual)}")
    print(f"  Multi-hop:      {len(multi_hop)}")
    print(f"  Unanswerable:   {len(unanswerable)}")
    print(f"  Adversarial:    {len(adversarial)}")
    print(f"  Strict-mode:    {len(strict)}")
    print(f"\n  Written to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
