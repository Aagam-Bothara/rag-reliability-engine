"""Structure-aware text chunker that splits by headings, paragraphs, and sentences."""

from __future__ import annotations

import re
from uuid import uuid4

import tiktoken

from rag_engine.chunking.overlap import compute_overlap_text
from rag_engine.config.constants import TIKTOKEN_ENCODING
from rag_engine.models.domain import Chunk


class StructureChunker:
    def __init__(
        self,
        doc_id: str = "",
        max_tokens: int = 512,
        overlap_pct: float = 0.15,
    ) -> None:
        self._doc_id = doc_id
        self._max_tokens = max_tokens
        self._overlap_pct = overlap_pct
        self._enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        doc_id = metadata.get("doc_id", self._doc_id) or str(uuid4())
        sections = self._split_by_headings(text)
        raw_chunks: list[dict] = []

        for heading_path, section_text in sections:
            if self._count_tokens(section_text) <= self._max_tokens:
                raw_chunks.append({
                    "text": section_text.strip(),
                    "heading_path": heading_path,
                })
            else:
                paragraphs = self._split_by_paragraphs(section_text)
                for para in paragraphs:
                    if self._count_tokens(para) <= self._max_tokens:
                        raw_chunks.append({
                            "text": para.strip(),
                            "heading_path": heading_path,
                        })
                    else:
                        sentences = self._split_by_sentences(para)
                        buffer = ""
                        for sent in sentences:
                            candidate = (buffer + " " + sent).strip() if buffer else sent
                            if self._count_tokens(candidate) <= self._max_tokens:
                                buffer = candidate
                            else:
                                if buffer:
                                    raw_chunks.append({
                                        "text": buffer.strip(),
                                        "heading_path": heading_path,
                                    })
                                buffer = sent
                        if buffer.strip():
                            raw_chunks.append({
                                "text": buffer.strip(),
                                "heading_path": heading_path,
                            })

        # Apply overlap
        chunks: list[Chunk] = []
        for i, rc in enumerate(raw_chunks):
            text_with_overlap = rc["text"]
            if i > 0 and self._overlap_pct > 0:
                overlap = compute_overlap_text(raw_chunks[i - 1]["text"], self._overlap_pct)
                if overlap:
                    text_with_overlap = overlap + "\n" + rc["text"]

            if not text_with_overlap.strip():
                continue

            chunks.append(Chunk(
                chunk_id=str(uuid4()),
                doc_id=doc_id,
                text=text_with_overlap,
                index=i,
                metadata={**metadata, "heading_path": rc["heading_path"]},
                token_count=self._count_tokens(text_with_overlap),
            ))

        return chunks

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
        """Split text by markdown headings. Returns list of (heading_path, section_text)."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections: list[tuple[list[str], str]] = []
        heading_stack: list[str] = []
        last_end = 0

        for match in heading_pattern.finditer(text):
            if match.start() > last_end:
                section_text = text[last_end : match.start()]
                if section_text.strip():
                    sections.append((list(heading_stack), section_text))

            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
            last_end = match.end()

        # Remaining text after last heading
        remaining = text[last_end:]
        if remaining.strip():
            sections.append((list(heading_stack), remaining))

        # If no headings found, return whole text as one section
        if not sections:
            sections = [([], text)]

        return sections

    @staticmethod
    def _split_by_paragraphs(text: str) -> list[str]:
        """Split text by double newlines (paragraphs)."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p for p in paragraphs if p.strip()]

    @staticmethod
    def _split_by_sentences(text: str) -> list[str]:
        """Split text by sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]
