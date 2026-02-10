"""PDF file parser using PyMuPDF4LLM for structured extraction."""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm


class PDFParser:
    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: str | Path, metadata: dict) -> tuple[str, dict]:
        file_path = Path(file_path)
        text = pymupdf4llm.to_markdown(str(file_path))

        enriched = {**metadata}
        # Extract basic PDF metadata
        try:
            import pymupdf

            doc = pymupdf.open(str(file_path))
            pdf_meta = doc.metadata
            if pdf_meta.get("title"):
                enriched["title"] = pdf_meta["title"]
            if pdf_meta.get("author"):
                enriched["author"] = pdf_meta["author"]
            enriched["page_count"] = doc.page_count
            doc.close()
        except Exception:
            pass

        return text, enriched
