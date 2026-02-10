"""Markdown file parser."""

from __future__ import annotations

import re
from pathlib import Path


class MarkdownParser:
    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: str | Path, metadata: dict) -> tuple[str, dict]:
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")

        # Strip YAML front matter if present
        text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        enriched = {**metadata}
        if title_match:
            enriched["title"] = title_match.group(1).strip()

        return text, enriched
