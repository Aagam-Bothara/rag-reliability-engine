"""Plain text file parser."""

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_path


class TextParser:
    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def parse(self, file_path: str | Path, metadata: dict) -> tuple[str, dict]:
        file_path = Path(file_path)
        result = from_path(file_path)
        best = result.best()
        text = str(best) if best else file_path.read_text(encoding="utf-8")
        enriched = {**metadata, "encoding": str(best.encoding) if best else "utf-8"}
        return text, enriched
