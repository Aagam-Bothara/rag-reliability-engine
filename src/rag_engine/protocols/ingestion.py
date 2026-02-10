"""Protocol for file parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileParser(Protocol):
    def parse(self, file_path: str | Path, metadata: dict) -> tuple[str, dict]:
        """Returns (extracted_text, enriched_metadata)."""
        ...

    @property
    def supported_extensions(self) -> list[str]: ...
