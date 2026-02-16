"""Registry mapping file extensions to parsers."""

from __future__ import annotations

from pathlib import Path

from rag_engine.exceptions import ParsingError
from rag_engine.ingestion.parser_html import HTMLParser
from rag_engine.ingestion.parser_markdown import MarkdownParser
from rag_engine.ingestion.parser_pdf import PDFParser
from rag_engine.ingestion.parser_text import TextParser
from rag_engine.protocols.ingestion import FileParser


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, FileParser] = {}

    def register(self, extension: str, parser: FileParser) -> None:
        self._parsers[extension.lower()] = parser

    def get_parser(self, filename: str) -> FileParser:
        ext = Path(filename).suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise ParsingError(
                f"No parser registered for extension '{ext}'. "
                f"Supported: {list(self._parsers.keys())}"
            )
        return parser

    def supported_types(self) -> list[str]:
        return list(self._parsers.keys())


def create_default_registry() -> ParserRegistry:
    """Create a registry with all built-in parsers."""
    registry = ParserRegistry()
    parsers: list[FileParser] = [TextParser(), MarkdownParser(), HTMLParser(), PDFParser()]
    for parser in parsers:
        for ext in parser.supported_extensions:
            registry.register(ext, parser)
    return registry
