"""HTML file parser using BeautifulSoup."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


class HTMLParser:
    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def parse(self, file_path: str | Path, metadata: dict) -> tuple[str, dict]:
        file_path = Path(file_path)
        html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, and nav elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract title
        enriched = {**metadata}
        title_tag = soup.find("title")
        if title_tag:
            enriched["title"] = title_tag.get_text(strip=True)

        # Convert headings to markdown-style markers
        lines: list[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td"]):
            tag_name = element.name
            text = element.get_text(strip=True)
            if not text:
                continue
            if tag_name.startswith("h") and len(tag_name) == 2:
                level = int(tag_name[1])
                lines.append(f"{'#' * level} {text}")
            else:
                lines.append(text)
            lines.append("")  # blank line between elements

        return "\n".join(lines), enriched
