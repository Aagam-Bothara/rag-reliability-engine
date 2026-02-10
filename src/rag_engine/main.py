"""Entrypoint: run the RAG Reliability Engine server."""

import uvicorn

from rag_engine.api.app import create_app
from rag_engine.config.settings import Settings


def main() -> None:
    settings = Settings()
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
