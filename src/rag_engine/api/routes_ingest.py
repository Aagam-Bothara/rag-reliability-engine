"""Document ingestion endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from rag_engine.api.dependencies import get_ingest_pipeline
from rag_engine.api.rate_limiter import rate_limit
from rag_engine.exceptions import IngestionError
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.models.schemas import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    metadata: str = Form("{}"),
    pipeline: IngestionPipeline = Depends(get_ingest_pipeline),
    _auth: dict = Depends(rate_limit),
) -> IngestResponse:
    # Parse metadata
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Save uploaded file to temp location
    suffix = Path(file.filename or "file.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await pipeline.ingest_file(tmp_path, parsed_metadata)
        return result
    except IngestionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
