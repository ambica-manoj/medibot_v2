import hashlib
import json
import os
from typing import Dict, List

from config import settings
from services import pdf_service, s3_service, vector_store
from utils.logger import get_logger

logger = get_logger(__name__)

PRELOADED_DOC_IDS: List[str] = []
PRELOADED_DOC_METADATA: Dict[str, dict] = {}


def _make_doc_id(path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    return f"preloaded-{digest}"


def get_preloaded_doc_ids() -> List[str]:
    return PRELOADED_DOC_IDS.copy()


def list_preloaded_documents() -> List[dict]:
    return [
        {
            "doc_id": doc_id,
            "filename": meta["filename"],
            "pages": meta.get("pages", 0),
            "chunks": meta.get("chunks", 0),
        }
        for doc_id, meta in PRELOADED_DOC_METADATA.items()
    ]


def load_preloaded_documents() -> None:
    path = settings.preloaded_pdf_path
    if not path:
        logger.info("No preloaded PDF configured")
        return

    if not os.path.exists(path):
        logger.warning("Preloaded PDF path configured but not found: %s", path)
        return

    filename = os.path.basename(path)
    doc_id = _make_doc_id(os.path.abspath(path))
    s3_key = f"{settings.preloaded_s3_prefix}/{doc_id}/{filename}"
    index_s3_prefix = f"{settings.preloaded_s3_prefix}/{doc_id}"

    if doc_id in PRELOADED_DOC_IDS:
        logger.info("Preloaded document already loaded: %s", filename)
        return

    if vector_store.index_exists(doc_id):
        logger.info("Using local preloaded index for %s", filename)
    elif vector_store.download_index_from_s3(doc_id, index_s3_prefix):
        logger.info("Downloaded preloaded index from S3 for %s", filename)
    else:
        pages = pdf_service.extract_pages_from_pdf(path)
        chunks = pdf_service.chunk_pages(pages)
        if not chunks:
            logger.warning("Preloaded PDF %s contains no extractable text", filename)
            return

        vector_store.build_index_for_document(doc_id, filename, chunks, pages=len(pages))

        try:
            vector_store.upload_index_to_s3(doc_id, index_s3_prefix)
        except Exception as exc:
            logger.warning("Failed to upload preloaded index to S3: %s", exc)

    try:
        s3_service.upload_file_to_s3(path, s3_key)
    except Exception as exc:
        logger.warning("Failed to upload preloaded PDF to S3: %s", exc)

    meta = _load_preload_metadata(doc_id, filename, s3_key)
    PRELOADED_DOC_IDS.append(doc_id)
    PRELOADED_DOC_METADATA[doc_id] = meta
    logger.info("Loaded preloaded document %s as doc_id=%s", filename, doc_id)


def _load_preload_metadata(doc_id: str, filename: str, s3_key: str) -> dict:
    paths = vector_store.get_index_paths(doc_id)
    pages = 0
    chunks = 0
    if os.path.exists(paths["meta"]):
        try:
            with open(paths["meta"], encoding="utf-8") as f:
                meta = json.load(f)
                pages = meta.get("pages", 0)
                chunks = meta.get("chunks", 0)
        except Exception:
            pass
    return {
        "filename": filename,
        "s3_key": s3_key,
        "pages": pages,
        "chunks": chunks,
    }
