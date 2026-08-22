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
        if _load_existing_local_index():
            return
        logger.info("No preloaded PDF configured")
        return

    if not os.path.exists(path):
        logger.warning("Preloaded PDF path configured but not found: %s", path)
        if _load_existing_local_index():
            return
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


def _load_existing_local_index() -> bool:
    """Register a bundled index when its source PDF is unavailable."""
    index_dir = settings.local_index_dir
    if not os.path.isdir(index_dir):
        return False

    for filename in os.listdir(index_dir):
        if not filename.startswith("preloaded-") or not filename.endswith("_meta.json"):
            continue

        doc_id = filename[: -len("_meta.json")]
        if not vector_store.index_exists(doc_id) or doc_id in PRELOADED_DOC_IDS:
            continue

        try:
            with open(os.path.join(index_dir, filename), encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read local index metadata %s: %s", filename, exc)
            continue

        document_name = metadata.get("filename", f"{doc_id}.pdf")
        PRELOADED_DOC_IDS.append(doc_id)
        PRELOADED_DOC_METADATA[doc_id] = {
            "filename": document_name,
            "s3_key": f"{settings.preloaded_s3_prefix}/{doc_id}/{document_name}",
            "pages": metadata.get("pages", 0),
            "chunks": metadata.get("chunks", 0),
        }
        logger.info("Loaded existing local index for %s as doc_id=%s", document_name, doc_id)
        return True

    return False


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
