"""
Thin wrapper around sentence-transformers so the model is loaded once and
shared across requests, rather than re-instantiated per document like the
old create_and_save_faiss_index() did.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    return np.array(embeddings).astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
