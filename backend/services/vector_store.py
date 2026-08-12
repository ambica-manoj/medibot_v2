"""
Hybrid retrieval: FAISS (dense) + BM25 (sparse), fused with Reciprocal Rank
Fusion, with a dynamic top-k cutoff based on a score-threshold ratio instead
of a fixed k=2 like the original project used.

Each uploaded document gets its own on-disk index, namespaced by doc_id:
  {index_dir}/{doc_id}.faiss
  {index_dir}/{doc_id}_chunks.pkl   (chunk text + page metadata)
  {index_dir}/{doc_id}_bm25.pkl
  {index_dir}/{doc_id}_meta.json
"""
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import settings
from services import s3_service
from services.embedding_service import embed_texts, embed_query
from services.pdf_service import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    pages: List[int]
    doc_id: str
    filename: str
    score: float


def _paths(doc_id: str):
    base = settings.local_index_dir
    os.makedirs(base, exist_ok=True)
    return {
        "faiss": os.path.join(base, f"{doc_id}.faiss"),
        "chunks": os.path.join(base, f"{doc_id}_chunks.pkl"),
        "bm25": os.path.join(base, f"{doc_id}_bm25.pkl"),
        "meta": os.path.join(base, f"{doc_id}_meta.json"),
    }


def get_index_paths(doc_id: str) -> dict[str, str]:
    return _paths(doc_id)


def index_exists(doc_id: str) -> bool:
    paths = _paths(doc_id)
    return all(os.path.exists(path) for path in paths.values())


def upload_index_to_s3(doc_id: str, s3_prefix: str) -> None:
    paths = _paths(doc_id)
    for key, local_path in paths.items():
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Index asset missing: {local_path}")
        s3_key = f"{s3_prefix}/{os.path.basename(local_path)}"
        s3_service.upload_file_to_s3(local_path, s3_key)


def download_index_from_s3(doc_id: str, s3_prefix: str) -> bool:
    paths = _paths(doc_id)
    keys = s3_service.list_documents(prefix=f"{s3_prefix}/")
    if not keys:
        return False

    expected = {os.path.basename(p) for p in paths.values()}
    if not expected.issubset(set(os.path.basename(k) for k in keys)):
        return False

    for local_path in paths.values():
        filename = os.path.basename(local_path)
        s3_key = f"{s3_prefix}/{filename}"
        s3_service.download_file_from_s3(s3_key, local_path)
    return True


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_index_for_document(doc_id: str, filename: str, chunks: List[Chunk], pages: int = 0) -> None:
    """Build and persist FAISS + BM25 indexes for a single document."""
    logger.info("Building hybrid index for doc_id=%s (%d chunks)", doc_id, len(chunks))
    paths = _paths(doc_id)

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, paths["faiss"])

    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    with open(paths["bm25"], "wb") as f:
        pickle.dump(bm25, f)

    with open(paths["chunks"], "wb") as f:
        pickle.dump(chunks, f)

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(chunks),
                "pages": pages,
            },
            f,
        )

    logger.info("Hybrid index built and saved for doc_id=%s", doc_id)


def _load_document_index(doc_id: str):
    paths = _paths(doc_id)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    index = faiss.read_index(paths["faiss"])
    with open(paths["bm25"], "rb") as f:
        bm25 = pickle.load(f)
    with open(paths["chunks"], "rb") as f:
        chunks: List[Chunk] = pickle.load(f)
    with open(paths["meta"]) as f:
        meta = json.load(f)
    return index, bm25, chunks, meta


def _reciprocal_rank_fusion(
    dense_ranking: List[int], sparse_ranking: List[int], k: int = 60
) -> dict:
    """Combine two rankings (lists of chunk indices, best first) into fused scores."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranking):
        scores[idx] = scores.get(idx, 0.0) + settings.dense_weight / (k + rank + 1)
    for rank, idx in enumerate(sparse_ranking):
        scores[idx] = scores.get(idx, 0.0) + settings.sparse_weight / (k + rank + 1)
    return scores


def search_document(doc_id: str, query: str, top_n_per_method: int = 15) -> List[RetrievedChunk]:
    loaded = _load_document_index(doc_id)
    if loaded is None:
        logger.warning("No index found for doc_id=%s", doc_id)
        return []
    index, bm25, chunks, meta = loaded

    # Dense search
    q_emb = embed_query(query).reshape(1, -1)
    k = min(top_n_per_method, len(chunks))
    distances, indices = index.search(q_emb, k)
    dense_ranking = [i for i in indices[0] if i != -1]

    # Sparse search
    bm25_scores = bm25.get_scores(_tokenize(query))
    sparse_ranking = list(np.argsort(bm25_scores)[::-1][:top_n_per_method])

    fused = _reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=settings.rrf_k)
    if not fused:
        return []

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranked[0][1]
    cutoff = top_score * settings.score_threshold_ratio

    results = []
    for idx, score in ranked:
        if score < cutoff and len(results) > 0:
            break
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                text=chunk.text,
                pages=chunk.pages,
                doc_id=doc_id,
                filename=meta["filename"],
                score=float(score),
            )
        )
        if len(results) >= settings.max_chunks:
            break

    logger.info("search_document(doc_id=%s) returned %d chunks", doc_id, len(results))
    return results


def search_documents(doc_ids: List[str], query: str) -> List[RetrievedChunk]:
    """Search across multiple documents (e.g. all docs in a session) and merge by score."""
    all_results: List[RetrievedChunk] = []
    for doc_id in doc_ids:
        all_results.extend(search_document(doc_id, query))
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[: settings.max_chunks]


def delete_document_index(doc_id: str) -> None:
    paths = _paths(doc_id)
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    logger.info("Deleted index files for doc_id=%s", doc_id)
