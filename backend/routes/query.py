import re
import time

from fastapi import APIRouter, HTTPException

from models.schemas import Citation, Metrics, QueryRequest, QueryResponse
from services import llm_service, preload_service, vector_store
from utils.logger import get_logger
from utils.spelling import suggest_correction

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


def _clean_answer(text: str) -> str:
    """Remove language that exposes the retrieval mechanism."""
    phrases = (
        "based on the provided text",
        "based on the provided information",
        "based on the provided context",
        "based on the provided document",
        "the provided text states",
        "the provided information states",
        "according to the provided",
        "from the provided",
    )
    for phrase in phrases:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    total_start = time.perf_counter()
    corrected_query = suggest_correction(payload.query)

    try:
        # Route greetings / small talk away from retrieval entirely
        if llm_service.is_general_chat(payload.query):
            answer, llm_ms = llm_service.answer_general_chat(payload.query)
            answer = _clean_answer(answer)
            total_ms = (time.perf_counter() - total_start) * 1000
            return QueryResponse(
                answer=answer,
                corrected_query=corrected_query,
                citations=[],
                follow_up_questions=[],
                metrics=Metrics(retrieval_ms=0.0, llm_ms=llm_ms, total_ms=total_ms, chunks_used=0),
                is_general_chat=True,
            )

        doc_ids = preload_service.get_preloaded_doc_ids()
        if not doc_ids:
            raise HTTPException(500, "Medical knowledge base not available")

        retrieval_start = time.perf_counter()
        chunks = vector_store.search_documents(doc_ids, payload.query)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not chunks:
            total_ms = (time.perf_counter() - total_start) * 1000
            return QueryResponse(
                answer="I don't have information about that topic.",
                corrected_query=corrected_query,
                citations=[],
                follow_up_questions=[],
                metrics=Metrics(retrieval_ms=retrieval_ms, llm_ms=0.0, total_ms=total_ms, chunks_used=0),
            )

        context = "\n\n".join(c.text.strip() for c in chunks)
        answer, llm_ms = llm_service.answer_with_context(payload.query, context)
        answer = _clean_answer(answer)
        follow_ups = llm_service.suggest_follow_ups(payload.query, answer)

        citations = [
            Citation(
                doc_id=c.doc_id,
                filename=c.filename,
                page=c.pages[0] if c.pages else 0,
                snippet=c.text[:220],
            )
            for c in chunks
        ]

        total_ms = (time.perf_counter() - total_start) * 1000
        return QueryResponse(
            answer=answer,
            corrected_query=corrected_query,
            citations=citations,
            follow_up_questions=follow_ups,
            metrics=Metrics(
                retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=total_ms, chunks_used=len(chunks)
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /api/query")
        raise HTTPException(500, "An error occurred while processing your query.") from exc
