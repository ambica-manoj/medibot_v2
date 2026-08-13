import time

from fastapi import APIRouter, HTTPException

from models.schemas import Citation, Metrics, QueryRequest, QueryResponse
from services import llm_service, preload_service, vector_store
from utils.logger import get_logger
from utils.spelling import suggest_correction

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    total_start = time.perf_counter()
    corrected_query = suggest_correction(payload.query)

    try:
        # Route greetings / small talk away from retrieval entirely
        if llm_service.is_general_chat(payload.query):
            answer, llm_ms = llm_service.answer_general_chat(payload.query)
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
            raise HTTPException(500, "No preloaded medical document is available")

        retrieval_start = time.perf_counter()
        chunks = vector_store.search_documents(doc_ids, payload.query)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not chunks:
            total_ms = (time.perf_counter() - total_start) * 1000
            return QueryResponse(
                answer="I couldn't find anything relevant to that in the medical document.",
                corrected_query=corrected_query,
                citations=[],
                follow_up_questions=[],
                metrics=Metrics(retrieval_ms=retrieval_ms, llm_ms=0.0, total_ms=total_ms, chunks_used=0),
            )

        context = "\n\n".join(
            f"[{c.filename}, page {c.pages[0] if c.pages else '?'}]\n{c.text}" for c in chunks
        )
        answer, llm_ms = llm_service.answer_with_context(payload.query, context)
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
        raise HTTPException(500, "An unexpected error occurred while processing the query.") from exc
