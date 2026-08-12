from typing import List, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int
    snippet: str


class QueryRequest(BaseModel):
    query: str


class Metrics(BaseModel):
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    chunks_used: int


class QueryResponse(BaseModel):
    answer: str
    corrected_query: Optional[str] = None
    citations: List[Citation] = []
    follow_up_questions: List[str] = []
    metrics: Metrics
    is_general_chat: bool = False


