# Medibot v2

Level-up of the original Flask + local Llama-2 + FAISS Medical Chat Bot into
a FastAPI + AWS Bedrock + hybrid-search (FAISS + BM25) architecture with a
React frontend, following the `aws_rag_chatbot-main` pattern.

## What changed vs. v1

- **Flask → FastAPI**, with a simplified query-only route using a preloaded medical document
- **Local Llama-2 GGML (CTransformers) → AWS Bedrock (Claude 3 Haiku)**, called directly via `boto3`, no LangChain chain
- **Local-only PDFs → S3** for source document and index storage backup
- **Fixed-size chunking → page-aware chunking**, with OCR fallback (pytesseract + pdf2image) for scanned pages
- **FAISS-only → FAISS + BM25 hybrid search**, fused with Reciprocal Rank Fusion, dynamic top-k via score threshold instead of a fixed `k=2`
- **Single preloaded document** replaces user-upload/session flow
- **Vanilla JS/Jinja → React (Vite)** frontend talking to a REST API
- Added: citations (filename + page), 3 follow-up question suggestions per answer, spelling-correction suggestions, latency/chunk metrics, general-chat routing for greetings

## Project layout

```
backend/
  main.py                 FastAPI app, CORS, lifespan startup
  config.py               Settings (env-driven)
  requirements.txt
  .env.example
  routes/
    query.py              hybrid retrieval -> Bedrock -> answer
  services/
    pdf_service.py        page-aware extraction + chunking + OCR fallback
    embedding_service.py  sentence-transformers wrapper
    vector_store.py       FAISS + BM25 + RRF fusion, per-doc index files
    preload_service.py    load/preload medical PDF + S3/FAISS sync
    s3_service.py         upload/download/list/delete in S3
    llm_service.py        Bedrock invoke_model calls, prompts
  models/
    schemas.py            Pydantic request/response models
  utils/
    logger.py
    spelling.py           pyspellchecker suggestions
frontend/
  src/
    App.jsx
    api.js                 fetch wrapper
    components/
      ChatBox.jsx
      Message.jsx
```

## Backend setup

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in AWS keys, S3 bucket, region, Bedrock model id, and PRELOADED_PDF_PATH
uvicorn backend.main:app --reload --port 8000
```

You'll need:
- An AWS account with **Bedrock model access enabled** for the Claude model in `BEDROCK_MODEL_ID`, in the region set by `AWS_REGION`.
- An **S3 bucket** (`S3_BUCKET_NAME`) the credentials can read/write.
- A local medical PDF file referenced by `PRELOADED_PDF_PATH` in `backend/.env`.
- (Optional, for OCR) `tesseract-ocr` and `poppler-utils` installed at the OS level, since `pytesseract`/`pdf2image` are just Python wrappers around them.

## Frontend setup

```bash
cd frontend
npm install
npm run dev   # runs on http://localhost:3000, talks to http://localhost:8000
```

Set `VITE_API_BASE` in a `.env` file inside `frontend/` if the backend isn't on `localhost:8000`.

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/query` | Ask a medical question about the loaded document |

## Notes / things you'll still want to do

- This scaffold builds a simplified medical Q/A service using one preloaded
  medical document. Review chunk sizes, the score-threshold ratio, and the
  Bedrock model choice against your own content and cost/latency needs.
- `vector_store.py` persists indexes to local disk under `faiss_index/`
  (configurable) — for real multi-instance deployment, use shared storage
  or rebuild from S3 on cold start.
- Add auth in front of `/api/*` before deploying anywhere public — this
  scaffold has none.
