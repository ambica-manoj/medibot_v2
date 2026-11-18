# Medibot v2

Medibot is a medical question-answering application built using **Retrieval-Augmented Generation (RAG)**. It retrieves relevant information from a preloaded medical PDF and uses **Amazon Bedrock** to generate answers through a React chat interface.

## Features

- Medical question answering using a PDF knowledge base.
- Hybrid search combining semantic search and keyword matching.
- Page-aware PDF processing with OCR fallback for scanned pages.
- Follow-up question and spelling suggestions.
- Source filename, page number, and text snippets returned by the API.
- Response time and retrieved-chunk count displayed in the chat.
- Local index storage with Amazon S3 backup and restore.

## Architecture

The application has two main stages:

### Document Processing

1. Extract text from the medical PDF using pypdf, with OCR fallback where needed.
2. Split the text into overlapping chunks while preserving page numbers.
3. Generate embeddings using SentenceTransformers and build a FAISS index.
4. Create a BM25 keyword index for the same chunks.
5. Save the indexes locally and back them up to Amazon S3.

### Question Answering

1. The React frontend sends the user's question to the FastAPI backend.
2. FAISS retrieves passages with similar meaning, while BM25 retrieves keyword matches.
3. Reciprocal Rank Fusion (RRF) combines both rankings to select context.
4. The question and retrieved context are sent to Amazon Bedrock to generate an answer.
5. The backend returns the answer, source metadata, follow-up suggestions, and timing metrics.

Recognized greetings bypass document retrieval.

## Tech Stack

| Component | Technology |
| --- | --- |
| Frontend | React, Vite |
| Backend | Python, FastAPI, Pydantic |
| PDF Processing | pypdf, Tesseract, pdf2image |
| Embeddings | SentenceTransformers — all-MiniLM-L6-v2 |
| Semantic Search | FAISS |
| Keyword Search | BM25 |
| Ranking Fusion | Reciprocal Rank Fusion |
| Answer Generation | Amazon Bedrock |
| Storage | Local files, Amazon S3 |
| AWS Integration | boto3 |

## Why Hybrid RAG?

Semantic search helps find passages with similar meaning, while keyword search helps match specific medical terms. Combining both methods provides two ways to find relevant context before generating an answer.

The PDF acts as an external knowledge base; it is not used to train or fine-tune the language model.


## Current Scope

- Uses a preloaded medical PDF rather than user uploads.
- Each question is processed independently; conversation history is not sent to the model.
- Source metadata is available through the API but is not currently displayed in the chat interface.
- The configured Bedrock model must match the request format used by the backend.

## Disclaimer

Medibot is an educational project for general medical information. It is not a clinically validated diagnostic tool or a substitute for professional medical advice.
