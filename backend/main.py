from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import query
from services import preload_service
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medibot API")
    try:
        preload_service.load_preloaded_documents()
    except Exception as exc:
        logger.error(
            "Failed to load preloaded documents during startup: %s. "
            "Queries will fail until a document is loaded.",
            exc,
            exc_info=True,
        )
    yield
    logger.info("Shutting down Medibot API")


app = FastAPI(title="Medibot API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        dict.fromkeys(
            [
                settings.frontend_origin.rstrip("/"),
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:3001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3001",
                "http://[::1]:3000",
                "http://[::1]:5173",
                "http://[::1]:3001",
            ]
        )
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
