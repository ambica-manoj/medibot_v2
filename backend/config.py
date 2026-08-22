"""
Centralized configuration for the Medibot backend.
Reads from environment / .env so nothing sensitive is hardcoded.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"

    # S3
    s3_bucket_name: str = "medibot-documents"

    # Bedrock
    bedrock_model_id: str = "google.gemma-3-4b-it"

    # Local index storage
    local_index_dir: str = "faiss_index"

    # Preloaded medical PDF
    preloaded_pdf_path: str = ""
    preloaded_s3_prefix: str = "preloaded"

    # CORS
    frontend_origin: str = "http://localhost:3000"

    # Retrieval
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    score_threshold_ratio: float = 0.75  # keep chunks scoring >= 75% of top fused score
    max_chunks: int = 8


settings = Settings()
