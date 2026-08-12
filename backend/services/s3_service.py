"""
S3-backed storage for uploaded source PDFs, replacing the old local-only
`Data/` folder. Indexes themselves stay on local disk (see vector_store.py);
only the original documents live in S3, so they persist and can be
re-downloaded / re-indexed if the local index is ever lost.
"""
import boto3
from botocore.exceptions import ClientError
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )
    return _client


def upload_file_to_s3(local_path: str, s3_key: str) -> str:
    client = _get_client()
    logger.info("Uploading %s to s3://%s/%s", local_path, settings.s3_bucket_name, s3_key)
    client.upload_file(local_path, settings.s3_bucket_name, s3_key)
    return s3_key


def download_file_from_s3(s3_key: str, local_path: str) -> str:
    client = _get_client()
    logger.info("Downloading s3://%s/%s to %s", settings.s3_bucket_name, s3_key, local_path)
    client.download_file(settings.s3_bucket_name, s3_key, local_path)
    return local_path


def list_documents(prefix: str = "") -> list[str]:
    client = _get_client()
    response = client.list_objects_v2(Bucket=settings.s3_bucket_name, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def delete_file_from_s3(s3_key: str) -> None:
    client = _get_client()
    try:
        client.delete_object(Bucket=settings.s3_bucket_name, Key=s3_key)
        logger.info("Deleted s3://%s/%s", settings.s3_bucket_name, s3_key)
    except ClientError as e:
        logger.error("Failed to delete s3://%s/%s: %s", settings.s3_bucket_name, s3_key, e)
        raise
