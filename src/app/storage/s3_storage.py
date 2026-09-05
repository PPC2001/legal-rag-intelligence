"""S3-compatible object storage service (MinIO / AWS S3) with local fallback.

Enables storing raw contract documents in MinIO or AWS S3 without filling local disk.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class S3StorageService:
    """Manages document uploads and retrieval from MinIO / AWS S3 with local disk fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._bucket_name = self._settings.s3_bucket_name
        self._s3_client = self._init_client()

    def _init_client(self):
        """Initialise S3 client if endpoint or credentials are configured."""
        endpoint = self._settings.s3_endpoint_url
        access_key = (
            self._settings.s3_access_key.get_secret_value()
            if self._settings.s3_access_key
            else None
        )
        secret_key = (
            self._settings.s3_secret_key.get_secret_value()
            if self._settings.s3_secret_key
            else None
        )

        if not endpoint and not access_key:
            logger.info("MinIO/S3 not configured; using local filesystem storage.")
            return None

        try:
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self._settings.s3_region,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
            # Ensure bucket exists
            try:
                client.head_bucket(Bucket=self._bucket_name)
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code")
                if error_code in ("404", "NoSuchBucket"):
                    logger.info("Creating S3/MinIO bucket: %s", self._bucket_name)
                    client.create_bucket(Bucket=self._bucket_name)
                else:
                    logger.warning("Bucket check returned: %s", exc)

            logger.info(
                "Connected to S3/MinIO storage at %s (bucket=%s)",
                endpoint,
                self._bucket_name,
            )
            return client
        except Exception as exc:
            logger.warning(
                "Failed to initialize S3/MinIO client (%s). Falling back to local disk.",
                exc,
            )
            return None

    @property
    def is_s3_enabled(self) -> bool:
        """True if S3/MinIO client is connected."""
        return self._s3_client is not None

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        """Upload a file to MinIO/S3 or local upload_dir.

        Returns:
            Storage key or relative file path identifier.
        """
        # Sanitize filename to prevent directory traversal
        safe_filename = Path(filename).name

        if self._s3_client:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self._s3_client.upload_fileobj(
                io.BytesIO(file_bytes),
                self._bucket_name,
                safe_filename,
                ExtraArgs=extra_args if extra_args else None,
            )
            logger.info("Uploaded %s to S3 bucket %s", safe_filename, self._bucket_name)
            return f"s3://{self._bucket_name}/{safe_filename}"

        # Local filesystem fallback
        upload_dir = Path(self._settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_path = upload_dir / safe_filename
        local_path.write_bytes(file_bytes)
        logger.info("Saved %s to local storage: %s", safe_filename, local_path)
        return str(local_path)

    def get_file_bytes(self, file_path_or_key: str) -> bytes:
        """Retrieve raw file bytes from S3/MinIO or local filesystem."""
        if file_path_or_key.startswith("s3://"):
            parts = file_path_or_key[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()

        local_path = Path(file_path_or_key)
        return local_path.read_bytes()

    def delete_file(self, file_path_or_key: str) -> bool:
        """Delete file from S3 or local disk."""
        try:
            if file_path_or_key.startswith("s3://"):
                parts = file_path_or_key[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1]
                self._s3_client.delete_object(Bucket=bucket, Key=key)
                return True
            local_path = Path(file_path_or_key)
            if local_path.exists():
                local_path.unlink()
                return True
            return False
        except Exception as exc:
            logger.error("Failed to delete %s: %s", file_path_or_key, exc)
            return False


_storage_service: S3StorageService | None = None


def get_storage_service(settings: Settings | None = None) -> S3StorageService:
    """Return a singleton S3StorageService instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = S3StorageService(settings)
    return _storage_service
