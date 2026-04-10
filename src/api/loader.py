"""File storage management for uploaded documents.

Two implementations share the BaseFileLoader interface:

    LocalFileLoader — saves files to disk under storage/{user_id}/active/{session_id}/
                      Active → archive on session delete.
                      cleanup_temp() is a no-op (file persists on disk).

    S3FileLoader    — uploads files permanently to S3 at {user_id}/active/{session_id}/{filename}.
                      Also writes a tempfile for Docling (which needs a local path).
                      cleanup_temp() deletes the tempfile after ingestion completes.
                      archive() copies S3 objects from active/ to archive/ prefix then deletes originals.

The active implementation is wired in main.py based on config.storage_config.deployment.
"""

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class BaseFileLoader(ABC):
    """Abstract file storage interface.

    Callers must:
      1. Call save() to store the file and get a local Path suitable for Docling.
      2. After the ingestion pipeline finishes, call cleanup_temp() on that path.
         For local storage this is a no-op; for cloud storage it deletes the tempfile.
    """

    @abstractmethod
    def save(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        session_id: str,
    ) -> Path:
        """Persist the file and return a local Path for the ingestion pipeline."""

    def cleanup_temp(self, path: Path) -> None:
        """Remove any temporary file created during save().

        Default is a no-op — override in cloud implementations where save()
        writes a tempfile that should not persist after ingestion.
        """

    @abstractmethod
    def list_files(self, user_id: str, session_id: str) -> list:
        """Return the stored files for a session (paths or S3 keys)."""

    @abstractmethod
    def archive(self, user_id: str, session_id: str) -> None:
        """Move a session's files to archive storage.  Called on session delete."""


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------

class LocalFileLoader(BaseFileLoader):
    """Stores uploaded files on the local filesystem.

    Layout:
        storage/{user_id}/active/{session_id}/{filename}   ← active
        storage/{user_id}/archive/{session_id}/{filename}  ← after session deleted
    """

    def __init__(self, base_dir: str = "storage") -> None:
        self._base = Path(base_dir)

    def save(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        session_id: str,
    ) -> Path:
        dest_dir = self._base / user_id / "active" / session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        dest_path.write_bytes(file_content)
        return dest_path.resolve()

    # cleanup_temp() — inherited no-op; file stays on disk.

    def list_files(self, user_id: str, session_id: str) -> list[Path]:
        active_dir = self._base / user_id / "active" / session_id
        if not active_dir.exists():
            return []
        return [p for p in active_dir.iterdir() if p.is_file()]

    def archive(self, user_id: str, session_id: str) -> None:
        src = self._base / user_id / "active" / session_id
        if not src.exists():
            return
        dest = self._base / user_id / "archive" / session_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


# ---------------------------------------------------------------------------
# AWS S3 implementation
# ---------------------------------------------------------------------------

class S3FileLoader(BaseFileLoader):
    """Stores uploaded files in AWS S3.

    S3 key layout:
        {user_id}/active/{session_id}/{filename}   ← active
        {user_id}/archive/{session_id}/{filename}  ← after session deleted

    save() uploads to S3 for permanent storage AND writes a NamedTemporaryFile
    so Docling (which needs a real file path) can process the file.
    The caller must call cleanup_temp(path) after the ingestion pipeline finishes.
    """

    def __init__(self, bucket: str, region: str) -> None:
        import boto3
        self._bucket = bucket
        self._s3 = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def save(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        session_id: str,
    ) -> Path:
        # Persist permanently in S3
        key = self._active_key(user_id, session_id, filename)
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=file_content)

        # Write a tempfile for the ingestion pipeline (Docling needs a local path)
        suffix = Path(filename).suffix or ".tmp"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_content)
        tmp.close()
        return Path(tmp.name)

    def cleanup_temp(self, path: Path) -> None:
        """Delete the tempfile created by save() after ingestion completes."""
        path.unlink(missing_ok=True)

    def list_files(self, user_id: str, session_id: str) -> list[str]:
        """Return S3 keys of all active files for the session."""
        prefix = f"{user_id}/active/{session_id}/"
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def archive(self, user_id: str, session_id: str) -> None:
        """Copy session objects from active/ to archive/ prefix, then delete originals."""
        prefix = f"{user_id}/active/{session_id}/"
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            src_key = obj["Key"]
            dest_key = src_key.replace("/active/", "/archive/", 1)
            self._s3.copy_object(
                Bucket=self._bucket,
                CopySource={"Bucket": self._bucket, "Key": src_key},
                Key=dest_key,
            )
            self._s3.delete_object(Bucket=self._bucket, Key=src_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_key(self, user_id: str, session_id: str, filename: str) -> str:
        return f"{user_id}/active/{session_id}/{filename}"
