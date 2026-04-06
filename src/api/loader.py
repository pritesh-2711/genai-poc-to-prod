"""File storage management for uploaded documents.

FileLoader manages the lifecycle of uploaded files on the local filesystem:

    Active files   → storage/{user_id}/active/{session_id}/{filename}
    Archived files → storage/{user_id}/archive/{session_id}/{filename}

When a session is deleted the entire session folder is moved from active to
archive.  FileLoader is only used by the API layer; the extraction module
receives the stored file path and does not interact with FileLoader directly.
"""

import shutil
from pathlib import Path


class FileLoader:
    """Handles saving, retrieving, and archiving uploaded files.

    Args:
        base_dir: Root storage directory. Defaults to "storage" (relative to
                  the working directory when the API server is started).
    """

    def __init__(self, base_dir: str = "storage") -> None:
        self._base = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        session_id: str,
    ) -> Path:
        """Write an uploaded file to active storage.

        Args:
            file_content: Raw bytes of the uploaded file.
            filename:     Original filename (used as the stored filename).
            user_id:      User identifier — becomes a path segment.
            session_id:   Session identifier — becomes a path segment.

        Returns:
            Absolute Path to the saved file.
        """
        dest_dir = self._active_dir(user_id, session_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        dest_path.write_bytes(file_content)
        return dest_path.resolve()

    def list_files(self, user_id: str, session_id: str) -> list[Path]:
        """Return all files currently in active storage for a session."""
        active_dir = self._active_dir(user_id, session_id)
        if not active_dir.exists():
            return []
        return [p for p in active_dir.iterdir() if p.is_file()]

    def archive(self, user_id: str, session_id: str) -> None:
        """Move a session's files from active to archive storage.

        Called when a session is deleted.  If there are no active files for
        the session this is a no-op.

        Args:
            user_id:    User identifier.
            session_id: Session identifier.
        """
        src = self._active_dir(user_id, session_id)
        if not src.exists():
            return
        dest = self._archive_dir(user_id, session_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_dir(self, user_id: str, session_id: str) -> Path:
        return self._base / user_id / "active" / session_id

    def _archive_dir(self, user_id: str, session_id: str) -> Path:
        return self._base / user_id / "archive" / session_id
