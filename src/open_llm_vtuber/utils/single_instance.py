"""Machine-wide single-instance enforcement for the Open-LLM-VTuber server."""

import os
import sys
from pathlib import Path
from types import TracebackType

from filelock import FileLock, Timeout

LOCK_FILENAME = "open-llm-vtuber-server.lock"


class SingleInstanceError(RuntimeError):
    """Raised when the server instance lock cannot be acquired."""


def default_lock_path() -> Path:
    """Return the shared lock-file path for the current operating system.

    Windows uses the machine-wide Windows temporary directory. POSIX systems
    use ``/tmp``, which is shared by users on both Linux and macOS. Keeping the
    path independent of the installation directory prevents copies of the
    project in different directories from starting concurrent servers.

    Returns:
        Path to the machine-wide server lock file.
    """
    if sys.platform == "win32":
        base_path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp"
    else:
        base_path = Path("/tmp")
    return base_path / LOCK_FILENAME


class SingleInstanceLock:
    """Hold an operating-system lock for one server process.

    The lock is held by the operating system rather than inferred from a PID
    file, so it is released automatically even if the server process crashes or
    is killed. The lock file itself is only coordination state and does not
    contain sensitive data.
    """

    def __init__(self, lock_path: Path | None = None) -> None:
        """Initialize a lock guard.

        Args:
            lock_path: Optional override used by tests and specialized
                deployments. When omitted, the machine-wide default is used.
        """
        self.lock_path = lock_path or default_lock_path()
        self._lock = FileLock(str(self.lock_path), mode=0o666)
        self._acquired = False

    def acquire(self) -> None:
        """Acquire the instance lock without waiting for another server.

        Raises:
            SingleInstanceError: If another server holds the lock or the lock
                cannot be enforced on this computer.
        """
        if self._acquired:
            return
        try:
            self._lock.acquire(blocking=False)
        except Timeout as exc:
            raise SingleInstanceError(
                "Open-LLM-VTuber is already running on this computer, or the "
                f"instance lock is inaccessible at '{self.lock_path}'. Close "
                "the existing server before starting another instance."
            ) from exc
        except (OSError, NotImplementedError) as exc:
            raise SingleInstanceError(
                "Open-LLM-VTuber cannot enforce a single instance because the "
                f"instance lock at '{self.lock_path}' could not be acquired: {exc}"
            ) from exc
        self._acquired = True

    def release(self) -> None:
        """Release the instance lock if this guard holds it."""
        if not self._acquired:
            return
        self._lock.release(force=True)
        self._acquired = False

    def __enter__(self) -> "SingleInstanceLock":
        """Acquire the lock when entering a context.

        Returns:
            This lock guard.
        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock when leaving a context.

        Args:
            exc_type: Exception type active in the context, if any.
            exc_value: Exception instance active in the context, if any.
            traceback: Traceback active in the context, if any.
        """
        self.release()
