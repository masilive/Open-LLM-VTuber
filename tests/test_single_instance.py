"""Tests for cross-process single-instance enforcement."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.open_llm_vtuber.utils.single_instance import (
    LOCK_FILENAME,
    SingleInstanceError,
    SingleInstanceLock,
    default_lock_path,
)


def test_default_lock_path_is_installation_independent() -> None:
    """The default lock path is shared rather than scoped to this repository."""
    path = default_lock_path()

    assert path.name == LOCK_FILENAME
    assert Path.cwd() not in path.parents


def test_second_lock_fails_until_first_is_released(tmp_path: Path) -> None:
    """Two guards cannot hold the same lock concurrently."""
    lock_path = tmp_path / "server.lock"
    first = SingleInstanceLock(lock_path)
    second = SingleInstanceLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(SingleInstanceError, match="already running"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_context_manager_releases_after_success_and_failure(tmp_path: Path) -> None:
    """Context-managed guards release the OS lock on both exit paths."""
    lock_path = tmp_path / "server.lock"

    with SingleInstanceLock(lock_path):
        pass

    with pytest.raises(RuntimeError, match="shutdown"):
        with SingleInstanceLock(lock_path):
            raise RuntimeError("shutdown")

    replacement = SingleInstanceLock(lock_path)
    replacement.acquire()
    replacement.release()


def test_lock_excludes_a_real_concurrent_process(tmp_path: Path) -> None:
    """A separate Python process cannot acquire a lock held by another process."""
    lock_path = tmp_path / "server.lock"
    ready_path = tmp_path / "ready"
    child_code = "\n".join(
        [
            "import time",
            "from pathlib import Path",
            "from src.open_llm_vtuber.utils.single_instance import SingleInstanceLock",
            f"lock = SingleInstanceLock(Path({str(lock_path)!r}))",
            "lock.acquire()",
            f"Path({str(ready_path)!r}).write_text('ready')",
            "time.sleep(30)",
        ]
    )
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ready_path.exists():
            if process.poll() is not None:
                break
            time.sleep(0.05)

        if not ready_path.exists():
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                "lock-holding child process did not start\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )

        contender = SingleInstanceLock(lock_path)
        with pytest.raises(SingleInstanceError, match="already running"):
            contender.acquire()
    finally:
        process.terminate()
        process.communicate(timeout=5)

    released = SingleInstanceLock(lock_path)
    released.acquire()
    released.release()


def test_posix_default_lock_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX platforms use the shared system temporary directory."""
    monkeypatch.setattr(sys, "platform", "linux")

    assert default_lock_path() == Path("/tmp") / LOCK_FILENAME


def test_acquire_and_release_are_idempotent(tmp_path: Path) -> None:
    """Repeated acquire/release calls are safe for lifecycle cleanup paths."""
    lock = SingleInstanceLock(tmp_path / "server.lock")

    lock.acquire()
    lock.acquire()
    lock.release()
    lock.release()


def test_lock_enforcement_failure_has_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operating-system lock failures are reported as enforcement failures."""

    class FailingFileLock:
        """File-lock test double that always raises an operating-system error."""

        def __init__(self, _path: str, mode: int) -> None:
            """Accept the production constructor contract."""

        def acquire(self, blocking: bool) -> None:
            """Simulate an operating-system locking failure."""
            raise OSError("permission denied")

    monkeypatch.setattr(
        "src.open_llm_vtuber.utils.single_instance.FileLock", FailingFileLock
    )

    with pytest.raises(SingleInstanceError, match="cannot enforce"):
        SingleInstanceLock(tmp_path / "server.lock").acquire()
