"""
Minimal Windows shim for the tiny subset of POSIX `fcntl` that antioch's
`antioch.core.locking` module imports (`flock` + the LOCK_* constants).

antioch-sim has no Windows-native locking backend; this shim lets its CLI
import cleanly on Windows by mapping flock() onto msvcrt.locking(). It is a
best-effort advisory lock (single byte range at offset 0) — good enough for
one agent driving one antioch invocation at a time, NOT a verified
substitute for antioch's real crash-safety guarantees under concurrent
access. Point PYTHONPATH at this directory's parent so this shadows any
real `fcntl` (there is none on Windows, so there's nothing to conflict with).
"""

import os
import time

try:
    import msvcrt
except ImportError:  # pragma: no cover - only meaningful off Windows
    msvcrt = None

# antioch.core.locking also calls the POSIX-only os.fchmod() (open_lock,
# atomic_write_bytes) to harden lock/credential files to 0600. Windows has no
# per-descriptor POSIX permission bits, so this is patched to a no-op here —
# best-effort only; it does NOT reproduce owner-only protection on Windows.
if not hasattr(os, "fchmod"):
    os.fchmod = lambda fd, mode: None

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

_REGION = 1  # bytes


def _fd(f):
    return f if isinstance(f, int) else f.fileno()


def _ensure_byte(fd):
    try:
        if os.fstat(fd).st_size < _REGION:
            pos = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, b"\0")
            os.lseek(fd, pos, os.SEEK_SET)
    except OSError:
        pass


def flock(fd, operation):
    if msvcrt is None:
        return
    fd = _fd(fd)

    if operation & LOCK_UN:
        pos = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _REGION)
        except OSError:
            pass
        finally:
            os.lseek(fd, pos, os.SEEK_SET)
        return

    _ensure_byte(fd)
    pos = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    nonblocking = bool(operation & LOCK_NB)
    try:
        if nonblocking:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _REGION)
        else:
            # msvcrt has no blocking primitive; poll LK_NBLCK instead.
            deadline = time.monotonic() + 30
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, _REGION)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
    except OSError as exc:
        os.lseek(fd, pos, os.SEEK_SET)
        raise BlockingIOError(str(exc)) from exc
    os.lseek(fd, pos, os.SEEK_SET)
