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

# antioch.core.locking.atomic_write_bytes() opens the parent DIRECTORY and
# fsyncs it — a POSIX durability idiom Windows refuses (os.open on a
# directory raises PermissionError), which broke `antioch auth login` with
# "Credential store ... could not be written safely". Substitute a NUL-device
# descriptor for directory opens and skip fsync on exactly those descriptors,
# so regular-file durability is untouched.
if os.name == "nt":
    _real_os_open = os.open
    _real_fsync = os.fsync
    _real_close = os.close
    _dir_fds: set = set()

    def _os_open(path, flags, mode=0o777, *, dir_fd=None):
        try:
            return _real_os_open(path, flags, mode, dir_fd=dir_fd)
        except OSError:
            try:
                is_dir = os.path.isdir(path)
            except (TypeError, OSError):
                is_dir = False
            if is_dir:
                fd = _real_os_open(os.devnull, os.O_RDWR)
                _dir_fds.add(fd)
                return fd
            raise

    def _fsync(fd):
        if fd in _dir_fds:
            return
        _real_fsync(fd)

    def _close(fd):
        _dir_fds.discard(fd)
        _real_close(fd)

    os.open = _os_open
    os.fsync = _fsync
    os.close = _close

    # antioch.cli.session rejects any credential store whose st_mode has
    # group/other bits (`st_mode & 0o077`). Windows cannot represent POSIX
    # owner-only modes — chmod maps everything onto the read-only flag and
    # stat reports 0o666 — so the check fails on every file that can exist
    # here. Mask group/other bits out of pathlib stat results instead;
    # actual secrecy comes from %USERPROFILE% NTFS ACLs, not POSIX bits.
    import pathlib

    _real_path_stat = pathlib.Path.stat

    def _path_stat(self, *, follow_symlinks=True):
        result = _real_path_stat(self, follow_symlinks=follow_symlinks)
        if not result.st_mode & 0o077:
            return result
        extra = {}
        for name in (
            "st_atime", "st_mtime", "st_ctime",
            "st_atime_ns", "st_mtime_ns", "st_ctime_ns",
            "st_file_attributes", "st_reparse_tag",
        ):
            if hasattr(result, name):
                extra[name] = getattr(result, name)
        values = (result.st_mode & ~0o077,) + tuple(result)[1:10]
        return os.stat_result(values, extra)

    pathlib.Path.stat = _path_stat

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
