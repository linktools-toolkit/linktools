#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified download infrastructure (spec §9).

Standalone module (PR 09): not yet wired into ``environ.downloads`` and does not
replace ``core/_url.py`` -- that migration is a follow-up. It composes the
persistence foundation built earlier: LockManager for per-content serialization
(§9.9), CacheStore for resume metadata (§9.9), utils.atomic_replace /
verify_file for atomic landing (§9.4) and integrity (§9.7).

Flow (spec §9.4)::

    per-content lock -> if destination validates, reuse -> else fetch .part ->
    validate -> flush/fsync -> os.replace(destination) -> store resume metadata

Resume (§9.5): when a previous fetch left a ``.part`` and the server supports
it, an HTTP transport re-issues a Range request guarded by If-Range so a changed
file is detected (server returns 200 -> restart, never append a full response).
"""

import os
import shutil
import time
import urllib.error as _urlerror
import urllib.request as _urlrequest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .errors import DownloadError, DownloadHttpError
from . import utils

__all__ = [
    "DownloadRequest", "DownloadResult", "DownloadProgress",
    "DownloadValidator", "SizeValidator", "HashValidator", "CompositeValidator",
    "DownloadTransport", "FileTransport", "HttpTransport", "DownloadManager",
]

PathLike = Any  # str | os.PathLike
_CHUNK = 1 << 16  # 64 KiB


# --------------------------------------------------------------------------- #
# Request / result / progress
# --------------------------------------------------------------------------- #

class DownloadRequest(object):
    """A download target (spec §9.2)."""

    def __init__(self, url, destination, sha256=None, size=None,
                 timeout=None, resume=True, headers=None, max_retries=3):
        # type: (str, PathLike, Optional[str], Optional[int], Optional[float], bool, Optional[Dict[str, str]], int) -> None
        self.url = url
        self.destination = Path(destination)
        self.sha256 = sha256
        self.size = size
        self.timeout = timeout
        self.resume = resume
        self.headers = dict(headers or {})
        self.max_retries = max_retries

    @property
    def lock_key(self):
        # type: () -> str
        # Prefer a content hash (stable across URL redirects/mirrors); fall back
        # to a hash of the URL.
        return (self.sha256 or utils.get_hash(self.url, "sha256"))[:64]


class DownloadResult(object):
    def __init__(self, path, size, from_cache):
        # type: (Path, int, bool) -> None
        self.path = path
        self.size = size
        self.from_cache = from_cache


class DownloadProgress(object):
    def __init__(self, downloaded, total):
        # type: (int, Optional[int]) -> None
        self.downloaded = downloaded
        self.total = total


# --------------------------------------------------------------------------- #
# Validators (§9.7)
# --------------------------------------------------------------------------- #

class DownloadValidator(object):
    def validate(self, path):
        # type: (PathLike) -> None
        raise NotImplementedError


class SizeValidator(DownloadValidator):
    def __init__(self, size):
        # type: (int) -> None
        self.size = int(size)

    def validate(self, path):
        actual = os.path.getsize(path)
        if actual != self.size:
            raise DownloadError("size mismatch: expected %d, got %d" % (self.size, actual))


class HashValidator(DownloadValidator):
    def __init__(self, digest, algorithm="sha256"):
        # type: (str, str) -> None
        self.digest = digest.lower()
        self.algorithm = algorithm

    def validate(self, path):
        if not utils.verify_file(path, self.digest, algorithm=self.algorithm):
            raise DownloadError("%s hash mismatch" % self.algorithm)


class CompositeValidator(DownloadValidator):
    def __init__(self, validators):
        # type: (List[DownloadValidator]) -> None
        self._validators = list(validators)

    def validate(self, path):
        for v in self._validators:
            v.validate(path)


# --------------------------------------------------------------------------- #
# Transports (§9.3)
# --------------------------------------------------------------------------- #

class DownloadTransport(object):
    def fetch(self, request, part, on_progress=None, meta=None):
        # type: (DownloadRequest, PathLike, Optional[Callable[[DownloadProgress], None]], Optional[Dict[str, Any]]) -> None
        raise NotImplementedError


class FileTransport(DownloadTransport):
    """Copy a local ``file://`` or bare-path source (no resume needed)."""

    def fetch(self, request, part, on_progress=None, meta=None):
        src = request.url
        if src.startswith("file://"):
            src = src[len("file://"):]
        Path(part).parent.mkdir(parents=True, exist_ok=True)
        total = os.path.getsize(src)
        with open(src, "rb") as r, open(part, "wb") as w:
            while True:
                chunk = r.read(_CHUNK)
                if not chunk:
                    break
                w.write(chunk)
                if on_progress is not None:
                    on_progress(DownloadProgress(downloaded=w.tell(), total=total))
            w.flush()
            os.fsync(w.fileno())
        if meta is not None:
            meta["url"] = request.url


class HttpTransport(DownloadTransport):
    """HTTP fetch with optional resume (Range + If-Range, spec §9.5)."""

    def __init__(self, headers=None):
        # type: (Optional[Dict[str, str]]) -> None
        self._base_headers = dict(headers or {})

    def fetch(self, request, part, on_progress=None, meta=None):
        part = Path(part)
        part.parent.mkdir(parents=True, exist_ok=True)

        # v4 §8.3: retry loop. If Content-Range validation fails (missing,
        # parse-failed, or start != have), close the response, delete .part,
        # and re-request without Range. At most 2 attempts (resume + 1 restart).
        for _attempt in range(2):
            headers = dict(self._base_headers)
            headers.update(request.headers)
            have = part.stat().st_size if (request.resume and part.exists()) else 0
            if have > 0:
                # Ask to continue; If-Range guards against a changed remote file.
                headers["Range"] = "bytes=%d-" % have
                if meta is not None:
                    etag = meta.get("etag")
                    last_mod = meta.get("last_modified")
                    if etag:
                        headers["If-Range"] = etag
                    elif last_mod:
                        headers["If-Range"] = last_mod

            req = _urlrequest.Request(request.url, headers=headers)
            try:
                response = _urlrequest.urlopen(req, timeout=request.timeout)
            except _urlerror.HTTPError as exc:
                # §7.3: 416 Range Not Satisfiable -- the .part may already be complete.
                if exc.code == 416 and have > 0 and part.exists():
                    part_size = part.stat().st_size
                    expected = (meta or {}).get("size")
                    if expected is not None and part_size >= expected:
                        return  # Part is complete; nothing more to download.
                    _discard(part)
                    raise DownloadError("server returned 416 and part is incomplete; will restart")
                raise DownloadHttpError(exc.code, str(exc))
            except _urlerror.URLError as exc:
                raise DownloadError("transport error for %s: %s" % (request.url, exc))

            try:
                code = response.getcode()
                appending = have > 0 and code == 206
                # v4 §8.2: STRICT Content-Range validation on 206.
                if appending:
                    cr = response.headers.get("Content-Range", "")
                    restart_needed = False
                    if not cr:
                        restart_needed = True
                    else:
                        try:
                            range_spec = cr.strip().split(" ")[-1]
                            start_str = range_spec.split("-")[0]
                            start = int(start_str)
                            if start != have:
                                restart_needed = True
                        except (ValueError, IndexError):
                            restart_needed = True
                    if restart_needed:
                        response.close()
                        _discard(part)
                        continue  # restart loop without Range

                mode = "ab" if appending else "wb"
                written = have if appending else 0
                total = response.length  # may be None
                content_encoding = ""
                if meta is not None:
                    meta["url"] = request.url
                    etag = response.headers.get("ETag")
                    last_mod = response.headers.get("Last-Modified")
                    if etag:
                        meta["etag"] = etag
                    if last_mod:
                        meta["last_modified"] = last_mod
                    disposition = response.headers.get("Content-Disposition")
                    if disposition:
                        _, params = utils.parse_header(disposition)
                        if "filename" in params:
                            meta["filename"] = params["filename"]
                content_encoding = response.headers.get("Content-Encoding", "") or ""
                with open(part, mode) as handle:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if on_progress is not None:
                            on_progress(DownloadProgress(downloaded=written, total=total))
                    handle.flush()
                    os.fsync(handle.fileno())
                if content_encoding.lower() == "gzip":
                    _gunzip_inplace(part)
                return  # success
            finally:
                response.close()

        # If we get here, the loop exhausted without returning (shouldn't happen
        # normally — the second attempt has have=0 so no Range is sent).
        raise DownloadError("download failed after Content-Range restart")


# --------------------------------------------------------------------------- #
# Manager (§9.2, §9.4)
# --------------------------------------------------------------------------- #

class DownloadManager(object):
    def __init__(self, environ):
        # type: (Any) -> None
        self._environ = environ

    # -- internals ---------------------------------------------------------

    def _build_validator(self, request):
        # type: (DownloadRequest) -> Optional[DownloadValidator]
        validators = []  # type: List[DownloadValidator]
        if request.size is not None:
            validators.append(SizeValidator(request.size))
        if request.sha256:
            validators.append(HashValidator(request.sha256))
        if not validators:
            return None
        return validators[0] if len(validators) == 1 else CompositeValidator(validators)

    def _resume_namespace(self):
        return self._environ.cache.namespace("download:resume")

    def _default_transport(self, request):
        url = request.url
        if url.startswith("http://") or url.startswith("https://"):
            return HttpTransport()
        return FileTransport()

    # -- public ------------------------------------------------------------

    def download(self, request, transport=None, on_progress=None):
        # type: (DownloadRequest, Optional[DownloadTransport], Optional[Callable[[DownloadProgress], None]]) -> DownloadResult
        transport = transport or self._default_transport(request)
        destination = request.destination
        validator = self._build_validator(request)
        resume_ns = self._resume_namespace()

        with self._environ.locks.process_lock("download:" + request.lock_key):
            # §9.4: an existing file that validates is reused as-is.
            if destination.exists() and validator is not None:
                try:
                    validator.validate(destination)
                    return DownloadResult(destination, destination.stat().st_size, from_cache=True)
                except DownloadError:
                    pass  # stale -- fall through and re-download

            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.parent / (destination.name + ".part")
            meta = dict(resume_ns.get(request.lock_key, {}) or {})  # type: Dict[str, Any]

            # §9.6 retry: transport/network errors are retried with exponential
            # backoff (cap 8s); validation failures are not retried here (a
            # hash-mismatch retry-once refinement is a follow-up).
            attempts = max(1, int(request.max_retries or 1))
            last_error = None  # type: Optional[DownloadError]
            for attempt in range(attempts):
                try:
                    transport.fetch(request, part, on_progress=on_progress, meta=meta)
                    last_error = None
                    break
                except DownloadError as exc:
                    last_error = exc
                    if attempt + 1 >= attempts:
                        break
                    time.sleep(min(2 ** attempt, 8))
            if last_error is not None:
                # Network failure: keep .part for a future resume.
                raise last_error

            # §7.4: hash-mismatch retry-once. If validation fails, discard the
            # .part and re-download from scratch exactly once; a second failure
            # raises DownloadError (not retried indefinitely).
            if validator is not None:
                try:
                    validator.validate(part)
                except DownloadError:
                    _discard(part)
                    transport.fetch(request, part, on_progress=on_progress, meta=meta)
                    try:
                        validator.validate(part)
                    except DownloadError:
                        _discard(part)
                        raise

            final_size = os.path.getsize(part)

            final_size = os.path.getsize(part)
            utils.atomic_replace(part, destination)
            meta["size"] = final_size
            resume_ns.set(request.lock_key, meta)
            return DownloadResult(destination, final_size, from_cache=False)


def _discard(path):
    # type: (PathLike) -> None
    try:
        os.remove(str(path))
    except FileNotFoundError:
        pass


def _gunzip_inplace(path):
    # type: (PathLike) -> None
    """Decompress a gzip file in place (temp -> os.replace)."""
    import gzip
    import shutil as _shutil
    path = str(path)
    tmp = path + ".gunzip"
    try:
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            _shutil.copyfileobj(src, dst)
        os.replace(tmp, path)
    except BaseException:
        _discard(tmp)
        raise
