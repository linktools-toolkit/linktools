"""Download subsystem: DownloadManager + UrlFile thin adapter (compact-layout spec §2.2).

Merged from the former linktools/_download.py and core/_url.py. UrlFile is a
thin adapter over DownloadManager (atomic landing / resume / hash validation).
Behaviour unchanged.
"""

import abc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .. import utils
from ..decorator import cached_property, timeoutable
from ..errors import DownloadError, DownloadHttpError
from ..utils import get_file_hash, guess_file_name

if TYPE_CHECKING:
    from typing import Callable, Literal, Union
    from ..types import EnvironType, PathType, TimeoutType

    # UrlFile.Validator(s) accepted by save(validators=...).
    UrlFileValidatorType = Union["UrlFile.Validator", Iterable["UrlFile.Validator"]]

__all__ = ['DownloadRequest','DownloadResult','DownloadProgress','DownloadValidator','SizeValidator','HashValidator','CompositeValidator','DownloadTransport','FileTransport','HttpTransport','DownloadManager','UrlFile','HttpFile','LocalFile']


PathLike = Any  # str | os.PathLike
_CHUNK = 1 << 16  # 64 KiB


# ---------------------------------------------------------------------------
# Request / result / progress
# ---------------------------------------------------------------------------

class DownloadRequest(object):
    """A download target (spec §9.2)."""

    def __init__(self, url: str, destination: "PathLike", sha256: "str | None" = None, size: "int | None" = None,
                 timeout: "float | None" = None, resume: bool = True, headers: "dict[str, str] | None" = None, max_retries: int = 3) -> None:
        self.url = url
        self.destination = Path(destination)
        self.sha256 = sha256
        self.size = size
        self.timeout = timeout
        self.resume = resume
        self.headers = dict(headers or {})
        self.max_retries = max_retries

    @property
    def lock_key(self) -> str:
        # Prefer a content hash (stable across URL redirects/mirrors); fall back
        # to a hash of the URL.
        return (self.sha256 or utils.get_hash(self.url, "sha256"))[:64]


class DownloadResult(object):
    def __init__(self, path: "Path", size: int, from_cache: bool) -> None:
        self.path = path
        self.size = size
        self.from_cache = from_cache


class DownloadProgress(object):
    def __init__(self, downloaded: int, total: "int | None") -> None:
        self.downloaded = downloaded
        self.total = total


# ---------------------------------------------------------------------------
# Validators (
# ---------------------------------------------------------------------------

class DownloadValidator(object):
    def validate(self, path: "PathLike") -> None:
        raise NotImplementedError


class SizeValidator(DownloadValidator):
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def validate(self, path):
        actual = os.path.getsize(path)
        if actual != self.size:
            raise DownloadError("size mismatch: expected %d, got %d" % (self.size, actual))


class HashValidator(DownloadValidator):
    def __init__(self, digest: str, algorithm: str = "sha256") -> None:
        self.digest = digest.lower()
        self.algorithm = algorithm

    def validate(self, path):
        if not utils.verify_file(path, self.digest, algorithm=self.algorithm):
            raise DownloadError("%s hash mismatch" % self.algorithm)


class CompositeValidator(DownloadValidator):
    def __init__(self, validators: "list[DownloadValidator]") -> None:
        self._validators = list(validators)

    def validate(self, path):
        for v in self._validators:
            v.validate(path)


# ---------------------------------------------------------------------------
# Transports (
# ---------------------------------------------------------------------------

class DownloadTransport(object):
    def fetch(self, request: "DownloadRequest", part: "PathLike", on_progress: "Callable[[DownloadProgress], None] | None" = None, meta: "dict[str, Any] | None" = None) -> None:
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

    def __init__(self, headers: "dict[str, str] | None" = None) -> None:
        self._base_headers = dict(headers or {})

    def fetch(self, request, part, on_progress=None, meta=None):
        # Imported lazily so `import linktools.core` does not pull urllib (~19ms);
        # only an actual HTTP download pays for it.
        import urllib.error as _urlerror
        import urllib.request as _urlrequest

        part = Path(part)
        part.parent.mkdir(parents=True, exist_ok=True)

        # Retry loop. If Content-Range validation fails (missing,
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
                # 416 Range Not Satisfiable: check if .part is already complete.
                if exc.code == 416 and have > 0 and part.exists():
                    part_size = part.stat().st_size
                    expected = (meta or {}).get("size")
                    if expected is not None and part_size >= expected:
                        return  # Part is complete; nothing more to download.
                    # incomplete + 416 → restart without Range (not throw).
                    _discard(part)
                    continue  # restart loop; have will be 0 next iteration
                raise DownloadHttpError(exc.code, str(exc))
            except _urlerror.URLError as exc:
                raise DownloadError("transport error for %s: %s" % (request.url, exc))

            try:
                code = response.getcode()
                appending = have > 0 and code == 206
                # STRICT Content-Range validation on 206.
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


# ---------------------------------------------------------------------------
# Manager (
# ---------------------------------------------------------------------------

class DownloadManager(object):
    def __init__(self, environ: "Any") -> None:
        self._environ = environ

    # -- internals ---------------------------------------------------------

    def _build_validator(self, request: "DownloadRequest") -> "DownloadValidator | None":
        validators: "list[DownloadValidator]" = []
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

    def download(self, request: "DownloadRequest", transport: "DownloadTransport | None" = None, on_progress: "Callable[[DownloadProgress], None] | None" = None) -> "DownloadResult":
        transport = transport or self._default_transport(request)
        destination = request.destination
        validator = self._build_validator(request)
        resume_ns = self._resume_namespace()

        with self._environ.locks.process_lock("download:" + request.lock_key):
            #  an existing file that validates is reused as-is.
            if destination.exists() and validator is not None:
                try:
                    validator.validate(destination)
                    return DownloadResult(destination, destination.stat().st_size, from_cache=True)
                except DownloadError:
                    pass  # stale -- fall through and re-download

            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.parent / (destination.name + ".part")
            meta: "dict[str, Any]" = dict(resume_ns.get(request.lock_key, {}) or {})

            #  retry: transport/network errors are retried with exponential
            # backoff (cap 8s); validation failures are not retried here (a
            # hash-mismatch retry-once refinement is a follow-up).
            attempts = max(1, int(request.max_retries or 1))
            last_error: "DownloadError | None" = None
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

            #  hash-mismatch retry-once. If validation fails, discard the
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
            utils.atomic_replace(part, destination)
            meta["size"] = final_size
            resume_ns.set(request.lock_key, meta)
            return DownloadResult(destination, final_size, from_cache=False)


def _discard(path: "PathLike") -> None:
    try:
        os.remove(str(path))
    except FileNotFoundError:
        pass


def _gunzip_inplace(path: "PathLike") -> None:
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


class UrlFile(metaclass=abc.ABCMeta):
    """A local or remote file obtainable via the environment's DownloadManager."""

    def __init__(self, environ: "EnvironType", url: str, is_local: bool):
        self._url = url
        self._environ = environ
        self._is_local = is_local

    @property
    def is_local(self):
        """Return whether the file is local."""
        return self._is_local

    @cached_property(lock=True)
    def _lock(self):
        # Per-download lock via the unified LockManager (spec 
        from linktools.utils import get_hash_ident
        ident = "%s_%s" % (get_hash_ident(self._url), guess_file_name(self._url)[-100:])
        return self._environ.locks.process_lock("download:" + ident)

    # -- the public save entry point (signature unchanged) ----------------

    @timeoutable
    def save(self,
             dest_dir: "PathType" = None, dest_name: str = None,
             timeout: "TimeoutType" = None, max_retries: int = 3,
             validators: "UrlFileValidatorType" = None, **kwargs) -> str:
        """Download (or copy) the file and return its local path.

        With ``dest_dir`` the file lands at ``dest_dir/(dest_name or url name)``;
        without it, the file lands in the environment's downloads area and that
        path is returned.
        """
        try:
            self._lock.acquire(timeout=timeout.remaining if timeout is not None else None)
            try:
                validator = _adapt_validators(validators)
                # Local file with no destination: return it in place (legacy
                # LocalFile returned the source path without copying).
                if self._is_local and dest_dir is None:
                    src = self._url
                    if not os.path.exists(src):
                        raise DownloadError("%s does not exist" % src)
                    if validator is not None:
                        validator.validate(src)
                    return src

                filename = dest_name or self._filename()
                if dest_dir:
                    dest_dir = os.fspath(dest_dir)
                    os.makedirs(dest_dir, exist_ok=True)
                    destination = os.path.join(dest_dir, filename)
                else:
                    destination = str(self._environ.paths.downloads / self._ident() / filename)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)

                request = DownloadRequest(
                    url=self._url,
                    destination=destination,
                    timeout=timeout.remaining if timeout is not None else None,
                    max_retries=max_retries,
                    headers=kwargs.get("headers") or ({"User-Agent": kwargs["user_agent"]} if "user_agent" in kwargs else None),
                )
                # DownloadManager owns atomic landing/resume/gzip/retry; UrlFile
                # validators are an additional layer applied to the landed file.
                self._environ.downloads.download(request, on_progress=None)
                if validator is not None:
                    validator.validate(destination)
                return destination
            finally:
                from linktools.utils import ignore_errors
                ignore_errors(self._lock.release)
        except DownloadError:
            raise
        except Exception as e:
            raise DownloadError(e)

    @timeoutable
    def clear(self, timeout: "TimeoutType" = None):
        """Clear cached download data for this URL."""
        try:
            self._lock.acquire(timeout=timeout.remaining if timeout is not None else None)
            try:
                self._clear()
            finally:
                from linktools.utils import ignore_errors
                ignore_errors(self._lock.release)
        except DownloadError:
            raise

    # -- hooks for subclasses ---------------------------------------------

    def _filename(self) -> str:
        return guess_file_name(self._url)

    def _ident(self) -> str:
        from linktools.utils import get_hash_ident
        return get_hash_ident(self._url)

    def _clear(self):
        # Remove the per-ident downloads dir + resume metadata (best effort).
        from linktools.utils import remove_file
        ident_dir = self._environ.paths.downloads / self._ident()
        if ident_dir.exists():
            remove_file(ident_dir)
        try:
            self._environ.cache.namespace("download:resume").delete(self._resume_key())
        except Exception:
            pass

    def _resume_key(self) -> str:
        from linktools.utils import get_hash
        return get_hash(self._url, "sha256")[:64]

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args, **kwargs):
        from linktools.utils import ignore_errors
        ignore_errors(self._lock.release)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._url})"

    # -- validators (public API preserved) --------------------------------

    class Validator(abc.ABC):
        """Validate downloaded file content."""
        @abc.abstractmethod
        def validate(self, file: "UrlFile", path: str):
            pass

    class HashValidator(Validator):
        """Validate a downloaded file hash."""
        def __init__(self, algorithm: "Literal['md5', 'sha1', 'sha256']", hash: str):
            self._algorithm = algorithm
            self._hash = hash

        def validate(self, file: "UrlFile", path: str):
            if get_file_hash(path, self._algorithm) != self._hash:
                raise DownloadError(f"{file} {self._algorithm} hash does not match {self._hash}")

    class SizeValidator(Validator):
        """Validate a downloaded file size."""
        def __init__(self, size: int):
            self._size = size

        def validate(self, file: "UrlFile", path: str):
            if os.path.getsize(path) != self._size:
                raise DownloadError(f"{file} size does not match {self._size}")


def _adapt_validators(validators):
    """Adapt UrlFile.Validator(s) to a DownloadManager-style validator (validate(path))."""
    if validators is None:
        return None

    single = validators if isinstance(validators, UrlFile.Validator) else None
    many = list(validators) if isinstance(validators, Iterable) and single is None else None

    class _Adapted(DownloadValidator):
        def __init__(self, inner):
            self._inner = inner  # UrlFile.Validator or list thereof

        def validate(self, path):
            # UrlFile validators take (file, path); pass a lightweight stand-in.
            sentinel = type("_F", (), {"__repr__": lambda self: "UrlFile"})()
            if isinstance(self._inner, UrlFile.Validator):
                self._inner.validate(sentinel, str(path))
            else:
                for v in self._inner:
                    v.validate(sentinel, str(path))

    if single is not None:
        return _Adapted(single)
    if many:
        return _Adapted(many)
    return None


class LocalFile(UrlFile):
    """A local filesystem path exposed as a UrlFile."""
    def __init__(self, environ: "EnvironType", url: str):
        super().__init__(environ, os.path.abspath(os.path.expanduser(url)), is_local=True)

    def _filename(self) -> str:
        return os.path.basename(self._url) or guess_file_name(self._url)


class HttpFile(UrlFile):
    """An HTTP/HTTPS resource exposed as a UrlFile."""
    def __init__(self, environ: "EnvironType", url: str):
        super().__init__(environ, url, is_local=False)
