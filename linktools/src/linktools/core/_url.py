#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UrlFile: the user-facing download surface (spec §9.1), now backed by the unified
DownloadManager (environ.downloads) instead of a bespoke fetch/resume pipeline.

The public API is unchanged so every caller (linktools-mobile's frida server,
APK/IPA, scrcpy; cntr) keeps working:
  - environ.get_url_file(url) -> UrlFile
  - url_file.save(dest_dir=None, dest_name=None, timeout=, max_retries=,
                  validators=UrlFile.Validator | [..], **kwargs) -> str(path)
  - ``with url_file:`` (per-download lock) and ``url_file.clear()``
  - UrlFile.Validator / HashValidator / SizeValidator

Download concerns (atomic landing, hash/size validation, resume metadata,
Content-Disposition filename, gzip, retry) are owned by DownloadManager and its
transports/validators; UrlFile is now a thin adapter over them.
"""
import abc
import os
import shutil
from typing import TYPE_CHECKING, Iterable

from linktools.decorator import cached_property, timeoutable
from linktools.errors import DownloadError
from linktools.utils import guess_file_name, get_file_hash

if TYPE_CHECKING:
    from typing import Literal, Union
    from linktools.types import PathType, TimeoutType, EnvironType

    UrlFileValidatorType = Union["UrlFile.Validator", Iterable["UrlFile.Validator"]]


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
        # Per-download lock via the unified LockManager (spec §7.11/§9.9).
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
        from linktools._download import DownloadRequest

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
    from linktools._download import CompositeValidator, DownloadValidator

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
