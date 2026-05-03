#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : _download.py
@time    : 2022/05/28
@site    :
@software: PyCharm

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
import abc
import contextlib
import os
import shelve
import shutil
from typing import TYPE_CHECKING, Iterable

from linktools.decorator import cached_property, timeoutable
from linktools.rich import create_progress
from linktools.types import DownloadError, DownloadHttpError, FileCache
from linktools.utils import get_file_hash, ignore_errors, parse_header, guess_file_name, user_agent, get_hash_ident, remove_file

if TYPE_CHECKING:
    from typing import Literal
    from ._environ import BaseEnviron
    from linktools.types import PathType, Timeout, TimeoutType

    UrlFileValidatorType = Union["UrlFile.Validator", Iterable["UrlFile.Validator"]]


class UrlFile(metaclass=abc.ABCMeta):
    """UrlFile."""

    def __init__(self, environ: "BaseEnviron", url: str, is_local: bool):
        self._url = url
        self._environ = environ
        self._ident = f"{get_hash_ident(url)}_{guess_file_name(url)[-100:]}"
        self._is_local = is_local

    @cached_property(lock=True)
    def _lock(self):
        cache = FileCache(self._environ.get_temp_path("download", "cache"))
        return cache.lock(self._ident)

    @property
    def is_local(self):
        """Return whether the file is local.

        Returns:
            Any: The property value.
        """
        return self._is_local

    @timeoutable
    def save(self,
             dest_dir: "PathType" = None, dest_name: str = None,
             timeout: "TimeoutType" = None, max_retries: int = 3,
             validators: "UrlFileValidatorType" = None, **kwargs) -> str:
        """Save or download data to a target path.

        Args:
            dest_dir (PathType): Destination directory path.
            dest_name (str): Destination file name.
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.
            max_retries (int): Maximum number of retry attempts.
            validators (UrlFileValidatorType): Validators used to check downloaded files.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            str: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        try:
            self._acquire(timeout=timeout.remain)

            temp_path, temp_name = self._download(
                max_retries=max_retries,
                timeout=timeout,
                validators=validators,
                **kwargs
            )
            if not dest_dir:
                return temp_path

            # Create the destination directory first.
            if not os.path.exists(dest_dir):
                self._environ.logger.debug(f"{dest_dir} does not exist, create")
                os.makedirs(dest_dir, exist_ok=True)

            # Copy the downloaded file to the target path.
            dest_path = os.path.join(dest_dir, dest_name or temp_name)
            self._environ.logger.debug(f"Copy {temp_path} to {dest_path}")
            shutil.copy(temp_path, dest_path)

            # The cache can be cleared after the file is moved into place.
            self.clear(timeout=timeout.remain)

            return dest_path

        except DownloadError:
            raise
        except Exception as e:
            raise DownloadError(e)
        finally:
            self._release()

    @timeoutable
    def clear(self, timeout: "TimeoutType" = None):
        """Clear cached or generated data.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.
        """
        try:
            self._acquire(timeout=timeout.remain)
            self._clear()
        finally:
            self._release()

    @abc.abstractmethod
    def _download(self, *, max_retries: int, timeout: "Timeout", validators: "UrlFileValidatorType", **kwargs) -> "tuple[str, str]":
        pass

    @abc.abstractmethod
    def _clear(self):
        pass

    @timeoutable
    def _acquire(self, timeout: "TimeoutType" = None):
        self._lock.acquire(timeout=timeout.remain)

    def _release(self):
        ignore_errors(self._lock.release)

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, *args, **kwargs):
        self._release()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._url})"

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


class LocalFile(UrlFile):

    """UrlFile implementation for local filesystem paths."""
    def __init__(self, environ: "BaseEnviron", url: str):
        super().__init__(
            environ,
            os.path.abspath(os.path.expanduser(url)),
            is_local=True,
        )

    def _download(self, *, validators: "UrlFileValidatorType", **kwargs) -> "tuple[str, str]":
        src_path = self._url
        if not os.path.exists(src_path):
            raise DownloadError(f"{src_path} does not exist")
        # Validate file integrity.
        if isinstance(validators, UrlFile.Validator):
            validators.validate(self, src_path)
        elif isinstance(validators, Iterable):
            for validator in validators:
                validator.validate(self, src_path)
        return src_path, guess_file_name(src_path)

    def _clear(self):
        self._environ.logger.debug(f"{self._url} is local file, skip")


class HttpFile(UrlFile):

    """UrlFile implementation for HTTP and HTTPS resources."""
    def __init__(self, environ: "BaseEnviron", url: str):
        super().__init__(environ, url, is_local=False)
        self._root_path = self._environ.get_temp_path("download", "data", self._ident)
        self._local_path = os.path.join(self._root_path, "file")
        self._context_path = os.path.join(self._root_path, "context")

    def _download(self, *, max_retries: int, timeout: "Timeout", validators: "UrlFileValidatorType", **kwargs) -> "tuple[str, str]":
        if not os.path.exists(self._root_path):
            os.makedirs(self._root_path, exist_ok=True)

        with HttpContext(self._environ, self._context_path) as context:
            if os.path.exists(self._local_path) and context.completed:
                # The file is already downloaded, so skip downloading it again.
                self._environ.logger.debug(f"{self._local_path} downloaded, skip")

            else:
                # Initialize context metadata.
                context.url = self._url
                context.file_path = self._local_path
                context.file_size = None
                context.completed = False

                if not context.file_name:
                    context.file_name = guess_file_name(self._url)
                if not context.user_agent:
                    context.user_agent = kwargs.pop("user_agent", None) or user_agent("chrome")

                # Start downloading.
                last_error = None
                max_retries = max(max_retries or 1, 1)
                for i in range(max_retries, 0, -1):
                    try:
                        if last_error is not None:
                            self._environ.logger.warning(
                                f"Download retry {max_retries - i}, "
                                f"{last_error.__class__.__name__}: {last_error}")
                        # Download the file.
                        context.download(timeout)
                        # Validate file integrity.
                        try:
                            if isinstance(validators, UrlFile.Validator):
                                validators.validate(self, self._local_path)
                            elif isinstance(validators, Iterable):
                                for validator in validators:
                                    validator.validate(self, self._local_path)
                        except Exception:
                            # Remove the file so a failed integrity check can be retried.
                            self._environ.logger.debug(
                                f"Validate failed, remove {self._local_path}")
                            os.remove(self._local_path)
                            raise
                        # Mark the download as completed.
                        context.completed = True
                        break
                    except Exception as e:
                        last_error = e

                if not context.completed:
                    raise last_error

            return self._local_path, context.file_name

    def _clear(self):
        if not os.path.exists(self._root_path):
            self._environ.logger.debug(f"{self._root_path} does not exist, skip")
            return
        self._environ.logger.debug(f"Clear {self._root_path}")
        if os.path.exists(self._local_path):
            os.remove(self._local_path)
        if os.path.exists(self._context_path):
            os.remove(self._context_path)
        if not os.listdir(self._root_path):
            shutil.rmtree(self._root_path, ignore_errors=True)


class HttpContextVar(property):

    """Context variable descriptor for HTTP download options."""
    def __init__(self, key, default=None):
        super().__init__(
            fget=lambda o: o._db.get(key, default),
            fset=lambda o, v: o._db.__setitem__(key, v)
        )


class HttpContext:
    """Context manager for temporary HTTP download options."""
    url: "str | None" = HttpContextVar("Url")
    user_agent: "str | None" = HttpContextVar("UserAgent")
    headers: "dict | None" = HttpContextVar("Headers")
    file_path: "str | None" = HttpContextVar("FilePath")
    file_size: "int | None" = HttpContextVar("FileSize")
    file_name: "str | None" = HttpContextVar("FileName")
    completed: bool = HttpContextVar("IsCompleted", False)
    content_encoding: str = HttpContextVar("ContentEncoding", "")

    def __init__(self, environ: "BaseEnviron", path: str):
        self._environ = environ
        self._db = shelve.open(path)

    def __enter__(self):
        self._db.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self._db.__exit__(*args, **kwargs)

    def download(self, timeout: "TimeoutType"):
        """Download the configured URL into the temporary file path.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        self._environ.logger.debug(f"Download file to temp path {self.file_path}")

        initial = 0
        # Resume the previous download if the file already exists.
        if os.path.exists(self.file_path):
            size = os.path.getsize(self.file_path)
            self._environ.logger.debug(f"{size} bytes downloaded, continue")
            initial = size

        self.headers = {
            "User-Agent": self.user_agent,
            "Range": f"bytes={initial}-",
        }

        try:
            import requests
            fn = self._download_with_requests
        except ModuleNotFoundError:
            fn = self._download_with_urllib

        with create_progress() as progress:
            task_id = progress.add_task(self.file_name, total=None)
            progress.advance(task_id, initial)

            with open(self.file_path, "ab") as fp:
                offset = 0
                for data in fn(timeout.remain):
                    advance = len(data)
                    offset += advance
                    fp.write(data)
                    progress.update(
                        task_id,
                        advance=advance,
                        description=self.file_name
                    )
                    if self.file_size is not None:
                        progress.update(
                            task_id,
                            total=initial + self.file_size
                        )

            if self.file_size is not None and self.file_size > offset:
                raise DownloadError(
                    f"download size {initial + self.file_size} bytes was expected, "
                    f"got {initial + offset} bytes"
                )

            if os.path.getsize(self.file_path) == 0:
                raise DownloadError(f"download {self.url} error")

            self._decompress_if_needed()

    def _decompress_if_needed(self):
        if self.content_encoding.lower() == "gzip":
            import gzip
            import shutil
            self._environ.logger.debug(f"Decompressing gzip file {self.file_path}")
            tmp_path = self.file_path + ".tmp"
            remove_file(tmp_path)
            try:
                with gzip.open(self.file_path, "rb") as gz, open(tmp_path, "wb") as out:
                    shutil.copyfileobj(gz, out)
            except:
                remove_file(self.file_path)
                remove_file(tmp_path)
                raise
            os.replace(tmp_path, self.file_path)

    def _download_with_requests(self, timeout: float):
        import requests
        from requests import HTTPError

        bs = 1024 * 8

        with requests.get(self.url, headers=self.headers, stream=True, timeout=timeout) as resp:

            try:
                resp.raise_for_status()
            except HTTPError as e:
                raise DownloadHttpError(resp.status_code, e)

            if "Content-Length" in resp.headers:
                self.file_size = int(resp.headers.get("Content-Length"))
            if "Content-Disposition" in resp.headers:
                _, params = parse_header(resp.headers["Content-Disposition"])
                if "filename" in params:
                    self.file_name = params["filename"]
            self.content_encoding = resp.headers.get("Content-Encoding", "")

            resp.raw.decode_content = False
            for chunk in resp.iter_content(bs):
                if chunk:
                    yield chunk

    def _download_with_urllib(self, timeout: float):
        from urllib.request import urlopen, Request
        from urllib.error import HTTPError

        bs = 1024 * 8

        url = Request(self.url, headers=self.headers)

        try:
            resp = urlopen(url=url, timeout=timeout)
        except HTTPError as e:
            raise DownloadHttpError(e.code, e)

        with contextlib.closing(resp) as fp:

            headers = fp.info()
            if "Content-Length" in headers:
                self.file_size = int(headers["Content-Length"])
            if "Content-Disposition" in headers:
                _, params = parse_header(headers["Content-Disposition"])
                if "filename" in params:
                    self.file_name = params["filename"]
            self.content_encoding = headers.get("Content-Encoding", "")

            while True:
                chunk = fp.read(bs)
                if not chunk:
                    break
                yield chunk
