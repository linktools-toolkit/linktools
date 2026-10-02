#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disposable local projections of pinned Asset bytes."""

import asyncio
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType, TracebackType
from typing import TypeVar

from ..errors import AIError, ErrorCode
from ._domain import AssetVersionRef
from ._store import AssetStoreReader

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class MaterializedAssets:
    """Local paths valid until their owning AssetMaterializer is closed."""

    root: Path
    files: Mapping[str, Path] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))


class AssetMaterializer:
    """Own private temporary directories without retaining Asset history.

    Consumers must stop processes using the returned paths before closing this
    owner. Every materialization has a separate directory, including empty
    packages, and can serve as a process working directory.
    """

    def __init__(self) -> None:
        self._roots: set[Path] = set()
        self._lock = asyncio.Lock()
        self._closing = False

    async def __aenter__(self) -> "AssetMaterializer":
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.close()
        except BaseException as cleanup_error:
            if exc_value is not None and cleanup_error is not exc_value:
                raise exc_value from cleanup_error
            raise

    async def materialize(
        self,
        reader: AssetStoreReader,
        files: Mapping[str, AssetVersionRef],
        *,
        executable_bits: Mapping[str, int] | None = None,
    ) -> MaterializedAssets:
        """Project verified versions using caller-owned relative file paths.

        Files are readable by their owner and have only the explicitly supplied
        executable bits. No paths, file modes, or versions are inferred from the
        current backend or from AssetKey identifiers.
        """
        if not isinstance(reader, AssetStoreReader):
            raise TypeError("reader must provide AssetStoreReader operations")
        ordered, modes = _validate_files(files, executable_bits)
        async with self._lock:
            self._ensure_open()
            refs = tuple(ref for _relative, ref in ordered)
            values = await reader.read_versions(refs)
            self._ensure_open()
            try:
                root = Path(tempfile.mkdtemp(prefix="linktools-assets-")).absolute()
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            self._roots.add(root)
            try:
                contents = tuple(
                    (relative, value)
                    for (relative, _ref), value in zip(ordered, values, strict=True)
                )
                local = await _complete_task(asyncio.create_task(
                    asyncio.to_thread(_write_files, root, contents, modes)
                ))
                self._ensure_open()
            except BaseException as primary_error:
                try:
                    await self._remove_root(root)
                except BaseException as cleanup_error:
                    raise primary_error from cleanup_error
                raise
            return MaterializedAssets(root, local)

    async def close(self) -> None:
        """Remove all owned projections before propagating caller cancellation.

        Failed removals remain owned and can be retried by calling close again.
        New materializations are rejected once closing has begun.
        """
        self._closing = True
        await _complete_task(asyncio.create_task(self._close_roots()))

    async def _close_roots(self) -> None:
        async with self._lock:
            failure: BaseException | None = None
            for root in tuple(self._roots):
                try:
                    await self._remove_root(root)
                except BaseException as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure

    async def _remove_root(self, root: Path) -> None:
        await _complete_task(asyncio.create_task(asyncio.to_thread(_remove_tree, root)))
        self._roots.discard(root)

    def _ensure_open(self) -> None:
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)


def _validate_files(
    files: Mapping[str, AssetVersionRef],
    executable_bits: Mapping[str, int] | None,
) -> tuple[tuple[tuple[str, AssetVersionRef], ...], dict[str, int]]:
    if not isinstance(files, Mapping):
        raise TypeError("files must be a mapping")
    selected = dict(files)
    for relative, ref in selected.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("materialized file path is invalid")
        try:
            relative.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("materialized file path is invalid") from error
        parts = relative.split("/")
        if (
            "\\" in relative
            or "\x00" in relative
            or ":" in relative
            or any(part in {"", ".", ".."} for part in parts)
            or (os.name == "nt" and any(
                part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
                for part in parts
            ))
        ):
            raise ValueError("materialized file path is invalid")
        if not isinstance(ref, AssetVersionRef):
            raise TypeError("files must contain AssetVersionRef values")
        if any("/".join(parts[:end]) in selected for end in range(1, len(parts))):
            raise ValueError("materialized file paths overlap")
    if executable_bits is None:
        modes = dict.fromkeys(selected, 0)
    else:
        if not isinstance(executable_bits, Mapping):
            raise TypeError("executable_bits must be a mapping")
        modes = dict(executable_bits)
        if set(modes) != set(selected) or any(
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or mode & ~0o111
            for mode in modes.values()
        ):
            raise ValueError("materialized executable bits are invalid")
    return tuple(sorted(selected.items())), modes


def _write_files(
    root: Path,
    files: tuple[tuple[str, bytes], ...],
    modes: Mapping[str, int],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    try:
        for relative, data in files:
            path = root.joinpath(*relative.split("/"))
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o400 | modes[relative])
            result[relative] = path
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    return result


def _remove_tree(root: Path) -> None:
    try:
        if os.name == "nt":
            shutil.rmtree(root, onerror=_remove_readonly_file)
        else:
            shutil.rmtree(root)
    except FileNotFoundError:
        if root.exists():
            raise AIError(
                ErrorCode.STORAGE_UNAVAILABLE,
                safe_details={"phase": "asset_materialization_cleanup"},
            ) from None
    except OSError as error:
        raise AIError(
            ErrorCode.STORAGE_UNAVAILABLE,
            safe_details={"phase": "asset_materialization_cleanup"},
        ) from error


def _remove_readonly_file(
    operation: Callable[[str], object],
    path: str,
    error_info: tuple[type[BaseException], BaseException, TracebackType | None],
) -> None:
    # Windows protects read-only files from unlink, unlike POSIX. Never walk
    # separately to adjust modes: a child junction can lead outside this tree.
    if (
        operation not in (os.unlink, os.remove)
        or not isinstance(error_info[1], PermissionError)
        or not stat.S_ISREG(os.lstat(path).st_mode)
    ):
        raise error_info[1]
    os.chmod(path, 0o600)
    operation(path)


async def _complete_task(task: "asyncio.Task[_ResultT]") -> _ResultT:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
        except BaseException:
            break
    if cancellation is not None:
        try:
            task.result()
        except BaseException as error:
            raise cancellation from error
        raise cancellation
    return task.result()


__all__ = ["AssetMaterializer", "MaterializedAssets"]
