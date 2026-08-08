#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-oriented filesystem asset backend with atomic batch operations."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from linktools.core import environ

from ..core.errors import AssetConflictError
from ..storage.files import atomic_write_bytes, read_bytes
from ._content import AssetContent, AssetContentInfo, compute_asset_etag

if TYPE_CHECKING:
    from collections.abc import Mapping

_logger = environ.get_logger("ai.asset.local")


@runtime_checkable
class AssetPathAdapter(Protocol):
    def to_disk(self, asset_path: str) -> str: ...

    def from_disk(self, disk_path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class _IdentityAssetPathAdapter:
    def to_disk(self, asset_path: str) -> str:
        return asset_path

    def from_disk(self, disk_path: str) -> str:
        return disk_path


@dataclass(frozen=True, slots=True)
class PrefixAssetPathAdapter:
    mapping: "Mapping[str, str]"

    def to_disk(self, asset_path: str) -> str:
        kind, separator, rest = asset_path.partition("/")
        return f"{self.mapping.get(kind, kind)}{separator}{rest}"

    def from_disk(self, disk_path: str) -> str:
        source_root, separator, rest = disk_path.partition("/")
        reverse = {value: key for key, value in self.mapping.items()}
        return f"{reverse.get(source_root, source_root)}{separator}{rest}"


_IDENTITY = _IdentityAssetPathAdapter()


class LocalAssetBackend:
    def __init__(
        self,
        root: "str | Path" = ".linktools",
        *,
        path_adapter: "AssetPathAdapter | None" = None,
    ) -> None:
        self.root = Path(root)
        self._adapter = path_adapter or _IDENTITY

    async def initialize_storage(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
        _logger.debug("initialized local asset storage: root=%s", self.root)

    async def get(self, path: str) -> "AssetContent | None":
        target = self._resolve(path)
        try:
            content = await asyncio.to_thread(read_bytes, target)
        except FileNotFoundError:
            return None
        return AssetContent(_info(path, content), content)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, AssetContent]":
        result: dict[str, AssetContent] = {}
        for path in paths:
            content = await self.get(path)
            if content is not None:
                result[path] = content
        return result

    async def stat(self, path: str) -> "AssetContentInfo | None":
        content = await self.get(path)
        return None if content is None else content.info

    async def list_info(self, *, kind: "str | None" = None) -> "tuple[AssetContentInfo, ...]":
        root = self.root.resolve()
        adapter = self._adapter

        def scan() -> "tuple[tuple[str, bytes], ...]":
            if not root.exists():
                return ()
            found: list[tuple[str, bytes]] = []
            for file in root.rglob("*"):
                if not file.is_file():
                    continue
                asset_path = adapter.from_disk(file.relative_to(root).as_posix())
                if kind is None or _kind_of(asset_path) == kind:
                    found.append((asset_path, file.read_bytes()))
            return tuple(sorted(found, key=lambda item: item[0]))

        return tuple(_info(path, content) for path, content in await asyncio.to_thread(scan))

    async def put(self, entry: AssetContent) -> "tuple[AssetContent, None]":
        entry.validate_etag()
        await asyncio.to_thread(atomic_write_bytes, self._resolve(entry.info.path), entry.content)
        _logger.info("stored local asset: path=%s", entry.info.path)
        return entry, None

    async def delete(self, path: str) -> None:
        await asyncio.to_thread(_unlink_if_exists, self._resolve(path))
        _logger.info("deleted local asset: path=%s", path)

    async def reset(self, entries: "tuple[AssetContent, ...]") -> None:
        for entry in entries:
            entry.validate_etag()
        keep = {entry.info.path for entry in entries}
        for info in await self.list_info():
            if info.path not in keep:
                await asyncio.to_thread(_unlink_if_exists, self._resolve(info.path))
        for entry in entries:
            await asyncio.to_thread(atomic_write_bytes, self._resolve(entry.info.path), entry.content)

    async def apply_batch(
        self,
        puts: "tuple[AssetContent, ...]",
        deletes: "tuple[str, ...]",
    ) -> None:
        for entry in puts:
            entry.validate_etag()
        put_paths = {entry.info.path for entry in puts}
        for entry in puts:
            await asyncio.to_thread(atomic_write_bytes, self._resolve(entry.info.path), entry.content)
        for path in deletes:
            if path not in put_paths:
                await asyncio.to_thread(_unlink_if_exists, self._resolve(path))

    def _resolve(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise AssetConflictError(f"invalid asset path: {path!r}")
        target = (self.root / self._adapter.to_disk(path)).resolve()
        root = self.root.resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise AssetConflictError(f"asset path escapes root: {path!r}") from error
        return target


def _info(path: str, content: bytes) -> AssetContentInfo:
    return AssetContentInfo(path, _kind_of(path), 1, compute_asset_etag(content), True)


def _kind_of(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "asset"


def _unlink_if_exists(target: Path) -> None:
    try:
        target.unlink(missing_ok=True)
    except OSError as error:
        _logger.warning("failed to delete local asset file: path=%s error=%s", target, error)


__all__ = ["AssetPathAdapter", "LocalAssetBackend", "PrefixAssetPathAdapter"]
