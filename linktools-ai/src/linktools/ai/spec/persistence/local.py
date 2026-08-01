#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plain filesystem-directory spec backend.

A spec at path ``agent/writer.md`` is stored as the file ``<root>/agent/writer.md``
whose body is the spec content. The layout mirrors the directory tree directly --
no manifest, no object store, no revision counter.

Metadata is derived from the file at read time:

- ``path``  -- the file's path relative to ``<root>`` (POSIX-formatted),
- ``etag``  -- ``sha256(content)``,
- ``kind``  -- the first ``/``-delimited segment of the path (``agent/writer.md`` →
  ``agent``); a path with no ``/`` defaults to ``"spec"``,
- ``version`` -- always 1 (the directory layout carries no version history),
- ``active`` -- always True.

Because the backend does not implement :class:`StorageMetadataBackend`,
:class:`StorageComposition` selects the ``ALWAYS`` refresh policy: every refresh
calls :meth:`list_info`, which walks the tree. There is no incremental PATCH.
"""


import asyncio
from pathlib import Path

from ...errors import SpecConflictError
from ...storage.local.files import atomic_write_bytes, read_bytes
from ..document import SpecDocument, SpecDocumentInfo, compute_spec_etag


class LocalSpecBackend:
    """Spec persistence over a plain directory tree."""

    def __init__(self, root: "str | Path" = ".linktools") -> None:
        self.root = Path(root)

    # ---- path safety ---------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Map a spec path to its filesystem path under ``<root>``, rejecting
        any path that escapes ``<root>`` (absolute or ``..`` traversal)."""
        target = (self.root / path).resolve()
        root = self.root.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SpecConflictError(f"spec path escapes root: {path!r}") from exc
        return target

    # ---- lifecycle -----------------------------------------------------

    async def initialize_storage(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    # ---- reader --------------------------------------------------------

    async def get(self, path: str) -> "SpecDocument | None":
        target = self._resolve(path)
        try:
            content = await asyncio.to_thread(read_bytes, target)
        except FileNotFoundError:
            return None
        return SpecDocument(_info(path, content), content)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, SpecDocument]":
        result: "dict[str, SpecDocument]" = {}
        for path in paths:
            doc = await self.get(path)
            if doc is not None:
                result[path] = doc
        return result

    async def stat(self, path: str) -> "SpecDocumentInfo | None":
        target = self._resolve(path)
        try:
            content = await asyncio.to_thread(read_bytes, target)
        except FileNotFoundError:
            return None
        return _info(path, content)

    async def list_info(self, *, kind: "str | None" = None) -> "tuple[SpecDocumentInfo, ...]":
        """Walk the tree and derive info for every file under ``<root>``. The
        path is the file's path relative to ``<root>`` (POSIX-formatted).
        ``kind`` optionally filters by the first path segment."""
        root = self.root.resolve()

        def _scan() -> "tuple[tuple[str, bytes], ...]":
            if not root.exists():
                return ()
            found: "list[tuple[str, bytes]]" = []
            for file in root.rglob("*"):
                if not file.is_file():
                    continue
                rel = file.relative_to(root)
                spec_path = rel.as_posix()
                if kind is not None and _kind_of(spec_path) != kind:
                    continue
                found.append((spec_path, file.read_bytes()))
            return tuple(found)

        scanned = await asyncio.to_thread(_scan)
        return tuple(_info(p, c) for p, c in sorted(scanned, key=lambda item: item[0]))

    # ---- writer --------------------------------------------------------

    async def put(self, entry: SpecDocument) -> SpecDocument:
        entry.validate_etag()
        target = self._resolve(entry.info.path)
        await asyncio.to_thread(atomic_write_bytes, target, entry.content)
        return entry

    async def delete(self, path: str) -> None:
        target = self._resolve(path)
        await asyncio.to_thread(_unlink_if_exists, target)

    async def reset(self, entries: "tuple[SpecDocument, ...]") -> None:
        for entry in entries:
            entry.validate_etag()
        # Full replacement: delete every existing file whose path is not in the
        # new set, then write the new set.
        keep = {entry.info.path for entry in entries}
        existing = {info.path for info in await self.list_info()}
        for path in existing - keep:
            await asyncio.to_thread(_unlink_if_exists, self._resolve(path))
        for entry in entries:
            target = self._resolve(entry.info.path)
            await asyncio.to_thread(atomic_write_bytes, target, entry.content)

    async def apply_batch(
        self,
        puts: "tuple[SpecDocument, ...]",
        deletes: "tuple[str, ...]",
    ) -> None:
        for entry in puts:
            entry.validate_etag()
        put_paths = {entry.info.path for entry in puts}
        for entry in puts:
            target = self._resolve(entry.info.path)
            await asyncio.to_thread(atomic_write_bytes, target, entry.content)
        for path in deletes:
            if path in put_paths:
                continue
            await asyncio.to_thread(_unlink_if_exists, self._resolve(path))


def _info(path: str, content: bytes) -> SpecDocumentInfo:
    return SpecDocumentInfo(
        path=path,
        kind=_kind_of(path),
        version=1,
        etag=compute_spec_etag(content),
        active=True,
    )


def _kind_of(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "spec"


def _unlink_if_exists(target: Path) -> None:
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass


__all__ = ["LocalSpecBackend"]
