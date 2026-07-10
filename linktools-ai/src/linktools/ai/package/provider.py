#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DirectoryPackageResourceProvider: the default file-backed
PackageResourceProvider. Reads resources from a per-package root directory with
the safety guarantees: path sandbox (no ``..`` escape), pagination,
and a max_bytes read clamp."""

from pathlib import Path
from typing import Mapping

from ..errors import PackageNotFoundError, PackageResourceAccessDeniedError, PackageResourceNotFoundError
from ..providers.package import PackageResourceProvider
from .resource import (
    ResourceContent,
    ResourceInfo,
    ResourceListResult,
    ResourceRef,
    sanitize_package_path,
)
from .scope import PackageScope

DEFAULT_MAX_READ_BYTES = 65536
DEFAULT_LIST_LIMIT = 50


class DirectoryPackageResourceProvider:
    """PackageResourceProvider over a ``package_id -> root Path`` mapping.

    ``from_base(base_dir)`` builds a provider whose roots are discovered lazily
    as ``<base_dir>/<package_id>``; ``(roots=...)`` takes an explicit mapping
    for deployments where package ids do not map 1:1 to directory names."""

    def __init__(
        self,
        roots: "Mapping[str, Path | str]",
        *,
        allow_extensions: "tuple[str, ...] | None" = None,
        deny_extensions: "tuple[str, ...]" = (),
    ) -> None:
        self._roots: "dict[str, Path]" = {pid: Path(p) for pid, p in roots.items()}
        # Extension allow/deny lists, normalized lowercase with
        # a leading dot. When set, read_resource refuses disallowed extensions.
        self._allow_ext = tuple(e.lower() for e in allow_extensions) if allow_extensions else None
        self._deny_ext = tuple(e.lower() for e in deny_extensions)

    @staticmethod
    def _ext_ok(path: str, allow: "tuple[str, ...] | None", deny: "tuple[str, ...]") -> bool:
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        if deny and ext in deny:
            return False
        if allow is not None and ext not in allow:
            return False
        return True

    @classmethod
    def from_base(cls, base_dir: "Path | str") -> "DirectoryPackageResourceProvider":
        # Roots resolve lazily on demand; store a single base and look packages
        # up under it. Implemented by subclassing-style via a sentinel-free
        # mapping populated on first access is avoided -- instead we treat the
        # base dir as the roots provider by overriding _root_for.
        return _BaseDirPackageResourceProvider(Path(base_dir))

    def _root_for(self, package_id: str) -> "Path | None":
        return self._roots.get(package_id)

    async def list_resources(
        self,
        scope: PackageScope,
        path: str = "",
        *,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: "str | None" = None,
    ) -> ResourceListResult:
        root = self._root_for(scope.package_id)
        if root is None:
            raise PackageNotFoundError(f"package not found: {scope.package_id}")
        rel = sanitize_package_path(path)
        target = root / rel if rel else root
        if not target.exists() or not target.is_dir():
            return ResourceListResult(items=[], next_cursor=None)

        root_resolved = root.resolve()
        # Defense-in-depth: rglob may follow symlinked directories (pre-3.13),
        # yielding paths outside the package root. Skip any entry that does not
        # resolve under root rather than letting relative_to() raise (which would
        # also leak the absolute outside-root path in the error message).
        files: "list[str]" = []
        for p in target.rglob("*"):
            if not p.is_file():
                continue
            try:
                if p.resolve().relative_to(root_resolved) is None:
                    continue
            except ValueError:
                continue
            files.append(p.relative_to(root).as_posix())
        files.sort()
        offset = int(cursor) if cursor else 0
        page = files[offset:offset + max(1, limit)]
        next_cursor = str(offset + len(page)) if offset + len(page) < len(files) else None
        items = [
            ResourceInfo(path=p, kind="file", size_bytes=(root / p).stat().st_size)
            for p in page
        ]
        return ResourceListResult(items=items, next_cursor=next_cursor)

    async def read_resource(
        self,
        ref: ResourceRef,
        *,
        max_bytes: "int | None" = None,
    ) -> ResourceContent:
        if ref.scope is None:
            raise PackageResourceAccessDeniedError("resource ref has no package scope")
        root = self._root_for(ref.scope.package_id)
        if root is None:
            raise PackageNotFoundError(f"package not found: {ref.scope.package_id}")
        rel = sanitize_package_path(ref.path)
        if not rel:
            raise PackageResourceAccessDeniedError("empty resource path")
        if not self._ext_ok(rel, self._allow_ext, self._deny_ext):
            raise PackageResourceAccessDeniedError(
                f"resource extension not allowed: {ref.path!r}"
            )
        target = (root / rel).resolve()
        # Defense-in-depth: confirm the resolved target stays under root even
        # though sanitize_package_path already rejected ``..``.
        try:
            target.relative_to(root.resolve())
        except ValueError:
            raise PackageResourceAccessDeniedError(
                f"path escapes package root: {ref.path!r}"
            )
        if not target.is_file():
            raise PackageResourceNotFoundError(f"resource not found: {ref.path!r}")

        cap = max_bytes if max_bytes is not None else DEFAULT_MAX_READ_BYTES
        true_size = target.stat().st_size
        # Bound the read itself (read one byte past the cap to detect truncation)
        # so a multi-GB resource cannot OOM the process regardless of max_bytes.
        with target.open("rb") as fh:
            data = fh.read(cap + 1)
        truncated = len(data) > cap
        payload: "bytes | str" = data[:cap] if truncated else data
        return ResourceContent(
            path=rel,
            content=payload,
            content_type="application/octet-stream",
            size_bytes=true_size,
            metadata={"truncated": truncated} if truncated else {},
        )


class _BaseDirPackageResourceProvider(DirectoryPackageResourceProvider):
    """Roots resolve as ``<base>/<package_id>`` on demand."""

    def __init__(self, base: Path) -> None:
        self._base = base

    def _root_for(self, package_id: str) -> "Path | None":
        candidate = (self._base / package_id).resolve()
        # Only admit directories that actually live under base.
        try:
            candidate.relative_to(self._base.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_dir() else None


# Re-export the Protocol alongside the default implementation.
__all__ = [
    "PackageResourceProvider",
    "DirectoryPackageResourceProvider",
    "DEFAULT_MAX_READ_BYTES",
    "DEFAULT_LIST_LIMIT",
]
