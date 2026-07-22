#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpecLoaderSource: the generic CatalogSource adapter over the in-repo
SpecLoader (catalog/parsing).

Every domain Catalog that reads from a filesystem / asset-backed loader uses
this adapter -- it is not agent-specific (it previously lived in
agent/catalog.py, which coupled every other domain to agent). The loader exposes
an int revision (a monotonic clock); CatalogSource declares a string revision,
so the int is stringified. ``read`` propagates the loader's native
RegistryNotFoundError so a domain keeps its existing not-found error type.
"""

from __future__ import annotations

from .contracts import CatalogSource
from .parsing import SpecLoader


class SpecLoaderSource:
    """CatalogSource adapter over a SpecLoader."""

    def __init__(self, loader: SpecLoader) -> None:
        self._loader = loader

    async def revision(self) -> str:
        return str(await self._loader.revision())

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._loader.list_ids(suffix)

    async def read(self, path: str) -> str:
        return await self._loader.read(path)


__all__: "list[str]" = ["SpecLoaderSource"]
