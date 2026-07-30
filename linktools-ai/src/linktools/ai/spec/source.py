#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt a text loader to the generic specification source protocol."""

from __future__ import annotations

from .parsing import SpecLoader


class SpecLoaderSource:
    """SpecSource adapter over a SpecLoader. ``identity`` exposes the content
    digest for ``raw`` so a caller caches by content identity in a single read,
    without a separate revision probe."""

    def __init__(self, loader: SpecLoader) -> None:
        self._loader = loader

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._loader.list_ids(suffix)

    async def read(self, path: str) -> str:
        return await self._loader.read(path)

    def identity(self, raw: str) -> str:
        return self._loader.identity(raw)


__all__: "list[str]" = ["SpecLoaderSource"]
