#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TieredContentCache: GET tries L1, then L2, ...; a hit at a lower layer
backfills every higher layer that missed. PUT/DELETE fan out to every layer,
best-effort. A single layer raising must never block a later layer or the
caller's fall-through to the origin -- a cache is optional infrastructure,
never a correctness dependency."""

from __future__ import annotations

from typing import Sequence


class TieredContentCache:
    def __init__(self, *, l1, l2=None, layers: "Sequence | None" = None) -> None:
        if layers is not None:
            self._layers = tuple(layers)
        else:
            self._layers = tuple(layer for layer in (l1, l2) if layer is not None)

    async def get(self, key: str) -> "bytes | None":
        missed = []
        for layer in self._layers:
            try:
                content = await layer.get(key)
            except Exception:
                continue
            if content is not None:
                for higher in missed:
                    try:
                        await higher.put(key, content)
                    except Exception:
                        pass
                return content
            missed.append(layer)
        return None

    async def put(self, key: str, content: bytes) -> None:
        for layer in self._layers:
            try:
                await layer.put(key, content)
            except Exception:
                continue

    async def delete(self, key: str) -> None:
        for layer in self._layers:
            try:
                await layer.delete(key)
            except Exception:
                continue


__all__: "list[str]" = ["TieredContentCache"]
