#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.asset: "assets" is just the business name for a particular
object-storage composition -- this module provides ``compose_assets()``, a
thin convenience over the storage.object kernel's ``RevisionedOverlayObjectStore``,
for any caller that wants a primary+overlays object composition under that
name."""

from ..storage.object.backend import ObjectReaderBackend, ObjectWriterBackend
from ..storage.object.overlay import RevisionedOverlayObjectStore


def compose_assets(
    *,
    primary: "ObjectWriterBackend | None" = None,
    overlays: "tuple[ObjectReaderBackend, ...]" = (),
    cache=None,
    index=None,
) -> RevisionedOverlayObjectStore:
    return RevisionedOverlayObjectStore(
        primary=primary, overlays=overlays, cache=cache, index=index
    )


__all__ = ["compose_assets"]
