#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic Catalog contracts: the shared Source/Codec/Cache primitives every
domain Catalog (AgentCatalog, SkillCatalog, MCPCatalog, ToolCatalog,
SwarmCatalog, ExtensionCatalog) composes.

These were previously inlined per-domain (each registry coupled Source +
cache + parse). Extracting them here gives every Catalog the same observable
semantics for free:

- cache key includes the source revision;
- a revision change atomically invalidates the cache and the id listing;
- concurrent refresh executes the source-revision check exactly once;
- not-found is distinguishable from parse-failure;
- errors carry the item id, the source, and (from the codec) the field path;
- the codec strictly rejects unknown fields.

A domain Catalog is a thin facade: it holds a CatalogSource, a CatalogCodec[T],
and a RevisionCache[T], and delegates list/get.
"""

from .contracts import (
    CatalogCodec,
    CatalogError,
    CatalogItemNotFoundError,
    CatalogItemParseError,
    CatalogSource,
    RevisionCache,
)

__all__: "list[str]" = [
    "CatalogCodec",
    "CatalogError",
    "CatalogItemNotFoundError",
    "CatalogItemParseError",
    "CatalogSource",
    "RevisionCache",
]
