#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared specification parsing, indexing, and persistence contracts."""

from .cache import SpecObjectCache
from .contracts import (
    SpecCodec,
    SpecConflictError,
    SpecError,
    SpecNotFoundError,
    SpecParseError,
    SpecSource,
)
from .document import SpecDocument, SpecDocumentInfo, compute_spec_etag
from .index import SpecIndex
from .parsing import (
    SpecLoader,
    StrictConfigReader,
    parse_json_text,
    parse_markdown_text,
    parse_yaml_text,
)
from .source import SpecLoaderSource
from .store import SpecStore

__all__: "list[str]" = [
    "SpecCodec",
    "SpecConflictError",
    "SpecError",
    "SpecNotFoundError",
    "SpecParseError",
    "SpecSource",
    "SpecDocument",
    "SpecDocumentInfo",
    "SpecIndex",
    "SpecLoader",
    "SpecLoaderSource",
    "SpecObjectCache",
    "SpecStore",
    "StrictConfigReader",
    "compute_spec_etag",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
]
