#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared specification parsing, indexing, and persistence contracts."""

from .contracts import (
    SpecCodec,
    SpecConflictError,
    SpecError,
    SpecNotFoundError,
    SpecParseError,
    SpecSource,
)
from .index import SpecIndex
from .document import SpecDocument, SpecDocumentChange, SpecDocumentInfo
from .parsing import SpecLoader, StrictConfigReader, parse_json_text, parse_markdown_text, parse_yaml_text
from .source import SpecLoaderSource
from .store import SpecReader, SpecStore, SpecWriter

__all__: "list[str]" = [
    "SpecCodec",
    "SpecConflictError",
    "SpecError",
    "SpecNotFoundError",
    "SpecParseError",
    "SpecSource",
    "SpecReader",
    "SpecDocument",
    "SpecDocumentChange",
    "SpecDocumentInfo",
    "SpecIndex",
    "SpecLoader",
    "SpecLoaderSource",
    "SpecStore",
    "SpecWriter",
    "StrictConfigReader",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
]
