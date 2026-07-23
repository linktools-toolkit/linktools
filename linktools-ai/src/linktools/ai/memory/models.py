#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory domain model: MemoryRecord (the persisted memory entry) and
MemoryMatch (a search result pairing a record with an optional relevance
score)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .._typing import JSONValue


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    # tenant_id is the hard isolation boundary (required on every record so
    # search can be tenant-scoped). user_id / workspace_id / session_id are
    # optional sub-scopes; a NULL there means "shared at the tenant level".
    # owner_id is retained as a display / compat field -- it is NOT an
    # authorization boundary (two tenants can share an owner_id and must stay
    # isolated), and search does not key on it alone.
    id: str
    tenant_id: str
    owner_id: str
    content: str
    category: "str | None"
    confidence: "float | None"
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: "Mapping[str, JSONValue]"
    user_id: "str | None" = None
    workspace_id: "str | None" = None
    session_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """One search hit: the matching record plus an optional relevance score.
    A keyword backend carries no real ranking signal, so it returns ``score=None``
    rather than fabricating a value; a backend that ranks (e.g. a vector index)
    fills in a ``float``."""

    record: MemoryRecord
    score: "float | None" = None
