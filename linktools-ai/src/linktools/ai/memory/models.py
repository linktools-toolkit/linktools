#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory domain model: MemoryRecord (the persisted memory entry)."""

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
    # The store is authoritative; indexes are derived and may be rebuilt.
    index_status: str = "pending"
    index_version: int = 0
    indexed_at: "datetime | None" = None
