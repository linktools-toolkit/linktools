#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory domain model: MemoryRecord (the persisted memory entry)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .._typing import JSONValue


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    owner_id: str
    content: str
    category: "str | None"
    confidence: "float | None"
    version: int
    created_at: datetime
    updated_at: datetime
    metadata: "Mapping[str, JSONValue]"
