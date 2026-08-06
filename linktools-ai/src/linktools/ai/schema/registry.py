#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Idempotent output schema registry with drift detection."""

from linktools.core import environ

from ..foundation.errors import ErrorCode, LinktoolsAIError
from .model import SchemaEntry

logger = environ.get_logger("ai.schema.registry")


class OutputSchemaRegistry:
    """Register and verify schema identities before a run can proceed."""

    def __init__(self) -> None:
        self._entries: "dict[tuple[str, int], SchemaEntry]" = {}

    def register(self, entry: SchemaEntry) -> SchemaEntry:
        key = (entry.schema_id, entry.revision)
        current = self._entries.get(key)
        if current is not None and current.fingerprint != entry.fingerprint:
            logger.warning("output schema drift schema=%s revision=%s", entry.schema_id, entry.revision)
            raise LinktoolsAIError(ErrorCode.OUTPUT_SCHEMA_DRIFT, "output schema fingerprint changed")
        self._entries[key] = entry
        logger.info("output schema registered schema=%s revision=%s", entry.schema_id, entry.revision)
        return current or entry

    def resolve(self, schema_id: str, revision: int) -> SchemaEntry:
        entry = self._entries.get((schema_id, revision))
        if entry is None:
            raise LinktoolsAIError(ErrorCode.OUTPUT_SCHEMA_UNKNOWN, "output schema is not registered")
        return entry

    def verify(self, schema_id: str, revision: int, fingerprint: str) -> SchemaEntry:
        entry = self.resolve(schema_id, revision)
        if entry.fingerprint != fingerprint:
            raise LinktoolsAIError(ErrorCode.OUTPUT_SCHEMA_DRIFT, "output schema fingerprint changed")
        return entry


__all__ = ["OutputSchemaRegistry"]
