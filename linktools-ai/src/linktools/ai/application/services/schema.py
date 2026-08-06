#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Output schema validation service."""

class SchemaService:
    def __init__(self, registry: object) -> None:
        self._registry = registry

    def verify(self, schema_id: str, revision: int, fingerprint: str) -> object:
        return self._registry.verify(schema_id, revision, fingerprint)


__all__ = ["SchemaService"]
