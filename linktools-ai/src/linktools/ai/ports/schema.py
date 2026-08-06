#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Output schema registry protocol."""

from typing import Protocol


class SchemaRegistry(Protocol):
    def register(self, entry: object) -> object: ...
    def resolve(self, schema_id: str, revision: int) -> object: ...
    def verify(self, schema_id: str, revision: int, fingerprint: str) -> object: ...


__all__ = ["SchemaRegistry"]
