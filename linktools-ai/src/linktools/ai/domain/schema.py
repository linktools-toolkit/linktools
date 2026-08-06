#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Output schema identity and fingerprint values."""

from pydantic import BaseModel, ConfigDict, Field


class SchemaKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(min_length=1)
    revision: int = Field(ge=1)


class SchemaEntry(SchemaKey):
    fingerprint: str = Field(min_length=1)
    python_type_path: str = Field(min_length=1)
    json_schema: "dict[str, object]"


__all__ = ["SchemaEntry", "SchemaKey"]
