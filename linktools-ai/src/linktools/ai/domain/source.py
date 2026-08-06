#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source and incremental index values."""

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    path: str
    digest: str
    content: str
    revision: int = Field(ge=1)


class SourceRevision(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    revision: int = Field(ge=1)
    digest: str


class IndexEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    path: str
    digest: str
    revision: int = Field(ge=1)


__all__ = ["Document", "IndexEntry", "SourceRevision"]
