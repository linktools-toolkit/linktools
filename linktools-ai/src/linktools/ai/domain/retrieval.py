#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Retrieval provenance values."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalScope(StrEnum):
    PROJECT = "project"
    TENANT = "tenant"
    PUBLIC = "public"


class RetrievalTrust(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class RetrievalContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    scope: RetrievalScope
    trust: RetrievalTrust
    limit: int = Field(default=10, ge=1, le=100)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    content: str
    digest: str
    scope: RetrievalScope
    trust: RetrievalTrust
    score: "float | None" = None


__all__ = ["RetrievalContext", "RetrievalResult", "RetrievalScope", "RetrievalTrust"]
