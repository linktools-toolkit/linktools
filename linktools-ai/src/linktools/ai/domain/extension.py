#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit extension declarations and provider values."""

from pydantic import BaseModel, ConfigDict, Field


class Extension(BaseModel):
    model_config = ConfigDict(frozen=True)

    extension_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    provider_id: str = Field(min_length=1)
    capabilities: "tuple[str, ...]" = ()
    digest: str = Field(min_length=1)


class ExtensionProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    extensions: "tuple[Extension, ...]" = ()
    signed: bool = False


class ExtensionResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    extension_id: str
    provider_id: str
    capability_ids: "tuple[str, ...]"


__all__ = ["Extension", "ExtensionProvider", "ExtensionResolution"]
