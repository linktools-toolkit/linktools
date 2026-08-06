#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Output and dependency contract declarations."""

from pydantic import BaseModel, ConfigDict, Field


class OutputContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    contract_id: str
    version: int = Field(ge=1)
    schema_id: str
    schema_revision: int = Field(ge=1)


class DependencyContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    contract_id: str
    version: int = Field(ge=1)
    dependency_ids: "tuple[str, ...]" = ()


__all__ = ["DependencyContract", "OutputContract"]
