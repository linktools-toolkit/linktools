#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit Activity binding DTOs."""

from pydantic import BaseModel, ConfigDict


class ModelBinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    route_id: str


class ToolActivityBinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_id: str
    operation_id: str
    approval_id: "str | None" = None


class LiveEventBinding(BaseModel):
    model_config = ConfigDict(frozen=True)
    publisher_id: str


__all__ = ["LiveEventBinding", "ModelBinding", "ToolActivityBinding"]
