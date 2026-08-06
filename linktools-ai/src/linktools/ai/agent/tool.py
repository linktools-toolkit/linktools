#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tool access metadata bound to one Activity."""

from pydantic import BaseModel, ConfigDict


class ToolAccess(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_id: str
    operation_id: str
    risk: str
    approval_required: bool = False


__all__ = ["ToolAccess"]
