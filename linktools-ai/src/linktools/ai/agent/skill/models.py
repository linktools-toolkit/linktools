#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime skill summary and content models."""

from typing import Any

from pydantic import BaseModel, Field

class SkillSummary(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    tags: "list[str]" = Field(default_factory=list)
    extension_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class SkillContent(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    content: str
    extension_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)
