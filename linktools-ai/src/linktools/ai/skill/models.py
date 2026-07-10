#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill summary/content models. Summaries are the lightweight form
injected into the prompt catalog; content is the full body returned by
read_skill on demand."""

from typing import Any

from pydantic import BaseModel, Field


class SkillSummary(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    tags: "list[str]" = Field(default_factory=list)
    package_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class SkillContent(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    content: str
    package_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)
