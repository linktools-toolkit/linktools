#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionScope: identifies an extension for scoped resolution. Two
extensions may both contain ``agents/grader.md``; the scope keeps their scoped
entrypoints and resources from colliding (internal key ``extension:<id>:...``)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtensionScope:
    extension_id: str
    extension_kind: "str | None" = None
