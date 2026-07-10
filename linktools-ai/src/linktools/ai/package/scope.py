#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PackageScope: identifies a package for scoped resolution. Two
packages may both contain ``agents/grader.md``; the scope keeps their scoped
entrypoints and resources from colliding (internal key ``package:<id>:...``)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PackageScope:
    package_id: str
    package_kind: "str | None" = None
