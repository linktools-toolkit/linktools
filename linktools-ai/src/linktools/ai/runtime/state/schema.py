#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment-only SQL schema entry point for Runtime storage."""

from ._plan import RuntimeDomain
from ._schema import build_runtime_sql_metadata

__all__ = ["RuntimeDomain", "build_runtime_sql_metadata"]
