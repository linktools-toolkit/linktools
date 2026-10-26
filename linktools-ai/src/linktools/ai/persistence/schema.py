#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment-only SQL schema entry point for Runtime storage."""

from ._plan import PersistenceDomain
from ._schema import build_runtime_sql_metadata

__all__ = ['PersistenceDomain', 'build_runtime_sql_metadata']
