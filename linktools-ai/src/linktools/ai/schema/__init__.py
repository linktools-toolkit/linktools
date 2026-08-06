#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stable output schema API."""

from .model import SchemaEntry, SchemaKey
from .registry import OutputSchemaRegistry

__all__ = ["OutputSchemaRegistry", "SchemaEntry", "SchemaKey"]
