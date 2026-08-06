#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stable Extension and Feature Registry API."""

from .model import Extension, ExtensionProvider, ExtensionResolution
from .registry import ExtensionRegistry, FeatureRegistry

__all__ = ["Extension", "ExtensionProvider", "ExtensionRegistry", "ExtensionResolution", "FeatureRegistry"]
