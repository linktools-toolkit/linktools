#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.asset: the asset domain's public model.
AssetStore is the Primary+Overlay composition; AssetPath is the
normalized path value type; Found/Masked/Missing are the three-state
asset lookup result, and WriteOptions carries conditional-write
preconditions."""

from .models import Found, Masked, Missing, WriteOptions
from .path import AssetPath
from .readonly import ReadOnlyAssetBackend
from .store import AssetStore

__all__ = [
    "AssetStore",
    "AssetPath",
    "ReadOnlyAssetBackend",
    "Found",
    "Masked",
    "Missing",
    "WriteOptions",
]
