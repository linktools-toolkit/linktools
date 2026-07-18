#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.storage.resource: the resource domain's public model.
ResourceStore is the Primary+Overlay composition; ResourcePath is the
normalized path value type; Found/Masked/Missing are the three-state
resource lookup result, and WriteOptions carries conditional-write
preconditions."""

from .models import Found, Masked, Missing, WriteOptions
from .path import ResourcePath
from .store import ResourceStore

__all__ = ["ResourceStore", "ResourcePath", "Found", "Masked", "Missing", "WriteOptions"]
