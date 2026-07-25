#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-addressed immutable blob storage: a generic storage capability with
no dependency on the artifact domain (or any other domain) that consumes it."""

from .protocols import BlobInfo, BlobStore

__all__: "list[str]" = ["BlobInfo", "BlobStore"]
