#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asset persistence backends."""

from .local import LocalAssetBackend, PrefixAssetPathAdapter, AssetPathAdapter

try:
    from .sqlalchemy import SqlAlchemyAssetBackend
except ImportError:
    SqlAlchemyAssetBackend = None

__all__ = ["LocalAssetBackend", "PrefixAssetPathAdapter", "AssetPathAdapter", "SqlAlchemyAssetBackend"]
