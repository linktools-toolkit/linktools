#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite reference Storage composition.

This is the ONE place in core where a database engine is constructed
. It
builds a sqlite+aiosqlite async engine + sessionmaker, composes Filesystem
artifact blobs + process-local coordination + DATABASE-scope features, and
delegates the store wiring to
:class:`~linktools.ai.storage.sqlalchemy.facade.SqlAlchemyStorageAdapter`.
Install the optional driver via ``linktools-ai[sqlite]``."""

from .facade import SqliteStorage

__all__: "list[str]" = ["SqliteStorage"]
