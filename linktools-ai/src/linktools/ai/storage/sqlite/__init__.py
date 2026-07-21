#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite reference Storage composition (plan §4.7 / §3.2).

This is the ONE place in core where a database engine is constructed
(plan §6.5: 'SQLite helper 可以构造 engine; 通用 adapter 不构造 engine'). It
builds a sqlite+aiosqlite async engine + sessionmaker, composes Filesystem
artifact blobs + process-local coordination + DATABASE-scope features, and
delegates the store wiring to
:class:`~linktools.ai.storage.sqlalchemy.facade.SqlAlchemyStorageAdapter`.
Install the optional driver via ``linktools-ai[sqlite]``."""

from .facade import SqliteStorage

__all__: "list[str]" = ["SqliteStorage"]
