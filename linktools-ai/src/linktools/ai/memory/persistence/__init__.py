#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\"memory\" persistence: the in-repo reference adapters (Filesystem +
SQLAlchemy) for the memory domain. Each adapter lives in its own domain so
the storage kernel never imports memory-domain models.

Import the specific backend module directly (...memory.persistence.filesystem
or ...memory.persistence.sqlalchemy); the package __init__ deliberately
does NOT auto-import the SQLAlchemy variant, which would create a circular
dependency through storage.sqlalchemy's __init__."""

__all__: "list[str]" = ["FilesystemMemoryStore", "SqlAlchemyMemoryStore"]
