#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\"tool\" persistence: the in-repo reference adapters (Filesystem +
SQLAlchemy) for the tool domain. Each adapter lives in its own domain so
the storage kernel never imports tool-domain models.

Import the specific backend module directly (...tool.persistence.filesystem
or ...tool.persistence.sqlalchemy); the package __init__ deliberately
does NOT auto-import the SQLAlchemy variant, which would create a circular
dependency through storage.sqlalchemy's __init__."""

__all__: "list[str]" = ["FilesystemIdempotencyStore", "SqlAlchemyIdempotencyStore"]
