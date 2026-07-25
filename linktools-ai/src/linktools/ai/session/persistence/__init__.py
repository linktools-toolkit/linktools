#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session persistence: the in-repo reference SessionStore adapters (Filesystem
+ SQLAlchemy) for the session domain. Each adapter lives in its own domain so
the storage kernel never imports session-domain models.

Import the specific backend module directly
(``from linktools.ai.session.persistence.filesystem import FilesystemSessionStore``
or ``...sqlalchemy import SqlAlchemySessionStore``); the package ``__init__``
deliberately does NOT auto-import the SQLAlchemy variant, which would create a
circular dependency through ``storage.sqlalchemy``'s eager ``__init__``."""

__all__: "list[str]" = ["FilesystemSessionStore", "SqlAlchemySessionStore"]
