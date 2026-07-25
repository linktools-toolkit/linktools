#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jobs persistence: the in-repo reference JobStore adapters (Filesystem +
SQLAlchemy) for the jobs domain. Each adapter lives in its own domain so the
storage kernel never imports jobs-domain models.

Import the specific backend module directly
(``from linktools.ai.jobs.persistence.filesystem import FilesystemJobStore``
or ``...sqlalchemy import SqlAlchemyJobStore``); the package ``__init__``
deliberately does NOT auto-import the SQLAlchemy variant, which would create a
circular dependency through ``storage.sqlalchemy``'s ``__init__``."""

__all__: "list[str]" = ["FilesystemJobStore", "SqlAlchemyJobStore"]
