#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run persistence: the in-repo reference adapters (Filesystem + SQLAlchemy)
for the run domain -- the RunStore, CheckpointStore, RunDefinitionStore, and
the per-backend RunCommitCoordinator (+ the Filesystem TransactionJournal the
FS coordinator writes). Each adapter lives in its own domain so the storage
kernel never imports run-domain models.

Import the specific backend module directly; the package ``__init__``
deliberately does NOT auto-import the SQLAlchemy variant, which would create a
circular dependency through ``storage.sqlalchemy``'s ``__init__``."""

__all__: "list[str]" = [
    "FilesystemRunStore",
    "FilesystemCheckpointStore",
    "FilesystemRunDefinitionStore",
    "FilesystemRunCommitCoordinator",
    "SqlAlchemyRunStore",
    "SqlAlchemyCheckpointStore",
    "SqlAlchemyRunDefinitionStore",
    "SqlAlchemyRunCommitCoordinator",
]
