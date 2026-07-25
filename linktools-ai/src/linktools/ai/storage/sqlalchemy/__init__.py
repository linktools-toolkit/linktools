#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy-backed storage. Importing this package pulls in SQLAlchemy, which
is an optional dependency -- prefer ``from linktools.ai.storage import
SqlAlchemyStorage`` (lazy) unless you specifically want eager import.

The ``SqlAlchemyStorage`` symbol is itself lazy: a domain adapter that needs
one of the per-domain Row classes from ``storage.sqlalchemy.models`` can import
that module directly without pulling the facade in (and the facade pulls in
every domain adapter, so eager-loading it from the package ``__init__`` would
create a cycle)."""

__all__ = ["SqlAlchemyStorage"]


def __getattr__(name: str):
    if name == "SqlAlchemyStorage":
        from .facade import SqlAlchemyStorage

        return SqlAlchemyStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
