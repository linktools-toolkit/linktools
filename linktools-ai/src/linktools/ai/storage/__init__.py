#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage package. ``Storage`` and ``FileStorage`` are core (no optional deps);
``SqlAlchemyStorage`` is loaded lazily on first access so ``import
linktools.ai.storage`` succeeds without SQLAlchemy installed."""

from .facade import FileStorage, Storage

__all__ = ["Storage", "FileStorage", "SqlAlchemyStorage"]


def __getattr__(name: str):
    if name == "SqlAlchemyStorage":
        try:
            from .sqlalchemy.facade import SqlAlchemyStorage
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"sqlalchemy", "aiosqlite"}:
                raise ImportError(
                    "SqlAlchemyStorage requires optional dependencies. "
                    "Install with: pip install 'linktools-ai[sqlite]' "
                    "or pip install 'linktools-ai[sqlalchemy]'."
                ) from exc
            raise
        return SqlAlchemyStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
