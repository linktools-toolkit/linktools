#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native cursor lifetime for SQLite engines owned by storage."""

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Self

_Parameters = Sequence[object] | Mapping[str, object]


class _OwnedCursor(sqlite3.Cursor):
    def execute(self, sql: str, parameters: _Parameters = (), /) -> Self:
        try:
            return super().execute(sql, parameters)
        except BaseException as error:
            self._close_failed(error)
            raise

    def executemany(self, sql: str, parameters: Iterable[_Parameters], /) -> Self:
        try:
            return super().executemany(sql, parameters)
        except BaseException as error:
            self._close_failed(error)
            raise

    def executescript(self, sql_script: str, /) -> Self:
        try:
            return super().executescript(sql_script)
        except BaseException as error:
            self._close_failed(error)
            raise

    def _close_failed(self, error: BaseException) -> None:
        # Finalizing a failed statement later on the event-loop thread can wait
        # on a COMMIT mutex whose reader can only be released by that same loop.
        try:
            super().close()
        except BaseException as close_error:
            raise error from close_error


class _OwnedConnection(sqlite3.Connection):
    def cursor(self, factory: type[sqlite3.Cursor] = _OwnedCursor) -> sqlite3.Cursor:
        return super().cursor(factory)

    def execute(self, sql: str, parameters: _Parameters = (), /) -> sqlite3.Cursor:
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[_Parameters], /) -> sqlite3.Cursor:
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        return self.cursor().executescript(sql_script)


__all__ = []
