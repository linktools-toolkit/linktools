#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preserve binary bind bytes on aiomysql's text-protocol cursor."""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from sqlalchemy import LargeBinary, String, false, true
from sqlalchemy.dialects.mysql import LONGBLOB, VARBINARY
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import TypeDecorator

if TYPE_CHECKING:
    from sqlalchemy import Dialect, Insert, TypeEngine
    from sqlalchemy.sql.elements import BindParameter, ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession


# Stay below aiomysql's default executemany statement budget. A single row
# still goes to the server intact, as in aiomysql's own batching.
_INSERT_BUDGET = 1_024_000



def _uses_aiomysql(dialect: "Dialect") -> bool:
    return dialect.name == "mysql" and dialect.driver == "aiomysql"


class _BinaryBind(FunctionElement):
    inherit_cache = True


@compiles(_BinaryBind)
def _compile_binary_bind(element, compiler, **kwargs):
    parameter, = element.clauses
    sql = compiler.process(parameter, **kwargs)
    return f"UNHEX({sql})" if _uses_aiomysql(compiler.dialect) else sql


class SqlBinary(TypeDecorator):
    """Keep the binary column/result type; adapt only the aiomysql bound value."""

    impl = LargeBinary
    cache_ok = True

    class comparator_factory(TypeDecorator.Comparator):
        def in_(self, other: object) -> "ColumnElement[bool]":
            # SQLAlchemy expands an empty IN to a predicate, not a list of
            # values. It cannot apply a binary bind function to that predicate.
            return false() if isinstance(other, (tuple, list)) and not other else super().in_(other)

        def not_in(self, other: object) -> "ColumnElement[bool]":
            return true() if isinstance(other, (tuple, list)) and not other else super().not_in(other)

        notin_ = not_in

    def __init__(self, length: int | None = None) -> None:
        self.length = length
        super().__init__(length)

    @property
    def python_type(self) -> type[bytes]:
        return bytes

    def load_dialect_impl(self, dialect: "Dialect") -> "TypeEngine":
        if dialect.name == "mysql":
            return dialect.type_descriptor(LONGBLOB() if self.length is None else VARBINARY(self.length))
        return dialect.type_descriptor(self.impl)

    def bind_expression(self, bindvalue: "BindParameter") -> "ColumnElement":
        return _BinaryBind(bindvalue)

    def bind_processor(self, dialect: "Dialect") -> Callable | None:
        if not _uses_aiomysql(dialect):
            return super().bind_processor(dialect)
        return self._hex

    def literal_processor(self, dialect: "Dialect") -> Callable | None:
        if not _uses_aiomysql(dialect):
            return super().literal_processor(dialect)
        literal = String().literal_processor(dialect)
        return lambda value: literal(self._hex(value))

    @staticmethod
    def _hex(value: bytes | bytearray | memoryview | None) -> str | None:
        return None if value is None else value.hex()


async def execute_binary_insert_many(
    session: "AsyncSession", statement: "Insert", rows: Sequence[Mapping[str, object]],
) -> int:
    """Preserve bulk inserts when aiomysql cannot coalesce UNHEX VALUES."""
    parameters = [dict(row) for row in rows]
    if not parameters:
        return 0
    binds_binary = any(isinstance(statement.table.c[key].type, SqlBinary) for key in parameters[0])
    if not binds_binary or not _uses_aiomysql(session.get_bind().dialect):
        return (await session.execute(statement, parameters)).rowcount

    count = 0
    for batch in _bounded_batches(parameters):
        count += (await session.execute(statement.values(batch))).rowcount
    return count


def _bounded_batches(rows: list[dict[str, object]]) -> Iterator[list[dict[str, object]]]:
    # Leave room for the INSERT and ON DUPLICATE KEY UPDATE clauses. Strings
    # and UTF-8 JSON can require twice their bytes after SQL escaping; binary
    # binds are hex, so likewise expand to two ASCII bytes for each input byte.
    used = 4_096
    start = 0
    for offset, row in enumerate(rows):
        size = sum(16 + _wire_value_size(value) for value in row.values())
        if offset > start and used + size > _INSERT_BUDGET:
            yield rows[start:offset]
            start = offset
            used = 4_096
        used += size
    yield rows[start:]


def _wire_value_size(value: object) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return 2 * (value.nbytes if isinstance(value, memoryview) else len(value))
    if isinstance(value, (Mapping, tuple, list)):
        return 2 * len(json.dumps(value).encode("utf-8"))
    return 2 * len(str(value).encode("utf-8"))


__all__ = ["SqlBinary", "execute_binary_insert_many"]
