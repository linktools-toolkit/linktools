#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoped DEBUG attribution for SQLAlchemy cursor activity, never SQL contents."""

import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from time import perf_counter_ns
from typing import ParamSpec, TypeVar

from linktools.core import environ

_logger = environ.get_logger("ai.storage.sql_observation")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
_VERBS = frozenset({
    "SELECT", "INSERT", "UPDATE", "DELETE", "REPLACE", "WITH", "BEGIN",
    "START", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "SET", "SHOW", "PRAGMA",
    "CREATE", "ALTER", "DROP", "EXPLAIN", "DESCRIBE", "CALL", "VALUES",
})


@dataclass(slots=True)
class _Scope:
    label: str
    counts: Counter[str] | None = None
    elapsed_ns: int = 0
    failures: int = 0
    closed: bool = False


_scopes: ContextVar[tuple[_Scope, ...]] = ContextVar("sql_observation_scopes", default=())


def current_sql_observation() -> tuple[str, ...]:
    """The current stack of application-defined SQL operation labels."""
    return tuple(scope.label for scope in _scopes.get() if not scope.closed)


def sql_observation_scope(label: str, *, inherit: bool = True) -> AbstractContextManager[None]:
    """Label an operation across awaits; SQL DEBUG uses cursor counts and timing.

    Use a static application label, never request text, IDs, or secret values.
    Child tasks inherit active labels but do not add work to a closed parent.
    Use inherit=False for the root of independent background work.
    Set the ai.storage.sql_observation logger to DEBUG for SDK-owned engines.
    Borrowed engines can read these labels in their existing cursor logger; the
    SDK installs no listener there. Summaries are inclusive of child scopes.
    Counts cover SQLAlchemy cursor calls, not DBAPI transaction methods.
    Summed cursor times are not full request wall time, and may overlap.
    """
    if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
        raise ValueError("SQL observation label must be an application operation name")
    return _scope(label, inherit=inherit)


@contextmanager
def _scope(label: str, *, inherit: bool) -> Iterator[None]:
    parents = tuple(scope for scope in _scopes.get() if not scope.closed) if inherit else ()
    scope = _Scope(label)
    token = _scopes.set((*parents, scope))
    result = "ok"
    try:
        yield
    except BaseException as error:
        result = type(error).__name__
        raise
    finally:
        scope.closed = True
        _scopes.reset(token)
        if scope.counts and _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "SQL scope operation=%s statements=%d by_verb=%s cursor_ms=%.3f failures=%d outcome=%s",
                "/".join((*[parent.label for parent in parents if not parent.closed], label)),
                sum(scope.counts.values()), dict(sorted(scope.counts.items())),
                scope.elapsed_ns / 1_000_000, scope.failures, result,
            )


_Params = ParamSpec("_Params")
_Value = TypeVar("_Value")


def sql_observation_phase(
    label: str, *, inherit: bool = True,
) -> Callable[[Callable[_Params, Awaitable[_Value]]], Callable[_Params, Awaitable[_Value]]]:
    """Decorate a coroutine with a static SQL phase spanning its full await."""
    if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
        raise ValueError("SQL observation label must be an application operation name")

    def decorate(fn: Callable[_Params, Awaitable[_Value]]) -> Callable[_Params, Awaitable[_Value]]:
        @wraps(fn)
        async def invoke(*args: _Params.args, **kwargs: _Params.kwargs) -> _Value:
            with _scope(label, inherit=inherit):
                return await fn(*args, **kwargs)
        return invoke
    return decorate


def _start(_connection, _cursor, statement, _parameters, context, _executemany):
    scopes = _scopes.get()
    if context is None or not scopes or not _logger.isEnabledFor(logging.DEBUG):
        return
    active = tuple(scope for scope in scopes if not scope.closed)
    if not active:
        return
    # Never emit the full string: even supposedly parameterized SQL can
    # contain literals. The only statement-derived label is an allowlisted verb.
    compiled = getattr(context, "compiled", None)
    query = getattr(compiled, "statement", None)
    verb = next((op for op in ("SELECT", "INSERT", "UPDATE", "DELETE")
                 if getattr(query, f"is_{op.lower()}", False)), None)
    if verb is None:
        prefix = statement.lstrip()[:16].split(maxsplit=1)
        verb = prefix[0].upper() if prefix else ""
        verb = verb if verb in _VERBS else "OTHER"
    context._linktools_sql_observation = (active, verb, perf_counter_ns())


def _finish(context, error: BaseException | None = None):
    sample = getattr(context, "_linktools_sql_observation", None)
    if sample is None:
        return
    del context._linktools_sql_observation
    scopes, verb, start = sample
    elapsed = perf_counter_ns() - start
    active = tuple(scope for scope in scopes if not scope.closed)
    if not active:
        return
    for scope in active:
        if scope.counts is None:
            scope.counts = Counter()
        scope.counts[verb] += 1
        scope.elapsed_ns += elapsed
        scope.failures += error is not None
    _logger.debug(
        "SQL cursor operation=%s verb=%s elapsed_ms=%.3f error=%s",
        "/".join(scope.label for scope in active), verb, elapsed / 1_000_000,
        "-" if error is None else type(error).__name__,
    )


def _success(_connection, _cursor, _statement, _parameters, context, _executemany):
    _finish(context)


def _failure(exception_context):
    _finish(exception_context.execution_context, exception_context.original_exception)


def attach_sql_observation(engine):
    """Install one set of transparent listeners on an SDK-owned engine."""
    from sqlalchemy import event

    engine = engine.sync_engine
    if not event.contains(engine, "before_cursor_execute", _start):
        event.listen(engine, "before_cursor_execute", _start)
        event.listen(engine, "after_cursor_execute", _success)
        event.listen(engine, "handle_error", _failure)
