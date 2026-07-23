#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dialect detection + integrity-violation classification contract.

The strategy layer (storage/sqlalchemy/dialects/) is the single place core
branches on a dialect name. These tests pin:

- ``resolve_dialect_strategy`` maps sqlite -> SqliteDialectStrategy and rejects
  an unknown dialect with ``UnsupportedSqlAlchemyDialectError`` at adapter
  construction (not on first write);
- each classifier maps the asset-domain unique-key conflicts to
  ASSET_KEY / IDEMPOTENCY_KEY and everything else to OTHER.

Classifier logic is exercised with SYNTHETIC IntegrityErrors (the classifiers
read ``error.orig`` via getattr), so all three dialects are verified here
without a running MySQL/PostgreSQL -- the end-to-end INSERT-conflict path runs
in CI against real MySQL/PostgreSQL via the LINKTOOLS_AI_TEST_*_DSN env vars."""

import types

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.sqlalchemy.dialects import (
    IntegrityViolationKind,
    MySqlDialectStrategy,
    PostgreSqlDialectStrategy,
    SqlAlchemyDialectStrategy,
    SqliteDialectStrategy,
    UnsupportedSqlAlchemyDialectError,
    resolve_dialect_strategy,
)


class _Orig:
    """Stand-in for the DBAPI exception SQLAlchemy wraps as IntegrityError.orig."""

    def __init__(self, message="", *, code=None, constraint=None):
        self._message = message
        self.args = (code,) if code is not None else ()
        if constraint is not None:
            self.diag = types.SimpleNamespace(constraint_name=constraint)

    def __str__(self):
        return self._message


class _Err:
    """Stand-in for a SQLAlchemy IntegrityError exposing ``.orig``."""

    def __init__(self, orig):
        self.orig = orig


class _FakeFactory:
    """A session_factory-like object whose ``kw['bind']`` is a fake engine
    exposing ``dialect.name`` -- enough for ``_bound_engine`` / resolution
    without constructing a real (driver-requiring) engine."""

    def __init__(self, dialect_name):
        self.kw = {
            "bind": types.SimpleNamespace(
                dialect=types.SimpleNamespace(name=dialect_name)
            )
        }


# --------------------------------------------------------------------------
# resolve_dialect_strategy
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_real_sqlite_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        strategy = resolve_dialect_strategy(factory)
        assert strategy.name == "sqlite"
        assert isinstance(strategy, SqliteDialectStrategy)
        # the Protocol is satisfied (runtime_checkable surface)
        assert isinstance(strategy, SqlAlchemyDialectStrategy)
    finally:
        await engine.dispose()


def test_resolve_maps_mysql_and_postgresql():
    assert resolve_dialect_strategy(_FakeFactory("mysql")).name == "mysql"
    assert isinstance(
        resolve_dialect_strategy(_FakeFactory("mysql")), MySqlDialectStrategy
    )
    assert resolve_dialect_strategy(_FakeFactory("postgresql")).name == "postgresql"
    assert isinstance(
        resolve_dialect_strategy(_FakeFactory("postgresql")), PostgreSqlDialectStrategy
    )


def test_resolve_rejects_unknown_dialect():
    with pytest.raises(UnsupportedSqlAlchemyDialectError) as exc_info:
        resolve_dialect_strategy(_FakeFactory("oracle"))
    assert exc_info.value.dialect_name == "oracle"


@pytest.mark.asyncio
async def test_adapter_construction_resolves_dialect(tmp_path):
    # Constructing the adapter runs the construct-time gate: a real SQLite
    # factory resolves cleanly (the gate is at construction, not first write).
    from linktools.ai.storage.sqlalchemy.facade import SqlAlchemyStorage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/gate.db")
    try:
        async with engine.begin() as conn:
            from linktools.ai.storage.sqlalchemy.models import Base

            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        storage = SqlAlchemyStorage(session_factory=factory, blobs_root=tmp_path)
        assert storage._dialect_strategy.name == "sqlite"
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------
# SqliteDialectStrategy.classify_integrity_error
# --------------------------------------------------------------------------


def test_sqlite_classifier_recognizes_asset_and_idempotency_keys():
    strategy = SqliteDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(_Orig("UNIQUE constraint failed: ai_assets.path"))
    ) is IntegrityViolationKind.ASSET_KEY
    assert strategy.classify_integrity_error(
        _Err(_Orig("UNIQUE constraint failed: ai_asset_idempotency.key"))
    ) is IntegrityViolationKind.IDEMPOTENCY_KEY


def test_sqlite_classifier_returns_other_for_unrelated_unique_violation():
    strategy = SqliteDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(_Orig("UNIQUE constraint failed: ai_idempotency.scope, ai_idempotency.key"))
    ) is IntegrityViolationKind.OTHER
    # a non-unique IntegrityError (NOT NULL etc.) is also OTHER
    assert strategy.classify_integrity_error(
        _Err(_Orig("NOT NULL constraint failed: ai_assets.etag"))
    ) is IntegrityViolationKind.OTHER


# --------------------------------------------------------------------------
# PostgreSqlDialectStrategy.classify_integrity_error
# --------------------------------------------------------------------------


def test_postgresql_classifier_uses_constraint_name_from_diag():
    strategy = PostgreSqlDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(_Orig(constraint="uq_ai_assets_tenant_path"))
    ) is IntegrityViolationKind.ASSET_KEY
    assert strategy.classify_integrity_error(
        _Err(_Orig(constraint="uq_ai_asset_idempotency_tenant_key"))
    ) is IntegrityViolationKind.IDEMPOTENCY_KEY


def test_postgresql_classifier_falls_back_to_message_constraint_name():
    # asyncpg wording: the constraint name appears in the message text, not on
    # a diag attribute.
    strategy = PostgreSqlDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(
            _Orig(
                'duplicate key value violates unique constraint '
                '"uq_ai_assets_tenant_path"'
            )
        )
    ) is IntegrityViolationKind.ASSET_KEY


def test_postgresql_classifier_returns_other_for_unrelated_constraint():
    strategy = PostgreSqlDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(_Orig(constraint="uq_event_stream_sequence"))
    ) is IntegrityViolationKind.OTHER


# --------------------------------------------------------------------------
# MySqlDialectStrategy.classify_integrity_error
# --------------------------------------------------------------------------


def test_mysql_classifier_recognizes_duplicate_key_with_named_constraint():
    strategy = MySqlDialectStrategy()
    assert strategy.classify_integrity_error(
        _Err(
            _Orig(
                "Duplicate entry 'x' for key 'uq_ai_assets_tenant_path'",
                code=1062,
            )
        )
    ) is IntegrityViolationKind.ASSET_KEY
    assert strategy.classify_integrity_error(
        _Err(
            _Orig(
                "Duplicate entry 'y' for key 'uq_ai_asset_idempotency_tenant_key'",
                code=1062,
            )
        )
    ) is IntegrityViolationKind.IDEMPOTENCY_KEY


def test_mysql_classifier_returns_other_for_non_duplicate_key_code():
    strategy = MySqlDialectStrategy()
    # a non-1062 error (e.g. NOT NULL) is never mistaken for a unique conflict
    assert strategy.classify_integrity_error(
        _Err(_Orig(code=1048))
    ) is IntegrityViolationKind.OTHER


def test_mysql_classifier_returns_other_for_duplicate_key_unknown_constraint():
    strategy = MySqlDialectStrategy()
    # 1062 but on a constraint the asset domain does not own
    assert strategy.classify_integrity_error(
        _Err(_Orig("Duplicate entry 'z' for key 'uq_event_stream_sequence'", code=1062))
    ) is IntegrityViolationKind.OTHER
