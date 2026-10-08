#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom SQL dialects classify mutation failures through public contracts."""

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from linktools.ai import storage
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    SQLiteDialect,
    SqlAlchemyDialect,
    SqlStorageContext,
    SqlTransactionDisposition,
    SqlTransactionPhase,
)


class _ClassifiedFailure(Exception):
    pass


class _ApplicationDialect(SQLiteDialect):
    def __init__(self) -> None:
        self.failures: list[tuple[SqlTransactionPhase, bool]] = []

    def classify_transaction_error(
        self,
        error: BaseException,
        *,
        phase: SqlTransactionPhase,
        connection_invalidated: bool,
    ) -> SqlTransactionDisposition:
        if not isinstance(error, _ClassifiedFailure):
            return super().classify_transaction_error(
                error, phase=phase, connection_invalidated=connection_invalidated,
            )
        self.failures.append((phase, connection_invalidated))
        if phase is SqlTransactionPhase.BODY:
            return SqlTransactionDisposition.RETRYABLE_ABORTED
        assert phase is SqlTransactionPhase.COMMIT
        return SqlTransactionDisposition.COMMIT_UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", (SqlTransactionPhase.BODY, SqlTransactionPhase.COMMIT))
async def test_public_sql_classifier_preserves_retry_and_unknown_commit_semantics(
    phase: SqlTransactionPhase,
) -> None:
    assert "SqlTransactionPhase" in storage.__all__
    assert "SqlTransactionDisposition" in storage.__all__
    dialect = _ApplicationDialect()
    contract: SqlAlchemyDialect = dialect
    assert isinstance(contract, SqlAlchemyDialect)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = SqlStorageContext(engine, async_sessionmaker(engine), contract)
    failure = _ClassifiedFailure("classified by application")
    attempts = 0

    def fail_commit(connection: Connection) -> None:
        del connection
        raise failure

    async def mutate(session: AsyncSession) -> int:
        nonlocal attempts
        attempts += 1
        await session.execute(text("INSERT INTO ai_values (value) VALUES (1)"))
        if phase is SqlTransactionPhase.BODY and attempts == 1:
            raise failure
        return 42

    try:
        await context.initialize()
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
        if phase is SqlTransactionPhase.COMMIT:
            event.listen(engine.sync_engine, "commit", fail_commit)
            with pytest.raises(AIError) as raised:
                await context.run_mutation(mutate)
            assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert raised.value.__cause__ is failure
            assert attempts == 1
        else:
            assert await context.run_mutation(mutate) == 42
            async with engine.connect() as connection:
                rows = await connection.execute(text("SELECT value FROM ai_values"))
                assert rows.scalars().all() == [1]
            assert attempts == 2
        assert dialect.failures == [(phase, False)]
    finally:
        if phase is SqlTransactionPhase.COMMIT:
            event.remove(engine.sync_engine, "commit", fail_commit)
        await context.close()
        await engine.dispose()
