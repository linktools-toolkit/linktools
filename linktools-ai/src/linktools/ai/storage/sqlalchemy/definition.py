#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyRunDefinitionStore: DB-backed RunDefinitionStore."""

import json
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RunDefinitionRow
from ...run.definition import RunDefinitionSnapshot


class SqlAlchemyRunDefinitionStore:
    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
    ) -> None:
        self._session_factory = session_factory

    async def create(self, snapshot: RunDefinitionSnapshot) -> None:
        from ...json import canonical_json

        async with self._session_factory() as session:
            async with session.begin():
                session.add(
                    RunDefinitionRow(
                        run_id=snapshot.run_id,
                        runnable_type=snapshot.runnable_type,
                        runnable_id=snapshot.runnable_id,
                        serialized_spec_json=canonical_json(snapshot.serialized_spec),
                        spec_fingerprint=snapshot.spec_fingerprint,
                        user_id=snapshot.user_id,
                        tenant_id=snapshot.tenant_id,
                        workspace=snapshot.workspace,
                        created_at=snapshot.created_at,
                        manifest_json=canonical_json(snapshot.manifest)
                        if snapshot.manifest
                        else "{}",
                        resumability=snapshot.resumability,
                    )
                )

    async def get(self, run_id: str) -> "RunDefinitionSnapshot | None":
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunDefinitionRow).where(RunDefinitionRow.run_id == run_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return RunDefinitionSnapshot(
                run_id=row.run_id,
                runnable_type=row.runnable_type,
                runnable_id=row.runnable_id,
                serialized_spec=json.loads(row.serialized_spec_json),
                spec_fingerprint=row.spec_fingerprint,
                user_id=row.user_id,
                tenant_id=row.tenant_id,
                workspace=row.workspace,
                provider_revision=None,
                created_at=row.created_at,
                manifest=json.loads(row.manifest_json) if row.manifest_json else {},
                resumability=row.resumability or "resumable",
            )
