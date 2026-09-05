#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operation-scoped SQLite MetricStore."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import AIError, ErrorCode
from ..storage import namespace_digest
from ._codec import (
    decode_definition_envelope,
    decode_observation_envelope,
    definition_envelope,
    definition_semantic_digest,
    observation_digest,
    observation_envelope,
    observation_payload_digest,
)
from ._model import MetricDefinition, Observation

_BUSY_RETRY_LIMIT = 5
_BUSY_RETRY_DELAY = 0.02


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _load_json(value: object) -> object:
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc


class SQLiteMetricStore:
    def __init__(self, path: str | Path) -> None:
        if (
            not isinstance(path, (str, Path))
            or not str(path).strip()
            or str(path) == ":memory:"
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        self._path = str(Path(path).expanduser().resolve(strict=False))

    @staticmethod
    def _module() -> Any:
        try:
            import aiosqlite
        except ImportError as exc:
            raise AIError(ErrorCode.OPTIONAL_DEPENDENCY_MISSING) from exc
        return aiosqlite

    async def _connect(self) -> Any:
        module = self._module()
        connection = await module.connect(self._path, timeout=5.0)
        connection.row_factory = module.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @staticmethod
    def _translate(error: BaseException) -> AIError:
        message = str(error).lower()
        if "no such table" in message:
            return AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        return AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def _mutate(self, callback: Any) -> Any:
        module = self._module()
        for attempt in range(_BUSY_RETRY_LIMIT):
            connection = await self._connect()
            try:
                await connection.execute("BEGIN")
                result = await callback(connection)
                await connection.commit()
                return result
            except module.OperationalError as exc:
                await connection.rollback()
                message = str(exc).lower()
                if (
                    ("locked" in message or "busy" in message)
                    and attempt + 1 < _BUSY_RETRY_LIMIT
                ):
                    await asyncio.sleep(_BUSY_RETRY_DELAY * (attempt + 1))
                    continue
                raise self._translate(exc) from exc
            except BaseException:
                await connection.rollback()
                raise
            finally:
                await connection.close()
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> None:
        namespace_key = namespace_digest(namespace)
        semantic_digest = definition_semantic_digest(definition)
        payload = _json(definition_envelope(namespace, definition))

        async def mutate(connection: Any) -> None:
            await connection.execute(
                """
                INSERT OR IGNORE INTO ai_metric_definitions (
                    namespace_digest, metric_name, revision, definition_digest,
                    observation_kind, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace_key,
                    definition.name,
                    definition.revision,
                    semantic_digest,
                    definition.observation_kind,
                    payload,
                ),
            )
            cursor = await connection.execute(
                """
                SELECT namespace_digest, metric_name, revision, definition_digest,
                       observation_kind, payload_json
                FROM ai_metric_definitions
                WHERE namespace_digest = ? AND metric_name = ? AND revision = ?
                """,
                (namespace_key, definition.name, definition.revision),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored = decode_definition_envelope(
                _load_json(row["payload_json"]),
                expected_namespace=namespace,
            )
            if (
                row["namespace_digest"] != namespace_key
                or row["metric_name"] != definition.name
                or row["revision"] != definition.revision
                or row["observation_kind"] != stored.observation_kind
                or definition_semantic_digest(stored) != row["definition_digest"]
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if str(row["definition_digest"]) != semantic_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._mutate(mutate)

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int | None,
    ) -> MetricDefinition | None:
        connection = await self._connect()
        try:
            if revision is None:
                cursor = await connection.execute(
                    """
                    SELECT metric_name, revision, definition_digest,
                           observation_kind, payload_json
                    FROM ai_metric_definitions
                    WHERE namespace_digest = ? AND metric_name = ?
                    ORDER BY revision DESC
                    LIMIT 1
                    """,
                    (namespace_digest(namespace), name),
                )
            else:
                cursor = await connection.execute(
                    """
                    SELECT metric_name, revision, definition_digest,
                           observation_kind, payload_json
                    FROM ai_metric_definitions
                    WHERE namespace_digest = ? AND metric_name = ? AND revision = ?
                    LIMIT 1
                    """,
                    (namespace_digest(namespace), name, revision),
                )
            row = await cursor.fetchone()
            await cursor.close()
        except self._module().OperationalError as exc:
            raise self._translate(exc) from exc
        finally:
            await connection.close()
        if row is None:
            return None
        definition = decode_definition_envelope(
            _load_json(row["payload_json"]), expected_namespace=namespace
        )
        if (
            definition.name != row["metric_name"]
            or definition.revision != row["revision"]
            or definition.observation_kind != row["observation_kind"]
            or definition_semantic_digest(definition) != row["definition_digest"]
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return definition

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        collapsed: dict[str, tuple[str, Observation]] = {}
        for observation in observations:
            identity = observation_digest(namespace, observation.observation_id)
            payload_digest = observation_payload_digest(namespace, observation)
            current = collapsed.get(identity)
            if current is not None and current[0] != payload_digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            collapsed[identity] = (payload_digest, observation)
        if not collapsed:
            return
        namespace_key = namespace_digest(namespace)

        async def mutate(connection: Any) -> None:
            await connection.executemany(
                """
                INSERT OR IGNORE INTO ai_metric_observations (
                    namespace_digest, observation_digest, payload_digest,
                    kind, occurred_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        namespace_key,
                        identity,
                        payload_digest,
                        observation.kind,
                        observation.occurred_at.isoformat(),
                        _json(observation_envelope(namespace, observation)),
                    )
                    for identity, (payload_digest, observation) in collapsed.items()
                ],
            )
            placeholders = ",".join("?" for _ in collapsed)
            cursor = await connection.execute(
                f"""
                SELECT observation_digest, payload_digest
                FROM ai_metric_observations
                WHERE observation_digest IN ({placeholders})
                """,
                tuple(collapsed),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            existing = {
                str(row["observation_digest"]): str(row["payload_digest"])
                for row in rows
            }
            if len(existing) != len(collapsed):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            for identity, (payload_digest, _) in collapsed.items():
                if existing.get(identity) != payload_digest:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._mutate(mutate)

    async def scan_observations(
        self,
        namespace: str,
        *,
        kind: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[Observation, ...]:
        connection = await self._connect()
        try:
            cursor = await connection.execute(
                """
                SELECT namespace_digest, observation_digest, payload_digest,
                       kind, occurred_at, payload_json
                FROM ai_metric_observations
                WHERE namespace_digest = ? AND kind = ?
                  AND occurred_at >= ? AND occurred_at < ?
                ORDER BY occurred_at, observation_digest
                LIMIT ?
                """,
                (
                    namespace_digest(namespace),
                    kind,
                    start.isoformat(),
                    end.isoformat(),
                    limit,
                ),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except self._module().OperationalError as exc:
            raise self._translate(exc) from exc
        finally:
            await connection.close()
        namespace_key = namespace_digest(namespace)
        values = []
        for row in rows:
            observation = decode_observation_envelope(
                _load_json(row["payload_json"]), expected_namespace=namespace
            )
            if (
                row["namespace_digest"] != namespace_key
                or row["kind"] != observation.kind
                or row["occurred_at"] != observation.occurred_at.isoformat()
                or row["observation_digest"]
                != observation_digest(namespace, observation.observation_id)
                or row["payload_digest"]
                != observation_payload_digest(namespace, observation)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(observation)
        return tuple(values)

    async def prune(self, namespace: str, *, before: datetime) -> int:
        async def mutate(connection: Any) -> int:
            cursor = await connection.execute(
                """
                DELETE FROM ai_metric_observations
                WHERE namespace_digest = ? AND occurred_at < ?
                """,
                (namespace_digest(namespace), before.isoformat()),
            )
            return int(cursor.rowcount)

        return int(await self._mutate(mutate))


__all__ = ["SQLiteMetricStore"]
