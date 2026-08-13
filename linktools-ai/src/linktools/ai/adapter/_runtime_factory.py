#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter-owned materialization of declarative Runtime storage targets."""

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic_ai_harness.step_persistence import StepStore

from ..errors import AIError, ErrorCode
from ..runtime import RuntimeDomain, RuntimeRetention, RuntimeStorage, RuntimeStores
from ..storage import ObjectRef, ObjectStore, TransientObjectStore
from ._persistence import (
    ApprovalRepositoryBackend,
    ArtifactRepositoryBackend,
    EvaluationRepositoryBackend,
    EventRepositoryBackend,
    ExecutionRepositoryBackend,
    ExternalRepositoryBackend,
    IdempotencyRepositoryBackend,
    MemoryRepositoryBackend,
    OperationRepositoryBackend,
    RecoveryCheckpointRepositoryBackend,
    RuntimeTransactionBinding,
    SessionRepositoryBackend,
    TaskRepositoryBackend,
    ToolRepositoryBackend,
    build_filesystem_runtime,
    build_in_memory_runtime,
    recovery_checkpoint_from_json,
)
from ._schema import build_runtime_sql_metadata
from ._step import (
    FilesystemStepArchive,
    RuntimeStepPersistence,
    SqlStepArchive,
    StagingStepStore,
)

_logger = environ.get_logger("ai.adapter.runtime_factory")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..storage import SqlContext


@dataclass(frozen=True, slots=True)
class RuntimePersistence:
    domain: RuntimeStores
    steps: StepStore
    close_callback: Callable[[], Awaitable[None]]
    promote_callback: Callable[[RuntimeDomain, str], Awaitable[None]]
    promote_from_callback: Callable[[RuntimeDomain, RuntimeDomain, str], Awaitable[None]]
    release_callback: Callable[[str], Awaitable[None]]

    async def close(self) -> None:
        callback = self.close_callback
        await callback()

    async def promote(self, runtime_domain: RuntimeDomain, run_id: str) -> None:
        await self.promote_callback(runtime_domain, run_id=run_id)

    async def promote_from(self, source_domain: RuntimeDomain, target_domain: RuntimeDomain, run_id: str) -> None:
        await self.promote_from_callback(source_domain, target_domain, run_id=run_id)

    async def release_transient(self, run_id: str) -> None:
        await self.release_callback(run_id)


@asynccontextmanager
async def open_runtime_persistence(
    storage: RuntimeStorage,
    *,
    namespace: str,
    tenant_id: str,
    sql_context: "SqlContext | None" = None,
) -> AsyncIterator[RuntimePersistence]:
    """Materialize one fixed namespace/tenant Runtime persistence boundary."""

    durable = frozenset(
        domain
        for domain in RuntimeDomain
        if storage.plan.route(domain).retention is RuntimeRetention.DURABLE
    )
    target_kind = storage.target_kind
    owned_context = False
    if target_kind in {"sqlite", "sql"} and sql_context is None:
        from sqlalchemy.ext.asyncio import create_async_engine

        from ..storage import create_sql_context

        if target_kind == "sqlite":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
            sql_context = create_sql_context(engine, owns_engine=True)
        else:
            engine = storage.target_engine
            if engine is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            sql_context = create_sql_context(engine)
        owned_context = True
    runtime = None
    steps: RuntimeStepPersistence | None = None
    try:
        if target_kind in {"sqlite", "sql"}:
            if sql_context is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            metadata = build_runtime_sql_metadata(storage.plan)
            from ..storage import provision_sql, validate_sql

            if target_kind == "sqlite":
                await provision_sql(sql_context.engine, metadata)
            else:
                await validate_sql(sql_context.engine, metadata)
        if target_kind == "memory":
            runtime = build_in_memory_runtime(namespace=namespace)
        elif target_kind == "filesystem":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            runtime = build_filesystem_runtime(str(path), namespace=namespace, persist=durable)
            from ._persistence import RuntimeObjectRouter

            runtime.persistence = replace(
                runtime.persistence,
                object_router=RuntimeObjectRouter(_working_object_stores(storage, runtime.persistence)),
            )
        elif target_kind in {"sqlite", "sql"} and sql_context is not None:
            runtime = await _open_sql_runtime(storage, namespace=namespace, tenant_id=tenant_id, sql_context=sql_context, durable=durable)
        else:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await runtime.initialize()
        archives = {}
        if target_kind == "filesystem":
            path = storage.target_path
            if path is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if domain in durable:
                    archives[domain] = FilesystemStepArchive._runtime(
                        path,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        runtime_domain=domain,
                        object_store=runtime.persistence.object_store(domain),
                        writer_lock=runtime.writer_lock,
                    )
                elif storage.plan.route(domain).retention is RuntimeRetention.VOLATILE:
                    archives[domain] = StagingStepStore()
        elif target_kind in {"sqlite", "sql"}:
            if sql_context is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if domain in durable:
                    archives[domain] = SqlStepArchive._runtime(
                        sql_context.engine,
                        namespace=namespace,
                        tenant_id=tenant_id,
                        runtime_domain=domain,
                        object_store=runtime.persistence.object_store(domain),
                        context=sql_context,
                    )
                elif storage.plan.route(domain).retention is RuntimeRetention.VOLATILE:
                    archives[domain] = StagingStepStore()
        elif target_kind == "memory":
            from ._persistence import RuntimeObjectRouter

            runtime.persistence = replace(
                runtime.persistence,
                object_router=RuntimeObjectRouter(_working_object_stores(storage, runtime.persistence)),
            )
            for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
                if storage.plan.route(domain).retention is RuntimeRetention.VOLATILE:
                    archives[domain] = StagingStepStore()
        steps = RuntimeStepPersistence(StagingStepStore(), archives)
        await steps.initialize()
    except BaseException:
        if steps is not None:
            try:
                await steps.close()
            except BaseException:
                _logger.error("runtime step cleanup failed during setup", exc_info=environ.debug)
        if runtime is not None:
            try:
                runtime.set_close_hook(None)
                await runtime.close()
            except BaseException:
                _logger.error("runtime cleanup failed during setup", exc_info=environ.debug)
        if owned_context and sql_context is not None:
            try:
                await sql_context.close()
            except BaseException:
                _logger.error("SQL context cleanup failed during setup", exc_info=environ.debug)
        raise
    if steps is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    transient_domains = frozenset(
        domain
        for domain in RuntimeDomain
        if storage.plan.route(domain).retention is RuntimeRetention.TRANSIENT
    )

    async def release_transient(run_id: str) -> None:
        release_domains = transient_domains - {RuntimeDomain.CONVERSATION}
        if release_domains:
            await runtime.release(release_domains)
        if release_domains:
            await steps.release(run_id, release_domains)

    persistence = RuntimePersistence(runtime.persistence, steps, runtime.close, steps.promote, steps.promote_from, release_transient)
    try:
        yield persistence
    finally:
        failure: BaseException | None = None
        for callback, label in (
            (steps.close, "runtime steps"),
            (persistence.close, "runtime stores"),
        ):
            try:
                await callback()
            except BaseException as error:
                if failure is None:
                    failure = error
                _logger.error("runtime cleanup failed: resource=%s", label, exc_info=environ.debug)
        if owned_context and sql_context is not None:
            try:
                await sql_context.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                _logger.error("SQL context cleanup failed", exc_info=environ.debug)
        if failure is not None:
            raise failure


__all__ = ["RuntimePersistence", "open_runtime_persistence"]


def runtime_storage_kind(storage: RuntimeStorage) -> str:
    return storage.target_kind


def runtime_storage_path(storage: RuntimeStorage) -> Path | None:
    return storage.target_path


def runtime_storage_engine(storage: RuntimeStorage) -> "AsyncEngine | None":
    return storage.target_engine


def runtime_durable_domains(storage: RuntimeStorage) -> frozenset[RuntimeDomain]:
    return frozenset(domain for domain in RuntimeDomain if storage.plan.route(domain).retention is RuntimeRetention.DURABLE)


__all__ += ["runtime_durable_domains", "runtime_storage_engine", "runtime_storage_kind", "runtime_storage_path"]


def _working_object_stores(storage: RuntimeStorage, persistence: RuntimeStores) -> dict[RuntimeDomain, ObjectStore]:
    stores: dict[RuntimeDomain, ObjectStore] = {}
    for domain in RuntimeDomain:
        route = storage.plan.route(domain)
        if route.object_store is not None:
            stores[domain] = route.object_store
        elif route.retention is RuntimeRetention.TRANSIENT:
            stores[domain] = TransientObjectStore()
        else:
            stores[domain] = persistence.object_store(domain)
    return stores


async def _open_sql_runtime(
    storage: RuntimeStorage,
    *,
    namespace: str,
    tenant_id: str,
    sql_context: "SqlContext",
    durable: frozenset[RuntimeDomain],
) -> object:
    from ..storage import InMemoryObjectStore, SqlObjectStore, TransientObjectStore
    from ._persistence import RuntimeObjectRouter
    runtime_holder: dict[str, object] = {}

    async def commit(_: RuntimeDomain) -> None:
        runtime = runtime_holder.get("runtime")
        if runtime is None or not hasattr(runtime, "persistence"):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            await _flush_sql_runtime(runtime.persistence, sql_context, namespace=namespace, tenant_id=tenant_id, durable=durable)
        except AIError:
            raise
        except Exception as error:
            from sqlalchemy.exc import SQLAlchemyError

            if isinstance(error, SQLAlchemyError):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            raise

    transaction_binding = RuntimeTransactionBinding(commit_callback=commit)
    runtime = build_in_memory_runtime(namespace=namespace, transaction_binding=transaction_binding)
    runtime_holder["runtime"] = runtime
    object_stores: dict[RuntimeDomain, object] = {}
    builtin_sql_store = SqlObjectStore._from_context(sql_context)
    for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY):
        route = storage.plan.route(domain)
        if route.object_store is not None:
            object_stores[domain] = route.object_store
        elif domain in durable:
            object_stores[domain] = builtin_sql_store
        else:
            object_stores[domain] = InMemoryObjectStore()
    for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY):
        if storage.plan.route(domain).retention is RuntimeRetention.TRANSIENT and storage.plan.route(domain).object_store is None:
            object_stores[domain] = TransientObjectStore()
    object_stores.setdefault(RuntimeDomain.TASK, InMemoryObjectStore())
    object_stores.setdefault(RuntimeDomain.EVALUATION, InMemoryObjectStore())
    domain = runtime.persistence
    runtime.persistence = replace(
        domain,
        object_router=RuntimeObjectRouter(object_stores),
    )
    try:
        runtime.persistence = await _load_sql_runtime(runtime.persistence, sql_context, namespace=namespace, tenant_id=tenant_id, durable=durable)
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def close_commit() -> None:
        await commit(RuntimeDomain.CONVERSATION)

    runtime.set_close_hook(close_commit)
    return runtime


async def _load_sql_runtime(domain: RuntimeStores, context: "SqlContext", *, namespace: str, tenant_id: str, durable: frozenset[RuntimeDomain]) -> RuntimeStores:
    from sqlalchemy import select

    from ..core import (
        ApprovalDecision,
        ApprovalStatus,
        EvaluationStatus,
        ExecutionEventType,
        ExecutionLineageKind,
        ExecutionStatus,
        ExternalCallStatus,
        IdempotencyStatus,
        OperationKind,
        OperationStatus,
        ResourceKind,
        SessionStatus,
        StopReason,
        TaskStatus,
        ToolOperationStatus,
    )
    from ..runtime import (
        ApprovalRecord,
        ArtifactRecord,
        ConversationCursor,
        EvaluationRecord,
        ExecutionEventRecord,
        ExecutionRecord,
        ExternalCallRecord,
        IdempotencyRecord,
        MemoryRecord,
        OperationLedgerRecord,
        ResultRecord,
        SessionRecord,
        TaskNodeView,
        ToolOperationRecord,
    )
    from ..task import TaskGraphView, TaskNode

    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    sessions = domain.conversation.sessions
    executions = domain.execution.executions
    if RuntimeDomain.CONVERSATION in durable and isinstance(sessions, SessionRepositoryBackend):
        table = build_runtime_sql_metadata(domain_plan(durable)).tables["runtime_sessions"]
        async with context.sessions() as connection:
            rows = (await connection.execute(select(table).where(table.c.namespace_key == namespace_digest, table.c.tenant_id == tenant_id))).mappings().all()
        for row in rows:
            cursor = None if row["continuation_step_run_id"] is None else ConversationCursor(str(row["continuation_step_run_id"]))
            sessions._records[(tenant_id, str(row["session_id"]))] = SessionRecord(str(row["session_id"]), tenant_id, str(row["owner_principal_id"]), str(row["binding_digest"]), SessionStatus(str(row["status"])), int(row["revision"]), int(row["resource_generation"]), None if row["cwd"] is None else str(row["cwd"]), row["metadata_json"] or {}, _utc(row["created_at"]), _utc(row["updated_at"]), None if row["closed_at"] is None else _utc(row["closed_at"]), cursor)
    if RuntimeDomain.EXECUTION in durable and isinstance(executions, ExecutionRepositoryBackend):
        table = build_runtime_sql_metadata(domain_plan(durable)).tables["runtime_executions"]
        async with context.sessions() as connection:
            rows = (await connection.execute(select(table).where(table.c.namespace_key == namespace_digest, table.c.tenant_id == tenant_id))).mappings().all()
        for row in rows:
            executions._records[(tenant_id, str(row["execution_id"]))] = ExecutionRecord(
                execution_id=str(row["execution_id"]),
                tenant_id=tenant_id,
                session_id=None if row["session_id"] is None else str(row["session_id"]),
                binding_digest=str(row["binding_digest"]),
                parent_execution_id=None if row["parent_execution_id"] is None else str(row["parent_execution_id"]),
                root_execution_id=str(row["root_execution_id"]),
                source_execution_id=None if row["source_execution_id"] is None else str(row["source_execution_id"]),
                base_execution_id=None if row["base_execution_id"] is None else str(row["base_execution_id"]),
                lineage_kind=ExecutionLineageKind(str(row["lineage_kind"])),
                status=ExecutionStatus(str(row["status"])),
                revision=int(row["revision"]),
                event_sequence=int(row["event_sequence"]),
                agent_run_sequence=int(row["agent_run_sequence"]),
                error_code=None if row["error_code"] is None else str(row["error_code"]),
                safe_error_details=row["safe_error_details_json"] or {},
                created_at=_utc(row["created_at"]),
                updated_at=_utc(row["updated_at"]),
                memory_scope=None if row["memory_scope"] is None else str(row["memory_scope"]),
                conversation_step_run_id=None if row["conversation_step_run_id"] is None else str(row["conversation_step_run_id"]),
            )
        terminal = executions._terminal
        if terminal is not None:
            for row in rows:
                status = ExecutionStatus(str(row["status"]))
                result_fields = tuple(row[name] for name in ("result_store_id", "result_object_key", "result_digest", "result_size"))
                accounting_fields = tuple(row[name] for name in ("stop_reason", "input_tokens", "output_tokens", "total_cost_micros", "result_created_at"))
                schema_fields = tuple(row[name] for name in ("output_schema_id", "output_schema_revision", "output_schema_fingerprint"))
                has_result = any(value is not None for value in accounting_fields)
                if status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    if not has_result or any(value is None for value in accounting_fields):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif has_result or any(value is not None for value in result_fields) or any(value is not None for value in schema_fields):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if not has_result:
                    continue
                if any(value is None for value in result_fields) and any(value is not None for value in result_fields):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                object_ref = None if all(value is None for value in result_fields) else _object_ref_from_row(row, "result_")
                has_schema = all(value is not None for value in schema_fields)
                if status is ExecutionStatus.SUCCEEDED and (not has_schema or object_ref is None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if status is not ExecutionStatus.SUCCEEDED and (has_schema or object_ref is not None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if object_ref is not None:
                    await _validate_object_reference(domain.object_store(RuntimeDomain.EXECUTION), object_ref)
                try:
                    result = ResultRecord(
                        execution_id=str(row["execution_id"]),
                        tenant_id=tenant_id,
                        output_schema_id=None if row["output_schema_id"] is None else str(row["output_schema_id"]),
                        output_schema_revision=None if row["output_schema_revision"] is None else int(row["output_schema_revision"]),
                        output_schema_fingerprint=None if row["output_schema_fingerprint"] is None else str(row["output_schema_fingerprint"]),
                        object_ref=object_ref,
                        stop_reason=StopReason(str(row["stop_reason"])),
                        input_tokens=int(row["input_tokens"]),
                        output_tokens=int(row["output_tokens"]),
                        total_cost_micros=int(row["total_cost_micros"]),
                        created_at=_utc(row["result_created_at"]),
                    )
                except AIError:
                    raise
                except (KeyError, TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                terminal._results[(tenant_id, str(row["execution_id"]))] = result
    if RuntimeDomain.EXECUTION in durable:
        metadata = build_runtime_sql_metadata(domain_plan(durable))
        async with context.sessions() as connection:
            if "runtime_events" in metadata.tables:
                event_rows = (await connection.execute(select(metadata.tables["runtime_events"]).where(metadata.tables["runtime_events"].c.namespace_key == namespace_digest, metadata.tables["runtime_events"].c.tenant_id == tenant_id).order_by(metadata.tables["runtime_events"].c.sequence))).mappings().all()
            else:
                event_rows = ()
            if "runtime_idempotency" in metadata.tables:
                identity_rows = (await connection.execute(select(metadata.tables["runtime_idempotency"]).where(metadata.tables["runtime_idempotency"].c.namespace_key == namespace_digest, metadata.tables["runtime_idempotency"].c.tenant_id == tenant_id, metadata.tables["runtime_idempotency"].c.runtime_domain == RuntimeDomain.EXECUTION.value))).mappings().all()
            else:
                identity_rows = ()
            if "runtime_operations" in metadata.tables:
                operation_rows = (await connection.execute(select(metadata.tables["runtime_operations"]).where(metadata.tables["runtime_operations"].c.namespace_key == namespace_digest, metadata.tables["runtime_operations"].c.tenant_id == tenant_id, metadata.tables["runtime_operations"].c.runtime_domain == RuntimeDomain.EXECUTION.value))).mappings().all()
            else:
                operation_rows = ()
        if isinstance(domain.execution.events, EventRepositoryBackend):
            for row in event_rows:
                item = ExecutionEventRecord(str(row["execution_id"]), tenant_id, int(row["sequence"]), ExecutionEventType(str(row["event_type"])), row["payload_json"] or {})
                domain.execution.events._items.setdefault((tenant_id, item.execution_id), []).append(item)
        if isinstance(domain.execution.idempotency, IdempotencyRepositoryBackend):
            for row in identity_rows:
                item = IdempotencyRecord(tenant_id, str(row["scope"]), str(row["key_hash"]), str(row["request_digest"]), str(row["resource_id"]), IdempotencyStatus(str(row["status"])), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), _utc(row["created_at"]), _utc(row["updated_at"]))
                domain.execution.idempotency._records[(tenant_id, item.scope, item.key_hash)] = item
        execution_operations = domain.execution.operations
        if isinstance(execution_operations, OperationRepositoryBackend):
            for row in operation_rows:
                item = OperationLedgerRecord(str(row["operation_id"]), tenant_id, ResourceKind(str(row["resource_kind"])), str(row["resource_id"]), None if row["execution_id"] is None else str(row["execution_id"]), OperationKind(str(row["operation_kind"])), OperationStatus(str(row["status"])), str(row["request_digest"]), None if row["result_ref"] is None else str(row["result_ref"]), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), bool(row["compactable"]), int(row["sequence"]), _utc(row["created_at"]), _utc(row["updated_at"]))
                execution_operations._records[(tenant_id, item.operation_id)] = item
    metadata = build_runtime_sql_metadata(domain_plan(durable))
    async with context.sessions() as connection:
        async def rows_for(name: str) -> list[Mapping[str, object]]:
            if name not in metadata.tables:
                return []
            table = metadata.tables[name]
            return list((await connection.execute(select(table).where(table.c.namespace_key == namespace_digest, table.c.tenant_id == tenant_id))).mappings().all())

        memory_rows = await rows_for("runtime_memories") if RuntimeDomain.MEMORY in durable else []
        artifact_rows = await rows_for("runtime_artifacts") if RuntimeDomain.ARTIFACT in durable else []
        task_graph_rows = await rows_for("runtime_task_graphs") if RuntimeDomain.TASK in durable else []
        task_node_rows = await rows_for("runtime_task_nodes") if RuntimeDomain.TASK in durable else []
        evaluation_rows = await rows_for("runtime_evaluations") if RuntimeDomain.EVALUATION in durable else []
        approval_rows = await rows_for("runtime_approvals") if RuntimeDomain.RECOVERY in durable else []
        external_rows = await rows_for("runtime_external_calls") if RuntimeDomain.RECOVERY in durable else []
        checkpoint_rows = await rows_for("runtime_recovery_checkpoints") if RuntimeDomain.RECOVERY in durable else []
        tool_rows = await rows_for("runtime_tool_operations") if RuntimeDomain.RECOVERY in durable else []
        counter_rows = await rows_for("runtime_operation_counters")
        operation_rows_all = await rows_for("runtime_operations")
        evaluation_identity_rows = []
        if RuntimeDomain.EVALUATION in durable and "runtime_idempotency" in metadata.tables:
            table = metadata.tables["runtime_idempotency"]
            evaluation_identity_rows = list((await connection.execute(select(table).where(table.c.namespace_key == namespace_digest, table.c.tenant_id == tenant_id, table.c.runtime_domain == RuntimeDomain.EVALUATION.value))).mappings().all())
    if isinstance(domain.memory.records, MemoryRepositoryBackend):
        for row in memory_rows:
            reference = _object_ref_from_row(row, "object_")
            await _validate_object_reference(domain.object_store(RuntimeDomain.MEMORY), reference)
            record = MemoryRecord(str(row["memory_id"]), tenant_id, str(row["memory_scope_key"]), reference, str(row["object_digest"]), row["metadata_json"] or {}, int(row["revision"]), _utc(row["created_at"]), _utc(row["updated_at"]))
            domain.memory.records._records[(tenant_id, record.memory_id)] = record
    if isinstance(domain.artifact.records, ArtifactRepositoryBackend):
        for row in artifact_rows:
            reference = _object_ref_from_row(row, "object_")
            await _validate_object_reference(domain.object_store(RuntimeDomain.ARTIFACT), reference)
            record = ArtifactRecord(str(row["artifact_id"]), str(row["execution_id"]), tenant_id, str(row["producer"]), str(row["media_type"]), int(row["object_size"]), str(row["object_digest"]), reference, _utc(row["created_at"]))
            domain.artifact.records._records[(tenant_id, record.artifact_id)] = record
    if isinstance(domain.task.tasks, TaskRepositoryBackend):
        nodes_by_graph: dict[str, list[TaskNode]] = {}
        for row in task_node_rows:
            node = TaskNode(str(row["node_id"]), tuple(row["dependencies_json"] or ()))
            nodes_by_graph.setdefault(str(row["graph_id"]), []).append(node)
            domain.task.tasks._nodes[(tenant_id, str(row["graph_id"]), node.node_id)] = TaskNodeView(str(row["graph_id"]), node.node_id, node.dependencies, TaskStatus(str(row["status"])), None if row["owner"] is None else str(row["owner"]), int(row["fence"]), None if row["lease_expires_at"] is None else _utc(row["lease_expires_at"]), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), None if row["error_digest"] is None else str(row["error_digest"]), None if row["execution_id"] is None else str(row["execution_id"]))
        for row in task_graph_rows:
            graph_id = str(row["graph_id"])
            domain.task.tasks._plans[(tenant_id, graph_id)] = TaskGraphView(graph_id, TaskStatus(str(row["status"])), tuple(nodes_by_graph.get(graph_id, ())))
    if isinstance(domain.evaluation.records, EvaluationRepositoryBackend):
        for row in evaluation_rows:
            record = EvaluationRecord(str(row["evaluation_id"]), tenant_id, str(row["execution_id"]), str(row["dataset_id"]), int(row["dataset_revision"]), str(row["evaluator_id"]), int(row["evaluator_revision"]), str(row["binding_digest"]), str(row["output_schema_fingerprint"]), None if row["artifact_digest"] is None else str(row["artifact_digest"]), EvaluationStatus(str(row["status"])), int(row["revision"]), row["metrics_json"] or {}, _utc(row["created_at"]), _utc(row["updated_at"]))
            domain.evaluation.records._records[(tenant_id, record.evaluation_id)] = record
    if isinstance(domain.evaluation.idempotency, IdempotencyRepositoryBackend):
        for row in evaluation_identity_rows:
            item = IdempotencyRecord(tenant_id, str(row["scope"]), str(row["key_hash"]), str(row["request_digest"]), str(row["resource_id"]), IdempotencyStatus(str(row["status"])), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), _utc(row["created_at"]), _utc(row["updated_at"]))
            domain.evaluation.idempotency._records[(tenant_id, item.scope, item.key_hash)] = item
    if isinstance(domain.recovery.approvals, ApprovalRepositoryBackend):
        for row in approval_rows:
            record = ApprovalRecord(str(row["approval_id"]), str(row["execution_id"]), tenant_id, str(row["operation_id"]), ApprovalStatus(str(row["status"])), None if row["idempotency_key_hash"] is None else str(row["idempotency_key_hash"]), None if row["decision"] is None else ApprovalDecision(str(row["decision"])), None if row["decided_by"] is None else str(row["decided_by"]), None if row["decision_digest"] is None else str(row["decision_digest"]), _utc(row["created_at"]), None if row["decided_at"] is None else _utc(row["decided_at"]))
            domain.recovery.approvals._records[(tenant_id, record.approval_id)] = record
    if isinstance(domain.recovery.external_calls, ExternalRepositoryBackend):
        for row in external_rows:
            reference = None if all(row[name] is None for name in ("object_store_id", "object_key", "object_digest", "object_size")) else _object_ref_from_row(row, "object_")
            if reference is not None:
                await _validate_object_reference(domain.object_store(RuntimeDomain.RECOVERY), reference)
            record = ExternalCallRecord(str(row["call_id"]), str(row["execution_id"]), tenant_id, str(row["operation_id"]), ExternalCallStatus(str(row["status"])), None if row["idempotency_key_hash"] is None else str(row["idempotency_key_hash"]), reference, None if row["payload_digest"] is None else str(row["payload_digest"]), _utc(row["created_at"]), None if row["supplied_at"] is None else _utc(row["supplied_at"]))
            domain.recovery.external_calls._records[(tenant_id, record.call_id)] = record
    if isinstance(domain.recovery.checkpoints, RecoveryCheckpointRepositoryBackend):
        for row in checkpoint_rows:
            payload = dict(row)
            payload.update(input=row["input_json"], terminal_handoff=row["terminal_handoff_json"])
            record = recovery_checkpoint_from_json(payload)
            if record.terminal_handoff is not None and record.terminal_handoff.outcome.recovery_object_ref is not None:
                await _validate_object_reference(domain.object_store(RuntimeDomain.RECOVERY), record.terminal_handoff.outcome.recovery_object_ref)
            domain.recovery.checkpoints._records[(tenant_id, record.execution_id)] = record
    if isinstance(domain.recovery.tools, ToolRepositoryBackend):
        for row in tool_rows:
            reference = None if all(row[name] is None for name in ("result_store_id", "result_object_key", "result_digest", "result_size")) else _object_ref_from_row(row, "result_")
            if reference is not None:
                await _validate_object_reference(domain.object_store(RuntimeDomain.RECOVERY), reference)
            record = ToolOperationRecord(str(row["tool_operation_id"]), tenant_id, str(row["step_run_id"]), str(row["tool_call_id"]), str(row["idempotency_key_hash"]), str(row["tool_name"]), str(row["arguments_hash"]), str(row["binding_fingerprint"]), bool(row["replay_safe"]), ToolOperationStatus(str(row["status"])), None if row["owner"] is None else str(row["owner"]), int(row["fence"]), None if row["lease_expires_at"] is None else _utc(row["lease_expires_at"]), reference, None if row["error_code"] is None else str(row["error_code"]), _utc(row["created_at"]), _utc(row["updated_at"]))
            domain.recovery.tools._records[(tenant_id, record.tool_operation_id)] = record
    operation_repositories = {
        RuntimeDomain.CONVERSATION: domain.conversation.operations,
        RuntimeDomain.EXECUTION: domain.execution.operations,
        RuntimeDomain.MEMORY: domain.memory.operations,
        RuntimeDomain.ARTIFACT: domain.artifact.operations,
        RuntimeDomain.TASK: domain.task.operations,
        RuntimeDomain.EVALUATION: domain.evaluation.operations,
        RuntimeDomain.RECOVERY: domain.recovery.operations,
    }
    for row in operation_rows_all:
        runtime_domain = RuntimeDomain(str(row["runtime_domain"]))
        repository = operation_repositories[runtime_domain]
        if isinstance(repository, OperationRepositoryBackend):
            item = OperationLedgerRecord(str(row["operation_id"]), tenant_id, ResourceKind(str(row["resource_kind"])), str(row["resource_id"]), None if row["execution_id"] is None else str(row["execution_id"]), OperationKind(str(row["operation_kind"])), OperationStatus(str(row["status"])), str(row["request_digest"]), None if row["result_ref"] is None else str(row["result_ref"]), None if row["result_digest"] is None else str(row["result_digest"]), None if row["error_code"] is None else str(row["error_code"]), bool(row["compactable"]), int(row["sequence"]), _utc(row["created_at"]), _utc(row["updated_at"]))
            repository._records[(tenant_id, item.operation_id)] = item
    for row in counter_rows:
        runtime_domain = RuntimeDomain(str(row["runtime_domain"]))
        repository = operation_repositories[runtime_domain]
        if isinstance(repository, OperationRepositoryBackend):
            repository._counters[(tenant_id, str(row["resource_kind"]), str(row["resource_id"]))] = int(row["last_sequence"])
    return domain


async def _flush_sql_runtime(
    domain: RuntimeStores,
    context: "SqlContext",
    *,
    namespace: str,
    tenant_id: str,
    durable: frozenset[RuntimeDomain],
) -> None:
    from sqlalchemy import delete, insert

    from ..core import ResourceKind

    metadata = build_runtime_sql_metadata(domain_plan(durable))
    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()

    def where_clause(table):
        return (table.c.namespace_key == namespace_digest, table.c.tenant_id == tenant_id)

    async with context.sessions.begin() as connection:
        for table_name, table in metadata.tables.items():
            if table_name.startswith("runtime_"):
                await connection.execute(delete(table).where(*where_clause(table)))
        if RuntimeDomain.CONVERSATION in durable:
            table = metadata.tables["runtime_sessions"]
            for record in domain.conversation.sessions._records.values():
                if record.tenant_id != tenant_id:
                    continue
                await connection.execute(insert(table).values(
                    namespace_key=namespace_digest, tenant_id=tenant_id, session_id=record.session_id,
                    owner_principal_id=record.owner_principal_id, binding_digest=record.binding_digest,
                    status=record.status.value, revision=record.revision, resource_generation=record.resource_generation,
                    cwd=record.cwd, metadata_json=dict(record.metadata), continuation_step_run_id=None if record.continuation is None else record.continuation.step_run_id,
                    closed_at=record.closed_at, created_at=record.created_at, updated_at=record.updated_at,
                ))
        if RuntimeDomain.EXECUTION in durable:
            table = metadata.tables["runtime_executions"]
            terminal = domain.execution.executions._terminal
            results = {} if terminal is None else terminal._results
            for record in domain.execution.executions._records.values():
                if record.tenant_id != tenant_id:
                    continue
                result = results.get((tenant_id, record.execution_id))
                reference = None if result is None else result.object_ref
                await connection.execute(insert(table).values(
                    namespace_key=namespace_digest, tenant_id=tenant_id, execution_id=record.execution_id,
                    session_id=record.session_id, binding_digest=record.binding_digest, parent_execution_id=record.parent_execution_id,
                    root_execution_id=record.root_execution_id, source_execution_id=record.source_execution_id, base_execution_id=record.base_execution_id,
                    lineage_kind=record.lineage_kind.value, status=record.status.value, revision=record.revision,
                    event_sequence=record.event_sequence, agent_run_sequence=record.agent_run_sequence, error_code=record.error_code,
                    safe_error_details_json=dict(record.safe_error_details), memory_scope=record.memory_scope,
                    conversation_step_run_id=record.conversation_step_run_id,
                    output_schema_id=None if result is None else result.output_schema_id,
                    output_schema_revision=None if result is None else result.output_schema_revision,
                    output_schema_fingerprint=None if result is None else result.output_schema_fingerprint,
                    result_store_id=None if reference is None else reference.store_id,
                    result_object_key=None if reference is None else reference.key,
                    result_digest=None if reference is None else reference.digest,
                    result_size=None if reference is None else reference.size,
                    stop_reason=None if result is None else result.stop_reason.value,
                    input_tokens=None if result is None else result.input_tokens,
                    output_tokens=None if result is None else result.output_tokens,
                    total_cost_micros=None if result is None else result.total_cost_micros,
                    result_created_at=None if result is None else result.created_at,
                    created_at=record.created_at, updated_at=record.updated_at,
                ))
            await _flush_execution_children(connection, metadata, domain, namespace_digest, tenant_id)
        if RuntimeDomain.MEMORY in durable and isinstance(domain.memory.records, MemoryRepositoryBackend):
            table = metadata.tables["runtime_memories"]
            for record in domain.memory.records._records.values():
                if record.tenant_id == tenant_id:
                    await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, memory_id=record.memory_id, memory_scope_key=record.memory_scope_key, object_store_id=record.content_ref.store_id, object_key=record.content_ref.key, object_digest=record.content_ref.digest, object_size=record.content_ref.size, metadata_json=dict(record.metadata), revision=record.revision, created_at=record.created_at, updated_at=record.updated_at))
        if RuntimeDomain.ARTIFACT in durable and isinstance(domain.artifact.records, ArtifactRepositoryBackend):
            table = metadata.tables["runtime_artifacts"]
            for record in domain.artifact.records._records.values():
                if record.tenant_id == tenant_id:
                    await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, artifact_id=record.artifact_id, execution_id=record.execution_id, producer=record.producer, media_type=record.media_type, object_store_id=record.object_ref.store_id, object_key=record.object_ref.key, object_digest=record.object_ref.digest, object_size=record.object_ref.size, created_at=record.created_at))
        if RuntimeDomain.TASK in durable and isinstance(domain.task.tasks, TaskRepositoryBackend):
            graph_table = metadata.tables["runtime_task_graphs"]
            node_table = metadata.tables["runtime_task_nodes"]
            for (record_tenant, graph_id), graph in domain.task.tasks._plans.items():
                if record_tenant != tenant_id:
                    continue
                await connection.execute(insert(graph_table).values(namespace_key=namespace_digest, tenant_id=tenant_id, graph_id=graph_id, status=graph.status.value, revision=0))
                for (node_tenant, node_graph_id, node_id), node in domain.task.tasks._nodes.items():
                    if node_tenant == tenant_id and node_graph_id == graph_id:
                        await connection.execute(insert(node_table).values(namespace_key=namespace_digest, tenant_id=tenant_id, graph_id=graph_id, node_id=node_id, dependencies_json=list(node.dependencies), status=node.status.value, revision=0, owner=node.owner, fence=node.fence, lease_expires_at=node.lease_expires_at, execution_id=node.execution_id, result_digest=node.result_digest, error_code=node.error_code, error_digest=node.error_digest))
        if RuntimeDomain.EVALUATION in durable and isinstance(domain.evaluation.records, EvaluationRepositoryBackend):
            table = metadata.tables["runtime_evaluations"]
            for record in domain.evaluation.records._records.values():
                if record.tenant_id == tenant_id:
                    await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, evaluation_id=record.evaluation_id, execution_id=record.execution_id, dataset_id=record.dataset_id, dataset_revision=record.dataset_revision, evaluator_id=record.evaluator_id, evaluator_revision=record.evaluator_revision, binding_digest=record.binding_digest, output_schema_fingerprint=record.output_schema_fingerprint, artifact_digest=record.artifact_digest, status=record.status.value, revision=record.revision, metrics_json=dict(record.metrics), created_at=record.created_at, updated_at=record.updated_at))
        if RuntimeDomain.RECOVERY in durable:
            if isinstance(domain.recovery.approvals, ApprovalRepositoryBackend):
                table = metadata.tables["runtime_approvals"]
                for record in domain.recovery.approvals._records.values():
                    if record.tenant_id == tenant_id:
                        await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, approval_id=record.approval_id, execution_id=record.execution_id, operation_id=record.operation_id, status=record.status.value, idempotency_key_hash=record.idempotency_key_hash, decision=None if record.decision is None else record.decision.value, decided_by=record.decided_by, decision_digest=record.decision_digest, created_at=record.created_at, decided_at=record.decided_at))
            if isinstance(domain.recovery.external_calls, ExternalRepositoryBackend):
                table = metadata.tables["runtime_external_calls"]
                for record in domain.recovery.external_calls._records.values():
                    if record.tenant_id == tenant_id:
                        reference = record.object_ref
                        await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, call_id=record.call_id, execution_id=record.execution_id, operation_id=record.operation_id, status=record.status.value, idempotency_key_hash=record.idempotency_key_hash, object_store_id=None if reference is None else reference.store_id, object_key=None if reference is None else reference.key, object_digest=None if reference is None else reference.digest, object_size=None if reference is None else reference.size, payload_digest=record.payload_digest, created_at=record.created_at, supplied_at=record.supplied_at))
            if isinstance(domain.recovery.checkpoints, RecoveryCheckpointRepositoryBackend):
                table = metadata.tables["runtime_recovery_checkpoints"]
                for record in domain.recovery.checkpoints._records.values():
                    if record.tenant_id == tenant_id:
                        payload = _record_payload(record)
                        await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, execution_id=record.execution_id, step_run_id=record.step_run_id, agent_run_sequence=record.agent_run_sequence, state=record.state.value, handoff_phase=record.handoff_phase.value, input_json=payload["input"], terminal_handoff_json=payload["terminal_handoff"], handoff_contract_digest=record.handoff_contract_digest, pending_operation_id=record.pending_operation_id, revision=record.revision, created_at=record.created_at, updated_at=record.updated_at))
            if isinstance(domain.recovery.tools, ToolRepositoryBackend):
                table = metadata.tables["runtime_tool_operations"]
                for record in domain.recovery.tools._records.values():
                    if record.tenant_id == tenant_id:
                        reference = record.result_object_ref
                        await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, tool_operation_id=record.tool_operation_id, step_run_id=record.run_id, tool_call_id=record.tool_call_id, idempotency_key_hash=record.idempotency_key_hash, tool_name=record.tool_name, arguments_hash=record.arguments_hash, binding_fingerprint=record.binding_fingerprint, replay_safe=record.replay_safe, status=record.status.value, owner=record.owner, fence=record.fence, lease_expires_at=record.lease_expires_at, result_store_id=None if reference is None else reference.store_id, result_object_key=None if reference is None else reference.key, result_digest=None if reference is None else reference.digest, result_size=None if reference is None else reference.size, error_code=record.error_code, created_at=record.created_at, updated_at=record.updated_at))
        if RuntimeDomain.EVALUATION in durable and isinstance(domain.evaluation.idempotency, IdempotencyRepositoryBackend):
            table = metadata.tables["runtime_idempotency"]
            for record_key, record in domain.evaluation.idempotency._records.items():
                if record.tenant_id == tenant_id:
                    await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, runtime_domain=RuntimeDomain.EVALUATION.value, scope=record.scope, key_hash=record_key[2], request_digest=record.request_digest, resource_kind=ResourceKind.EVALUATION.value, resource_id=record.execution_id, status=record.status.value, result_digest=record.result_digest, error_code=record.error_code, created_at=record.created_at, updated_at=record.updated_at))
        if "runtime_operations" in metadata.tables:
            table = metadata.tables["runtime_operations"]
            for runtime_domain in durable:
                repository = _operation_repository_for_domain(domain, runtime_domain)
                if not isinstance(repository, OperationRepositoryBackend):
                    continue
                for record in repository._records.values():
                    if record.tenant_id != tenant_id:
                        continue
                    await connection.execute(insert(table).values(
                        namespace_key=namespace_digest, tenant_id=tenant_id, runtime_domain=runtime_domain.value,
                        resource_kind=record.resource_kind.value, resource_id=record.resource_id,
                        operation_kind=record.kind.value, operation_id=record.operation_id, sequence=record.sequence,
                        status=record.status.value, execution_id=record.execution_id, request_digest=record.request_digest,
                        result_ref=record.result_ref, result_digest=record.result_digest, error_code=record.error_code,
                        compactable=record.compactable, created_at=record.created_at, updated_at=record.updated_at,
                    ))
        if "runtime_operation_counters" in metadata.tables:
            table = metadata.tables["runtime_operation_counters"]
            for runtime_domain in durable:
                repository = _operation_repository_for_domain(domain, runtime_domain)
                if not isinstance(repository, OperationRepositoryBackend):
                    continue
                for (record_tenant, resource_kind, resource_id), sequence in repository._counters.items():
                    if record_tenant == tenant_id:
                        await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, runtime_domain=runtime_domain.value, resource_kind=resource_kind, resource_id=resource_id, last_sequence=sequence, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
    _logger.info("SQL runtime state committed: namespace=%s tenant=%s durable=%s", namespace, tenant_id, sorted(item.value for item in durable))


async def _flush_execution_children(connection, metadata, domain: RuntimeStores, namespace_digest: str, tenant_id: str) -> None:
    from sqlalchemy import insert

    from ..core import ResourceKind

    if "runtime_events" in metadata.tables and isinstance(domain.execution.events, EventRepositoryBackend):
        table = metadata.tables["runtime_events"]
        for events in domain.execution.events._items.values():
            for event in events:
                if event.tenant_id == tenant_id:
                    await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, execution_id=event.execution_id, sequence=event.sequence, event_type=event.event_type.value, payload_json=event.payload, created_at=datetime.now(timezone.utc)))
    if "runtime_idempotency" in metadata.tables and isinstance(domain.execution.idempotency, IdempotencyRepositoryBackend):
        table = metadata.tables["runtime_idempotency"]
        for record in domain.execution.idempotency._records.values():
            if record.tenant_id == tenant_id:
                await connection.execute(insert(table).values(namespace_key=namespace_digest, tenant_id=tenant_id, runtime_domain=RuntimeDomain.EXECUTION.value, scope=record.scope, key_hash=record.key_hash, request_digest=record.request_digest, resource_kind=ResourceKind.EXECUTION.value, resource_id=record.execution_id, status=record.status.value, result_digest=record.result_digest, error_code=record.error_code, created_at=record.created_at, updated_at=record.updated_at))


def _operation_repository_for_domain(domain: RuntimeStores, runtime_domain: RuntimeDomain):
    return {
        RuntimeDomain.CONVERSATION: domain.conversation.operations,
        RuntimeDomain.EXECUTION: domain.execution.operations,
        RuntimeDomain.MEMORY: domain.memory.operations,
        RuntimeDomain.ARTIFACT: domain.artifact.operations,
        RuntimeDomain.TASK: domain.task.operations,
        RuntimeDomain.EVALUATION: domain.evaluation.operations,
        RuntimeDomain.RECOVERY: domain.recovery.operations,
    }[runtime_domain]


def domain_plan(durable: frozenset[RuntimeDomain]):
    from ..runtime import RuntimeStoragePlan, RuntimeStorageRoute
    return RuntimeStoragePlan({domain: RuntimeStorageRoute.durable() if domain in durable else RuntimeStorageRoute.volatile() for domain in RuntimeDomain})


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _object_ref_from_row(row: Mapping[str, object], prefix: str) -> ObjectRef:
    fields = tuple(row.get(f"{prefix}{name}") for name in ("store_id", "object_key", "digest", "size"))
    if any(value is None for value in fields):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ObjectRef(str(fields[0]), str(fields[1]), str(fields[2]), int(fields[3]))


async def _validate_object_reference(store: ObjectStore, reference: ObjectRef) -> None:
    if reference.store_id != store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    stat = await store.stat(reference.key)
    if stat is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if stat.digest != reference.digest or stat.size != reference.size:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _record_payload(record: object) -> dict[str, object]:
    return _json_value(asdict(record))


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
