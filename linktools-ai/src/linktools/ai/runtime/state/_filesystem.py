#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-domain generation-1 filesystem runtime backend."""

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from linktools.core import environ

from ...core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    StopReason,
    TaskStatus,
    ToolOperationStatus,
    UsageMetrics,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import (
    FilesystemObjectStore,
    FilesystemWriterLock,
    ObjectRef,
    ObjectStore,
    namespace_digest,
    read_json,
    write_json_atomic,
)
from ...task import TaskGraphView, TaskNode, TaskNodeView
from .._tool import ToolOperationRecord
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ConversationCursor,
    EvaluationRecord,
    ExecutionEventRecord,
    ExecutionRecord,
    ExternalCallRecord,
    IdempotencyRecord,
    MemoryRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    ResultRecord,
    RuntimeRepository,
    SessionRecord,
)
from ._memory import (
    RuntimeTransactionBinding,
    _ApprovalRepository,
    _ArtifactRepository,
    _build_in_memory_domains,
    _EvaluationRepository,
    _EventRepository,
    _ExecutionRepository,
    _ExternalRepository,
    _IdempotencyRepository,
    _MemoryRepository,
    _OperationRepository,
    _RecoveryCheckpointRepository,
    _restore_runtime_snapshot,
    _SessionRepository,
    _TaskRepository,
    _TerminalCommitRepository,
    _ToolRepository,
)
from ._plan import RuntimeDomain
from ._recovery_codec import (
    recovery_handoff_from_json,
    recovery_handoff_to_json,
    recovery_input_from_json,
    recovery_input_to_json,
)
from ._transaction import TransactionHub

_logger = environ.get_logger("ai.runtime.state.filesystem")


def _tenant_scope_key(tenant_id: str) -> str:
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()


def _filesystem_scope_root(
    route_root: Path,
    *,
    namespace: str,
    tenant_id: str,
) -> Path:
    return (
        route_root.expanduser().resolve(strict=False)
        / namespace_digest(namespace)
        / _tenant_scope_key(validate_tenant_id(tenant_id))
    )


async def _release_lock_after_prepare_failure(
    lock: FilesystemWriterLock,
    *,
    primary: BaseException,
) -> None:
    del primary
    task = asyncio.create_task(
        lock.release(),
        name="linktools-filesystem-prepare-lock-release",
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except BaseException:
            _logger.error(
                "filesystem writer lock release failed after prepare cancellation",
                exc_info=True,
            )
    except BaseException:
        _logger.error(
            "filesystem writer lock release failed after prepare failure",
            exc_info=True,
        )


class _FilesystemDomainBackend:
    """Own one durable Runtime domain and its generation-1 files."""

    def __init__(self, root: Path, *, namespace: str, tenant_id: str, domain: RuntimeDomain) -> None:
        self._route_root = root.expanduser().resolve(strict=False)
        self._root = _filesystem_scope_root(
            self._route_root,
            namespace=namespace,
            tenant_id=tenant_id,
        )
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._domain = domain
        self._manifest = self._root / "manifest.json"
        self._records = self._root / "records.json"
        self._object_store = FilesystemObjectStore(self._root / "objects")
        self._writer_lock = FilesystemWriterLock(self._root / "runtime.lock")
        self._hub = TransactionHub()
        self._binding = RuntimeTransactionBinding()
        parts = _build_in_memory_domains(
            namespace=namespace,
            domains=frozenset({domain}),
            transaction_hub=self._hub,
            transaction_binding=self._binding,
        )
        self._parts = parts
        self._state = parts.states[domain]
        self._components = parts.components
        self._commit_lock = threading.Lock()
        self._released = False
        self._binding.commit_callback = self._commit_domain
        self._binding.rollback_callback = self._rollback_domains
        self._hub.configure(
            snapshot=self._binding.snapshot,
            restore=self._binding.restore,
            commit=self._binding.commit,
            rollback=self._binding.rollback,
        )

    @property
    def state(self) -> object:
        return self._state

    @property
    def physical_root(self) -> Path:
        return self._root

    @property
    def object_store(self) -> ObjectStore:
        return self._object_store

    @property
    def writer_lock(self) -> FilesystemWriterLock:
        return self._writer_lock

    @property
    def components(self) -> tuple[RuntimeRepository, ...]:
        return self._components

    async def prepare(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AIError(
                ErrorCode.STORAGE_UNAVAILABLE,
                "failed to prepare filesystem runtime state",
            ) from error
        await self._writer_lock.acquire()
        try:
            await asyncio.to_thread(self._load)
            await self._validate_loaded_objects()
            if not self._manifest.is_file():
                await asyncio.to_thread(self._write_manifest)
            _logger.info("filesystem domain prepared: namespace=%s domain=%s", self._namespace, self._domain.value)
        except BaseException as primary:
            await _release_lock_after_prepare_failure(
                self._writer_lock,
                primary=primary,
            )
            raise

    async def release(self) -> None:
        if self._released:
            return
        await self._writer_lock.release()
        self._released = True
        _logger.debug("filesystem domain released: namespace=%s domain=%s", self._namespace, self._domain.value)

    def _load(self) -> None:
        try:
            if not self._manifest.is_file():
                if any(entry != self._writer_lock.path for entry in self._root.iterdir()):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._load_payload(_empty_payload())
                self._validate_payload()
                return
            manifest = read_json(self._manifest)
            if manifest.get("format") != "linktools-ai-runtime-state":
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            if manifest.get("generation") != 1:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            if (
                manifest.get("namespace") != self._namespace
                or manifest.get("tenant_id") != self._tenant_id
                or manifest.get("domain") != self._domain.value
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            records = read_json(self._records) if self._records.is_file() else _empty_payload()
            if not isinstance(records, dict):
                raise ValueError("runtime records must be an object")
            _validate_record_scope(records, self._tenant_id)
            _validate_state_record_uniqueness(_domain_validation_records(records))
            self._load_payload(records)
            self._validate_payload()
        except AIError:
            raise
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _load_payload(self, records: Mapping[str, JsonValue]) -> None:
        payload = _empty_payload()
        _merge_domain_payload(payload, dict(records))
        self._clear_repositories()
        sessions = self._repository(_SessionRepository)
        executions = self._repository(_ExecutionRepository)
        terminal = self._repository(_TerminalCommitRepository)
        idempotency = self._repository(_IdempotencyRepository)
        events = self._repository(_EventRepository)
        tasks = self._repository(_TaskRepository)
        evaluations = self._repository(_EvaluationRepository)
        memories = self._repository(_MemoryRepository)
        artifacts = self._repository(_ArtifactRepository)
        approvals = self._repository(_ApprovalRepository)
        external_calls = self._repository(_ExternalRepository)
        checkpoints = self._repository(_RecoveryCheckpointRepository)
        tools = self._repository(_ToolRepository)
        operations = self._repository(_OperationRepository)
        if isinstance(sessions, _SessionRepository):
            for raw in payload["sessions"]:
                if isinstance(raw, dict):
                    record = _session_from_json(raw)
                    sessions._records[(record.tenant_id, record.session_id)] = record
        if isinstance(executions, _ExecutionRepository):
            for raw in payload["executions"]:
                if isinstance(raw, dict):
                    record = _execution_from_json(raw)
                    executions._records[(record.tenant_id, record.execution_id)] = record
        if isinstance(idempotency, _IdempotencyRepository):
            for raw in payload["idempotency"]:
                if isinstance(raw, dict):
                    record = _idempotency_from_json(raw)
                    if _operation_storage_domain(record.resource_kind.value) is self._domain:
                        idempotency_key_digest = str(raw["idempotency_key_digest"])
                        idempotency._records[(record.tenant_id, record.scope, idempotency_key_digest)] = record
        if isinstance(terminal, _TerminalCommitRepository):
            for raw in payload["results"]:
                if isinstance(raw, dict):
                    record = _result_from_json(raw)
                    terminal._results[(record.tenant_id, record.execution_id)] = record
        if isinstance(events, _EventRepository):
            for raw in payload["events"]:
                if isinstance(raw, dict):
                    record = _event_from_json(raw)
                    events._items.setdefault((record.tenant_id, record.execution_id), []).append(record)
        if isinstance(tasks, _TaskRepository):
            for raw in payload["task_plans"]:
                if isinstance(raw, dict):
                    view = _task_plan_from_json(raw.get("view", raw))
                    tasks._plans[(self._tenant_id, view.graph_id)] = view
            for raw in payload["task_nodes"]:
                if isinstance(raw, dict):
                    node = _task_node_from_json(raw.get("node", raw))
                    tasks._nodes[(self._tenant_id, node.graph_id, node.node_id)] = node
        if isinstance(evaluations, _EvaluationRepository):
            for raw in payload["evaluations"]:
                if isinstance(raw, dict):
                    record = _evaluation_from_json(raw)
                    evaluations._records[(record.tenant_id, record.evaluation_id)] = record
        if isinstance(memories, _MemoryRepository):
            for raw in payload["memories"]:
                if isinstance(raw, dict):
                    record = _memory_from_json(raw)
                    memories._records[(record.tenant_id, record.memory_id)] = record
        if isinstance(artifacts, _ArtifactRepository):
            for raw in payload["artifacts"]:
                if isinstance(raw, dict):
                    record = _artifact_from_json(raw)
                    artifacts._records[(record.tenant_id, record.artifact_id)] = record
        if isinstance(approvals, _ApprovalRepository):
            for raw in payload["approvals"]:
                if isinstance(raw, dict):
                    record = _approval_from_json(raw)
                    approvals._records[(record.tenant_id, record.approval_id)] = record
        if isinstance(external_calls, _ExternalRepository):
            for raw in payload["external_calls"]:
                if isinstance(raw, dict):
                    record = _external_from_json(raw)
                    external_calls._records[(record.tenant_id, record.call_id)] = record
        if isinstance(checkpoints, _RecoveryCheckpointRepository):
            for raw in payload["recovery_checkpoints"]:
                if isinstance(raw, dict):
                    record = _recovery_checkpoint_from_json(raw)
                    checkpoints._records[(record.tenant_id, record.execution_id)] = record
        if isinstance(operations, _OperationRepository):
            for raw in payload["operations"]:
                if isinstance(raw, dict):
                    record = _operation_from_json(raw)
                    if _operation_storage_domain(record.resource_kind.value) is self._domain:
                        operations._records[(record.tenant_id, record.operation_id)] = record
                        key = (record.tenant_id, record.resource_kind.value, record.resource_id)
                        operations._counters[key] = max(operations._counters.get(key, 0), record.sequence)
        if isinstance(tools, _ToolRepository):
            for raw in payload["tools"]:
                if isinstance(raw, dict):
                    record = _tool_from_json(raw)
                    tools._records[(record.tenant_id, record.tool_operation_id)] = record

    def _clear_repositories(self) -> None:
        for component in self._components:
            if isinstance(component, (_SessionRepository, _ExecutionRepository, _IdempotencyRepository, _MemoryRepository, _ArtifactRepository, _ApprovalRepository, _ExternalRepository, _RecoveryCheckpointRepository, _OperationRepository, _EvaluationRepository, _ToolRepository)):
                component._records.clear()
            if isinstance(component, _OperationRepository):
                component._counters.clear()
            if isinstance(component, _EventRepository):
                component._items.clear()
            if isinstance(component, _TerminalCommitRepository):
                component._results.clear()
            if isinstance(component, _TaskRepository):
                component._plans.clear()
                component._nodes.clear()

    def _repository(self, kind: type[object]) -> object:
        return next((component for component in self._components if isinstance(component, kind)), None)

    def _write_manifest(self) -> None:
        write_json_atomic(
            self._manifest,
            {
                "format": "linktools-ai-runtime-state",
                "generation": 1,
                "namespace": self._namespace,
                "tenant_id": self._tenant_id,
                "domain": self._domain.value,
            },
            fsync=True,
        )

    def _flush_domain(self, domain: RuntimeDomain) -> None:
        if domain is not self._domain:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._records, self._domain_records(), fsync=True)
        _logger.debug("filesystem domain committed: namespace=%s domain=%s", self._namespace, domain.value)

    def _commit_domain(self, domain: RuntimeDomain) -> None:
        with self._commit_lock:
            if self._released:
                raise AIError(ErrorCode.STORAGE_CLOSED)
            try:
                self._flush_domain(domain)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED:
                    raise
                self._load()
                raise
            except BaseException as error:
                self._load()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    def _rollback_domains(self, domains: frozenset[RuntimeDomain]) -> None:
        if self._domain not in domains:
            return
        with self._commit_lock:
            snapshot = self._hub.snapshot_value
            if snapshot is not None:
                _restore_runtime_snapshot(snapshot)
            self._load()

    def _domain_records(self) -> dict[str, JsonValue]:
        values = _empty_payload()
        sessions = self._repository(_SessionRepository)
        executions = self._repository(_ExecutionRepository)
        terminal = self._repository(_TerminalCommitRepository)
        idempotency = self._repository(_IdempotencyRepository)
        events = self._repository(_EventRepository)
        tasks = self._repository(_TaskRepository)
        evaluations = self._repository(_EvaluationRepository)
        memories = self._repository(_MemoryRepository)
        artifacts = self._repository(_ArtifactRepository)
        approvals = self._repository(_ApprovalRepository)
        external_calls = self._repository(_ExternalRepository)
        checkpoints = self._repository(_RecoveryCheckpointRepository)
        tools = self._repository(_ToolRepository)
        operations = self._repository(_OperationRepository)
        if isinstance(operations, _OperationRepository):
            values["operations"] = [_json_record(item) for item in operations._records.values()]
        if self._domain is RuntimeDomain.CONVERSATION and isinstance(sessions, _SessionRepository):
            values["sessions"] = [_json_record(item) for item in sessions._records.values()]
        elif self._domain is RuntimeDomain.EXECUTION:
            if isinstance(executions, _ExecutionRepository):
                values["executions"] = [_json_record(item) for item in executions._records.values()]
            if isinstance(terminal, _TerminalCommitRepository):
                values["results"] = [_json_record(item) for item in terminal._results.values()]
            if isinstance(events, _EventRepository):
                values["events"] = [_json_record(item) for items in events._items.values() for item in items]
            if isinstance(idempotency, _IdempotencyRepository):
                values["idempotency"] = [
                    _idempotency_json(item, idempotency_key_digest)
                    for (tenant, scope, idempotency_key_digest), item in idempotency._records.items()
                    if scope != "evaluation.run"
                ]
        elif self._domain is RuntimeDomain.MEMORY and isinstance(memories, _MemoryRepository):
            values["memories"] = [_json_record(item) for item in memories._records.values()]
        elif self._domain is RuntimeDomain.ARTIFACT and isinstance(artifacts, _ArtifactRepository):
            values["artifacts"] = [_json_record(item) for item in artifacts._records.values()]
        elif self._domain is RuntimeDomain.TASK and isinstance(tasks, _TaskRepository):
            values["tasks"] = [
                *(
                    {"record_type": "plan", "tenant_id": tenant, "view": _task_view_json(view)}
                    for (tenant, _), view in tasks._plans.items()
                ),
                *({"record_type": "node", "tenant_id": tenant, "node": _json_record(node)} for (tenant, _, _), node in tasks._nodes.items()),
            ]
        elif self._domain is RuntimeDomain.EVALUATION:
            if isinstance(evaluations, _EvaluationRepository):
                values["evaluations"] = [_json_record(item) for item in evaluations._records.values()]
            if isinstance(idempotency, _IdempotencyRepository):
                values["idempotency"] = [
                    _idempotency_json(item, idempotency_key_digest)
                    for (tenant, scope, idempotency_key_digest), item in idempotency._records.items()
                    if scope == "evaluation.run"
                ]
        elif self._domain is RuntimeDomain.RECOVERY:
            if isinstance(approvals, _ApprovalRepository):
                values["approvals"] = [_json_record(item) for item in approvals._records.values()]
            if isinstance(external_calls, _ExternalRepository):
                values["external_calls"] = [_json_record(item) for item in external_calls._records.values()]
            if isinstance(checkpoints, _RecoveryCheckpointRepository):
                values["recovery_checkpoints"] = [_json_record(item) for item in checkpoints._records.values()]
            if isinstance(tools, _ToolRepository):
                values["tools"] = [_json_record(item) for item in tools._records.values()]
        return values

    def _validate_payload(self) -> None:
        executions = self._repository(_ExecutionRepository)
        terminal = self._repository(_TerminalCommitRepository)
        if isinstance(executions, _ExecutionRepository) and isinstance(terminal, _TerminalCommitRepository):
            for key, result in terminal._results.items():
                execution = executions._records.get(key)
                if execution is None or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                _validate_terminal_result(execution, result)
            for key, execution in executions._records.items():
                if execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and key not in terminal._results:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = self._repository(_EventRepository)
        if isinstance(events, _EventRepository):
            for key, records in events._items.items():
                if [item.sequence for item in records] != list(range(1, len(records) + 1)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution = executions._records.get(key) if isinstance(executions, _ExecutionRepository) else None
                if execution is None or execution.event_sequence != len(records):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _validate_loaded_objects(self) -> None:
        references: list[ObjectRef] = []
        terminal = self._repository(_TerminalCommitRepository)
        if self._domain is RuntimeDomain.EXECUTION and isinstance(terminal, _TerminalCommitRepository):
            references.extend(result.object_ref for result in terminal._results.values() if result.object_ref is not None)
        memories = self._repository(_MemoryRepository)
        if self._domain is RuntimeDomain.MEMORY and isinstance(memories, _MemoryRepository):
            references.extend(record.content_ref for record in memories._records.values())
        artifacts = self._repository(_ArtifactRepository)
        if self._domain is RuntimeDomain.ARTIFACT and isinstance(artifacts, _ArtifactRepository):
            references.extend(record.object_ref for record in artifacts._records.values())
        external_calls = self._repository(_ExternalRepository)
        if self._domain is RuntimeDomain.RECOVERY and isinstance(external_calls, _ExternalRepository):
            references.extend(record.object_ref for record in external_calls._records.values() if record.object_ref is not None)
        tools = self._repository(_ToolRepository)
        if self._domain is RuntimeDomain.RECOVERY and isinstance(tools, _ToolRepository):
            references.extend(record.result_object_ref for record in tools._records.values() if record.result_object_ref is not None)
        checkpoints = self._repository(_RecoveryCheckpointRepository)
        if self._domain is RuntimeDomain.RECOVERY and isinstance(checkpoints, _RecoveryCheckpointRepository):
            references.extend(
                checkpoint.terminal_handoff.outcome.recovery_object_ref
                for checkpoint in checkpoints._records.values()
                if checkpoint.terminal_handoff is not None and checkpoint.terminal_handoff.outcome.recovery_object_ref is not None
            )
        for reference in references:
            await _validate_object_reference(self._object_store, reference)


async def _build_filesystem_domain(root: Path, *, namespace: str, tenant_id: str, domain: RuntimeDomain) -> _FilesystemDomainBackend:
    return _FilesystemDomainBackend(root, namespace=namespace, tenant_id=tenant_id, domain=domain)


def _validate_state_record_uniqueness(records: "dict[str, JsonValue]") -> None:
    keys = {
        "sessions": ("tenant_id", "session_id"),
        "executions": ("tenant_id", "execution_id"),
        "results": ("tenant_id", "execution_id"),
        "idempotency": ("tenant_id", "scope", "idempotency_key_digest"),
        "events": ("tenant_id", "execution_id", "sequence"),
        "evaluations": ("tenant_id", "evaluation_id"),
        "memories": ("tenant_id", "memory_id"),
        "artifacts": ("tenant_id", "artifact_id"),
        "approvals": ("tenant_id", "approval_id"),
        "external_calls": ("tenant_id", "call_id"),
        "operations": ("tenant_id", "operation_id"),
        "tools": ("tenant_id", "tool_operation_id"),
    }
    for name, fields in keys.items():
        seen: set[tuple[str, ...]] = set()
        for item in records[name]:
            if not isinstance(item, dict):
                raise ValueError(f"runtime {name} record must be an object")
            try:
                identity = tuple(str(item[field]) for field in fields)
            except KeyError as error:
                raise ValueError(f"runtime {name} identity is incomplete") from error
            if identity in seen:
                raise ValueError(f"runtime {name} identity is duplicated")
            seen.add(identity)
    for item in records["idempotency"]:
        if re.fullmatch(r"[0-9a-f]{64}", str(item["idempotency_key_digest"])) is None:
            raise ValueError("runtime idempotency key hash is invalid")
    task_keys: set[tuple[str, str, str]] = set()
    for item in records["tasks"]:
        if not isinstance(item, dict) or item.get("record_type") not in {"plan", "node"} or not isinstance(item.get("tenant_id"), str):
            raise ValueError("runtime task identity is invalid")
        payload = item.get("view") if item["record_type"] == "plan" else item.get("node")
        if not isinstance(payload, dict):
            raise ValueError("runtime task payload is invalid")
        identity = (str(item["tenant_id"]), str(payload.get("graph_id")), str(payload.get("node_id", "")))
        if identity in task_keys:
            raise ValueError("runtime task identity is duplicated")
        task_keys.add(identity)


def _validate_terminal_result(execution: ExecutionRecord, result: ResultRecord) -> None:
    schema = (result.output_schema_id, result.output_schema_revision, result.output_schema_fingerprint)
    has_schema = all(value is not None for value in schema)
    if execution.status is ExecutionStatus.SUCCEEDED and (not has_schema or result.object_ref is None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if execution.status is not ExecutionStatus.SUCCEEDED and (has_schema or result.object_ref is not None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _domain_validation_records(records: dict[str, JsonValue]) -> dict[str, JsonValue]:
    empty = {"sessions": [], "executions": [], "results": [], "idempotency": [], "events": [], "tasks": [], "evaluations": [], "memories": [], "artifacts": [], "approvals": [], "external_calls": [], "operations": [], "tools": []}
    for key in empty:
        if key in records:
            empty[key] = records[key]
    return empty


def _validate_record_scope(records: dict[str, JsonValue], tenant_id: str) -> None:
    allowed = frozenset((*_empty_payload(), "tasks"))
    unknown = set(records) - allowed
    if unknown:
        raise ValueError(f"runtime records contain unknown fields: {sorted(unknown)}")
    for value in records.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and "tenant_id" in item and item["tenant_id"] != tenant_id:
                raise ValueError("runtime record tenant scope is invalid")


def _empty_payload() -> dict[str, JsonValue]:
    return {
        "sessions": [], "executions": [], "results": [], "idempotency": [], "events": [],
        "task_plans": [], "task_nodes": [], "evaluations": [], "memories": [], "artifacts": [],
        "approvals": [], "external_calls": [], "recovery_checkpoints": [], "operations": [], "tools": [],
    }


def _domain_payload(payload: dict[str, JsonValue], domain: RuntimeDomain) -> dict[str, JsonValue]:
    keys: dict[RuntimeDomain, tuple[str, ...]] = {
        RuntimeDomain.CONVERSATION: ("sessions", "operations"),
        RuntimeDomain.EXECUTION: ("executions", "results", "idempotency", "events", "operations"),
        RuntimeDomain.MEMORY: ("memories", "operations"),
        RuntimeDomain.ARTIFACT: ("artifacts", "operations"),
        RuntimeDomain.TASK: ("task_plans", "task_nodes", "operations"),
        RuntimeDomain.EVALUATION: ("evaluations", "idempotency", "operations"),
        RuntimeDomain.RECOVERY: ("approvals", "external_calls", "recovery_checkpoints", "tools", "operations"),
    }
    values = {key: payload[key] for key in keys[domain] if key != "operations"}
    if domain is RuntimeDomain.EXECUTION:
        values["idempotency"] = [item for item in values["idempotency"] if isinstance(item, dict) and item.get("scope") != "evaluation.run"]
    elif domain is RuntimeDomain.EVALUATION:
        values["idempotency"] = [item for item in values["idempotency"] if isinstance(item, dict) and item.get("scope") == "evaluation.run"]
    if "operations" in keys[domain]:
        values["operations"] = [
            item for item in payload["operations"]
            if isinstance(item, dict) and _operation_storage_domain(item.get("resource_kind")) is domain
        ]
    values["tasks"] = [
        {"record_type": "plan", **item}
        for item in values.pop("task_plans", [])
        if isinstance(item, dict)
    ] + [
        {"record_type": "node", **item}
        for item in values.pop("task_nodes", [])
        if isinstance(item, dict)
    ]
    return values


def _operation_storage_domain(value: object) -> RuntimeDomain:
    return {
        "SESSION": RuntimeDomain.CONVERSATION,
        "EXECUTION": RuntimeDomain.EXECUTION,
        "MEMORY": RuntimeDomain.MEMORY,
        "ARTIFACT": RuntimeDomain.ARTIFACT,
        "TASK_GRAPH": RuntimeDomain.TASK,
        "EVALUATION": RuntimeDomain.EVALUATION,
        "DOWNLOAD_GRANT": RuntimeDomain.ARTIFACT,
        "APPROVAL": RuntimeDomain.RECOVERY,
        "EXTERNAL_CALL": RuntimeDomain.RECOVERY,
        "TOOL_OPERATION": RuntimeDomain.RECOVERY,
    }.get(str(value), RuntimeDomain.RECOVERY)


def _merge_domain_payload(payload: dict[str, JsonValue], records: dict[str, JsonValue]) -> None:
    for key, value in records.items():
        if key == "tasks":
            if not isinstance(value, list):
                raise ValueError("runtime task records must be a list")
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("runtime task record must be an object")
                target = "task_plans" if item.get("record_type") == "plan" else "task_nodes" if item.get("record_type") == "node" else None
                if target is None:
                    raise ValueError("runtime task record type is invalid")
                payload[target].append({key: value for key, value in item.items() if key != "record_type"})
        elif key in payload:
            if not isinstance(value, list):
                raise ValueError("runtime domain records must be lists")
            payload[key].extend(value)
async def _validate_object_reference(store: ObjectStore, reference: ObjectRef) -> None:
    if reference.store_id != store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    stat = await store.stat(reference.key)
    if stat is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if stat.digest != reference.digest or stat.size != reference.size:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _record_json(record: SessionRecord | ExecutionRecord | IdempotencyRecord) -> dict[str, JsonValue]:
    value = asdict(record)
    return _json_value(value)


def _json_value(value: "JsonValue | datetime | Enum") -> JsonValue:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported persisted value: {type(value).__name__}")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _session_from_json(value: dict[str, JsonValue]) -> SessionRecord:
    raw_cursor = value.get("continuation")
    continuation = None if raw_cursor is None else ConversationCursor(str(raw_cursor["step_run_id"]))
    return SessionRecord(session_id=str(value["session_id"]), tenant_id=str(value["tenant_id"]), owner_principal_id=str(value["owner_principal_id"]), binding_digest=str(value["binding_digest"]), status=SessionStatus(str(value["status"])), revision=int(value["revision"]), resource_generation=int(value["resource_generation"]), cwd=None if value.get("cwd") is None else str(value["cwd"]), metadata=value.get("metadata", {}), created_at=_time(value["created_at"]), updated_at=_time(value["updated_at"]), closed_at=None if value.get("closed_at") is None else _time(value["closed_at"]), continuation=continuation)


def _execution_from_json(value: dict[str, JsonValue]) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=str(value["execution_id"]),
        tenant_id=str(value["tenant_id"]),
        session_id=(
            None
            if value.get("session_id") is None
            else str(value["session_id"])
        ),
        binding_digest=str(value["binding_digest"]),
        parent_execution_id=(
            None
            if value.get("parent_execution_id") is None
            else str(value["parent_execution_id"])
        ),
        root_execution_id=str(value["root_execution_id"]),
        source_execution_id=(
            None
            if value.get("source_execution_id") is None
            else str(value["source_execution_id"])
        ),
        base_execution_id=(
            None
            if value.get("base_execution_id") is None
            else str(value["base_execution_id"])
        ),
        lineage_kind=ExecutionLineageKind(str(value.get("lineage_kind", "RUN"))),
        status=ExecutionStatus(str(value["status"])),
        revision=int(value["revision"]),
        event_sequence=int(value["event_sequence"]),
        agent_run_sequence=int(value.get("agent_run_sequence", 0)),
        error_code=None if value.get("error_code") is None else str(value["error_code"]),
        safe_error_details=value.get("safe_error_details", {}),
        created_at=_time(value["created_at"]),
        updated_at=_time(value["updated_at"]),
        memory_scope=(
            None
            if value.get("memory_scope") is None
            else str(value["memory_scope"])
        ),
        conversation_step_run_id=(
            None
            if value.get("conversation_step_run_id") is None
            else str(value["conversation_step_run_id"])
        ),
    )


def _idempotency_from_json(value: dict[str, JsonValue]) -> IdempotencyRecord:
    idempotency_key_digest = str(value["idempotency_key_digest"])
    if re.fullmatch(r"[0-9a-f]{64}", idempotency_key_digest) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return IdempotencyRecord(
        tenant_id=str(value["tenant_id"]),
        runtime_domain=RuntimeDomain(str(value["runtime_domain"])),
        scope=str(value["scope"]),
        idempotency_key_digest=idempotency_key_digest,
        request_digest=str(value["request_digest"]),
        resource_kind=ResourceKind(str(value["resource_kind"])),
        resource_id=str(value["resource_id"]),
        status=IdempotencyStatus(str(value["status"])),
        result_digest=None if value.get("result_digest") is None else str(value["result_digest"]),
        error_code=None if value.get("error_code") is None else str(value["error_code"]),
        created_at=_time(value["created_at"]),
        updated_at=_time(value["updated_at"]),
    )


def _recovery_checkpoint_from_json(value: dict[str, JsonValue]) -> RecoveryCheckpoint:
    try:
        return RecoveryCheckpoint(
            execution_id=str(value["execution_id"]),
            tenant_id=str(value["tenant_id"]),
            input=recovery_input_from_json(value["input"]),
            step_run_id=None if value.get("step_run_id") is None else str(value["step_run_id"]),
            agent_run_sequence=int(value["agent_run_sequence"]),
            state=RecoveryCheckpointState(str(value["state"])),
            handoff_phase=RecoveryHandoffPhase(str(value["handoff_phase"])),
            terminal_handoff=recovery_handoff_from_json(value.get("terminal_handoff")),
            handoff_contract_digest=(
                None
                if value.get("handoff_contract_digest") is None
                else str(value["handoff_contract_digest"])
            ),
            pending_operation_id=(
                None
                if value.get("pending_operation_id") is None
                else str(value["pending_operation_id"])
            ),
            revision=int(value["revision"]),
            created_at=_time(value["created_at"]),
            updated_at=_time(value["updated_at"]),
        )
    except AIError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _json_record(record: object) -> dict[str, JsonValue]:
    if isinstance(record, RecoveryCheckpoint):
        value = _json_value(asdict(record))
        value["input"] = recovery_input_to_json(record.input)
        value["terminal_handoff"] = recovery_handoff_to_json(record.terminal_handoff)
        return value
    return _json_value(asdict(record))


def _idempotency_json(record: IdempotencyRecord, idempotency_key_digest: str) -> dict[str, JsonValue]:
    value = _record_json(record)
    value["idempotency_key_digest"] = idempotency_key_digest
    return value


def _result_from_json(value: dict[str, JsonValue]) -> ResultRecord:
    raw_ref = value.get("object_ref")
    object_ref = None
    if raw_ref is not None:
        if not isinstance(raw_ref, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object_ref = ObjectRef(str(raw_ref["store_id"]), str(raw_ref["key"]), str(raw_ref["digest"]), int(raw_ref["size"]))
    return ResultRecord(
        str(value["execution_id"]),
        str(value["tenant_id"]),
        None if value.get("output_schema_id") is None else str(value["output_schema_id"]),
        None if value.get("output_schema_revision") is None else int(value["output_schema_revision"]),
        None if value.get("output_schema_fingerprint") is None else str(value["output_schema_fingerprint"]),
        object_ref,
        StopReason(str(value["stop_reason"])),
        _usage_from_json(value["usage"]),
        _time(str(value["created_at"])),
    )


def _usage_from_json(value: object) -> UsageMetrics:
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return UsageMetrics(
            model_requests=int(value["model_requests"]),
            tool_calls=int(value["tool_calls"]),
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            cache_read_tokens=int(value["cache_read_tokens"]),
            cache_write_tokens=int(value["cache_write_tokens"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _event_from_json(value: dict[str, JsonValue]) -> ExecutionEventRecord:
    return ExecutionEventRecord(str(value["execution_id"]), str(value["tenant_id"]), int(value["sequence"]), ExecutionEventType(str(value["event_type"])), value.get("payload", {}))


def _task_plan_from_json(value: dict[str, JsonValue]) -> TaskGraphView:
    nodes = tuple(
        TaskNode(
            str(item["node_id"]),
            tuple(item.get("dependencies", [])),
            input=item.get("input", {}),
            budget_cost=int(item.get("budget_cost", 1)),
        )
        for item in value.get("nodes", [])
    )
    return TaskGraphView(str(value["graph_id"]), TaskStatus(str(value["status"])), nodes)


def _task_view_json(value: TaskGraphView) -> dict[str, JsonValue]:
    return {
        "graph_id": value.graph_id,
        "status": value.status.value,
        "nodes": [
            {
                "node_id": node.node_id,
                "dependencies": list(node.dependencies),
                "input": node.input,
                "budget_cost": node.budget_cost,
            }
            for node in value.nodes
        ],
    }


def _task_node_from_json(value: dict[str, JsonValue]) -> TaskNodeView:
    return TaskNodeView(str(value["graph_id"]), str(value["node_id"]), tuple(value.get("dependencies", [])), TaskStatus(str(value["status"])), None if value.get("owner") is None else str(value["owner"]), int(value["fence"]), None if value.get("lease_expires_at") is None else _time(str(value["lease_expires_at"])), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), None if value.get("error_digest") is None else str(value["error_digest"]), None if value.get("execution_id") is None else str(value["execution_id"]))


def _evaluation_from_json(value: dict[str, JsonValue]) -> EvaluationRecord:
    return EvaluationRecord(str(value["evaluation_id"]), str(value["tenant_id"]), str(value["execution_id"]), str(value["dataset_id"]), int(value["dataset_revision"]), str(value["evaluator_id"]), int(value["evaluator_revision"]), str(value["binding_digest"]), str(value["output_schema_fingerprint"]), None if value.get("artifact_digest") is None else str(value["artifact_digest"]), EvaluationStatus(str(value["status"])), int(value["revision"]), value.get("metrics", {}), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _memory_from_json(value: dict[str, JsonValue]) -> MemoryRecord:
    raw_ref = value["content_ref"]
    if not isinstance(raw_ref, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return MemoryRecord(
        str(value["memory_id"]),
        str(value["tenant_id"]),
        str(value["memory_scope_digest"]),
        ObjectRef(
            str(raw_ref["store_id"]),
            str(raw_ref["key"]),
            str(raw_ref["digest"]),
            int(raw_ref["size"]),
        ),
        str(value["content_digest"]),
        value.get("metadata", {}),
        int(value["revision"]),
        _time(str(value["created_at"])),
        _time(str(value["updated_at"])),
    )


def _artifact_from_json(value: dict[str, JsonValue]) -> ArtifactRecord:
    raw_ref = value["object_ref"]
    if not isinstance(raw_ref, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ArtifactRecord(str(value["artifact_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["producer"]), str(value["media_type"]), int(value["size"]), str(value["digest"]), ObjectRef(str(raw_ref["store_id"]), str(raw_ref["key"]), str(raw_ref["digest"]), int(raw_ref["size"])), _time(str(value["created_at"])))


def _approval_from_json(value: dict[str, JsonValue]) -> ApprovalRecord:
    return ApprovalRecord(str(value["approval_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), ApprovalStatus(str(value["status"])), None if value.get("idempotency_key_digest") is None else str(value["idempotency_key_digest"]), None if value.get("decision") is None else ApprovalDecision(str(value["decision"])), None if value.get("decided_by") is None else str(value["decided_by"]), None if value.get("decision_digest") is None else str(value["decision_digest"]), _time(str(value["created_at"])), None if value.get("decided_at") is None else _time(str(value["decided_at"])))


def _external_from_json(value: dict[str, JsonValue]) -> ExternalCallRecord:
    return ExternalCallRecord(str(value["call_id"]), str(value["execution_id"]), str(value["tenant_id"]), str(value["operation_id"]), ExternalCallStatus(str(value["status"])), None if value.get("idempotency_key_digest") is None else str(value["idempotency_key_digest"]), None, None if value.get("payload_digest") is None else str(value["payload_digest"]), _time(str(value["created_at"])), None if value.get("supplied_at") is None else _time(str(value["supplied_at"])))


def _operation_from_json(value: dict[str, JsonValue]) -> OperationLedgerRecord:
    return OperationLedgerRecord(str(value["operation_id"]), str(value["tenant_id"]), ResourceKind(str(value["resource_kind"])), str(value["resource_id"]), None if value.get("execution_id") is None else str(value["execution_id"]), OperationKind(str(value["operation_kind"])), OperationStatus(str(value["status"])), str(value["request_digest"]), None if value.get("result_ref") is None else str(value["result_ref"]), None if value.get("result_digest") is None else str(value["result_digest"]), None if value.get("error_code") is None else str(value["error_code"]), bool(value["compactable"]), int(value["sequence"]), _time(str(value["created_at"])), _time(str(value["updated_at"])))


def _tool_from_json(value: dict[str, JsonValue]) -> ToolOperationRecord:
    raw_ref = value.get("result_object_ref")
    object_ref = None
    if raw_ref is not None:
        if not isinstance(raw_ref, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object_ref = ObjectRef(str(raw_ref["store_id"]), str(raw_ref["key"]), str(raw_ref["digest"]), int(raw_ref["size"]))
    return ToolOperationRecord(
        tool_operation_id=str(value["tool_operation_id"]), tenant_id=str(value["tenant_id"]), step_run_id=str(value["step_run_id"]),
        tool_call_id=str(value["tool_call_id"]), idempotency_key_digest=str(value["idempotency_key_digest"]), tool_name=str(value["tool_name"]),
        arguments_digest=str(value["arguments_digest"]), binding_fingerprint=str(value["binding_fingerprint"]), replay_safe=bool(value["replay_safe"]),
        status=ToolOperationStatus(str(value["status"])), owner=None if value.get("owner") is None else str(value["owner"]), fence=int(value["fence"]),
        lease_expires_at=None if value.get("lease_expires_at") is None else _time(value["lease_expires_at"]), result_object_ref=object_ref,
        error_code=None if value.get("error_code") is None else str(value["error_code"]), created_at=_time(str(value["created_at"])), updated_at=_time(str(value["updated_at"])),
    )


__all__ = []
