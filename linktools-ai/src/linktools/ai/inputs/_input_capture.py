#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized, immutable input capture shared by execution producers."""

from ..storage import StorageRetentionMode

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, TypeAlias

from ..agent import AgentBindingContract, AgentInputCaptureRef
from ..asset import AssetStoreReader, AssetVersionRef
from ..core import (
    AuthorizationAction, AuthorizationPolicy, ImmutableJsonMapping, JsonValue,
    Principal, ResourceKind, ResourceRef, TaskStatus, WorkspaceFileInput, canonical_json_bytes,
    canonical_sha256, deterministic_id, validate_idempotency_key, principal_identity_payload, validate_page_limit,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, StoredPayload
from ..task import (
    TaskDependencyCapture, TaskDependencyState, TaskGraph,
    TaskGraphAdmission, TaskGraphCaptureRef, TaskGraphTemplate, TaskGraphTemplateRef,
    TaskInvocationInputContract, TaskInvocationInputRef, TaskNode,
    TaskNodeInvocation, TaskNodeResultRef, TaskRef, TaskResultRef,
)
from ..task import AgentTaskNode
from ._input import (
    CanonicalUserInput, ExecutionInputMaterializer, UserPromptInput,
    captured_input_files, captured_input_prompt, decode_task_prompt_draft, task_prompt_draft,
)
from ..execution import ExecutionInputContext, RuntimePayloadRef
from ._context_archive import ExecutionContextArchive
from ..model import MaterializedUserContent
from ..model.attachments import read_attachment_payload
from ..persistence.objects import read_runtime_object, put_runtime_object
from ._task_graph_binding_capture import TaskGraphBindingCaptureStore
from ..execution import ExecutionResult, ExecutionView
from ..persistence import Persistence
from ..persistence.objects import ObjectKeyFactory
from ..core import PromptLimits
from ..persistence import PersistenceDomain
from ..persistence.contracts import ExecutionRepositories, TaskRepositories
from ..persistence.captures import (
    input_capture_key, input_capture_expiry_key, capture_record, capture_record_key, capture_record_value, capture_retention_value,
    read_capture_record, publish_capture_record, prepare_capture_value, restore_capture_value,
)
from ..persistence.codec import decode_domain, encode_domain, iter_runtime_object_refs
from ..persistence.state import RecordQuery, StateTransaction, scope_key
from ..execution import StoredUserInput

ExecutionInputCaptureRef: TypeAlias = AgentInputCaptureRef | TaskInvocationInputRef


@dataclass(frozen=True, slots=True)
class CaptureInputRequest:
    principal: Principal
    idempotency_key: str
    context_policy: str = "captured"

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if self.context_policy not in {"captured", "clean"}:
            raise ValueError("context policy is invalid")


@dataclass(frozen=True, slots=True)
class CaptureGraphRequest:
    principal: Principal
    idempotency_key: str
    mode: str = "declaration_graph"
    context_policy: str = "captured"

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if self.mode not in {"declaration_graph", "materialized_graph"}:
            raise ValueError("graph capture mode is invalid")
        if self.context_policy not in {"captured", "clean"}:
            raise ValueError("context policy is invalid")


@dataclass(frozen=True, slots=True)
class AgentInputCapture:
    prompt: CanonicalUserInput
    binding: AgentBindingContract | None
    original_input: Mapping[str, JsonValue]
    source_execution_id: str | None
    source_invocation_id: str | None = None
    repository_instructions: JsonValue = None
    task_input: TaskInvocationInputContract | None = None
    source_principal: Mapping[str, JsonValue] | None = None
    input_context: ExecutionInputContext | None = None
    tool_response_ref: AssetVersionRef | None = None

    def require_importable(self) -> None:
        """Reject imports that cannot retain all accepted invocation inputs."""
        if self.tool_response_ref is not None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE,
                          safe_details={"reason": "tool_response_fixture_import_unsupported"})


class InputCaptureExecutionPort(Protocol):
    """Authorized execution reads and retention holds required while capturing input."""

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...

    async def acquire_dependency_hold(
        self, execution_id: str, *, tenant_id: str, hold_id: str,
    ) -> bool: ...

    async def release_dependency_hold(
        self, execution_id: str, *, tenant_id: str, hold_id: str,
    ) -> None: ...


class InputCaptures:
    def __init__(
        self,
        namespace: str,
        authorization: AuthorizationPolicy,
        execution: InputCaptureExecutionPort,
        materializer: ExecutionInputMaterializer,
        *,
        tenant_id: str,
        execution_repositories: ExecutionRepositories,
        task_repositories: TaskRepositories,
        object_stores: Mapping[PersistenceDomain, ObjectStore],
        context_archive: ExecutionContextArchive,
        execution_retention: StorageRetentionMode,
        task_retention: StorageRetentionMode,
        prepare_pending_capture: Callable[[str, Principal], Awaitable[None]] | None = None,
    ) -> None:
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._execution_repositories = execution_repositories
        self._task_repositories = task_repositories
        self._object_stores = object_stores
        self._context_archive = context_archive
        self._execution_retention = execution_retention
        self._task_retention = task_retention
        self._authorization = authorization
        self._execution = execution
        self._materializer = materializer
        self._prepare_pending_capture = prepare_pending_capture
        self._state_store = task_repositories.tasks.state_store

    @classmethod
    def from_persistence(
        cls,
        persistence: Persistence,
        *,
        authorization: AuthorizationPolicy,
        execution: InputCaptureExecutionPort,
        materializer: ExecutionInputMaterializer | None = None,
        asset_sources: Mapping[str, AssetStoreReader] | None = None,
        prepare_pending_capture: Callable[[str, Principal], Awaitable[None]] | None = None,
    ) -> "InputCaptures":
        """Compose captures over borrowed storage and the producer's execution boundary."""
        if materializer is None:
            materializer = ExecutionInputMaterializer(
                None, PromptLimits(),
                object_store=persistence.object_store(PersistenceDomain.EXECUTION),
                object_key_factory=ObjectKeyFactory(persistence.namespace),
            )
        return cls(
            persistence.namespace, authorization, execution, materializer,
            tenant_id=persistence.tenant_id,
            execution_repositories=persistence.execution,
            task_repositories=persistence.task,
            object_stores={domain: persistence.object_store(domain) for domain in PersistenceDomain},
            context_archive=ExecutionContextArchive(
                persistence.execution.executions.state_store,
                persistence.archive_object_store(PersistenceDomain.EXECUTION),
                namespace=persistence.namespace, tenant_id=persistence.tenant_id,
                asset_sources=asset_sources,
                object_reader=lambda reference: persistence.read_archive_object(PersistenceDomain.EXECUTION, reference),
            ),
            execution_retention=persistence.plan.route(PersistenceDomain.EXECUTION).retention,
            task_retention=persistence.plan.route(PersistenceDomain.TASK).retention,
            prepare_pending_capture=prepare_pending_capture,
        )

    def _key(self, tenant_id: str, kind: str, identity: str) -> str:
        return input_capture_key(self._namespace, tenant_id, kind, identity)

    async def _put(self, key: str, value: JsonValue) -> str:
        digest = canonical_sha256(value)
        stored = await prepare_capture_value(key, value, write_object=lambda raw: put_runtime_object(
            self._object_stores[PersistenceDomain.TASK], ObjectKeyFactory(self._namespace),
            PersistenceDomain.TASK, self._tenant_id, raw))
        await publish_capture_record(self._state_store, namespace=self._namespace,
            tenant_id=self._tenant_id, key=key, value=stored,
            source_digest=digest if stored != value else None)
        return digest

    async def _get(self, key: str, digest: str | None = None) -> JsonValue | None:
        record = await read_capture_record(self._state_store, namespace=self._namespace,
                                           tenant_id=self._tenant_id, key=key)
        if record is None:
            return None
        value = await restore_capture_value(record, read_object=lambda ref: read_runtime_object(
            self._object_stores[PersistenceDomain.TASK], ref))
        if digest is not None and canonical_sha256(value) != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def _authorize(self, principal: Principal, kind: ResourceKind, id: str, action: AuthorizationAction,
                         owner_principal_id: str | None = None) -> None:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, action, ResourceRef(kind, id, principal.tenant_id, owner_principal_id))

    def _publication(self, kind: str, principal: Principal, key: str, payload: Mapping[str, JsonValue],
                     *, expires_at: datetime | None = None) -> tuple[str, Mapping[str, JsonValue]]:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if self._task_retention is StorageRetentionMode.TRANSIENT:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        validate_idempotency_key(key)
        identity = canonical_sha256({"kind": kind, "key": key})
        if expires_at is not None and (expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc)):
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        value = {"kind": kind, "version": 1, "namespace": self._namespace,
                 "tenant_id": principal.tenant_id, "principal": principal_identity_payload(principal), "payload": dict(payload)}
        if kind == "agent":
            value["request_identity_version"] = 2
        if expires_at is not None:
            value["expires_at"] = expires_at.isoformat()
        return identity, value

    async def _publish(self, kind: str, principal: Principal, key: str, payload: Mapping[str, JsonValue],
                       *, expires_at: datetime | None = None) -> tuple[str, str]:
        if kind == "agent":
            payload = dict(payload)
            instructions = payload["repository_instructions"]
            if instructions is not None:
                payload["repository_instructions"] = await self._context_archive.encode_repository_instructions(instructions)
            context = payload.get("input_context")
            if context is not None:
                context = dict(context)
                if context["repository_instructions"] is not None:
                    context["repository_instructions"] = (
                        payload["repository_instructions"] if context["repository_instructions"] == instructions
                        else await self._context_archive.encode_repository_instructions(context["repository_instructions"])
                    )
                payload["input_context"] = context
        identity, value = self._publication(kind, principal, key, payload, expires_at=expires_at)
        object_key = self._key(principal.tenant_id, kind, identity)
        if await self._get(input_capture_expiry_key(object_key)) is not None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        try:
            digest = await self._put(object_key, value)
        except AIError as error:
            if kind != "agent" or error.code is not ErrorCode.IDEMPOTENCY_CONFLICT:
                raise
            existing = await self._get(object_key)
            if not isinstance(existing, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            version = existing.get("request_identity_version", 1)
            if type(version) is not int or version not in (1, 2):
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
            candidate = dict(value)
            if version == 1:
                candidate["payload"] = self._legacy_agent_payload(payload)
            if "request_identity_version" in existing:
                candidate["request_identity_version"] = version
            else:
                candidate.pop("request_identity_version", None)
            if canonical_json_bytes(candidate) != canonical_json_bytes(existing):
                raise
            digest = canonical_sha256(existing)
        return identity, digest

    @classmethod
    def _legacy_agent_payload(cls, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result = dict(payload)
        result["prompt"] = AgentTaskNode.prompt_identity_payload(
            payload["prompt"], request_identity_version=1,
        )
        view = payload.get("input_view")
        if isinstance(view, Mapping):
            view = dict(view)
            prompt = view.get("prompt")
            if isinstance(prompt, Mapping):
                view["prompt"] = AgentTaskNode.prompt_identity_payload(
                    prompt, request_identity_version=1,
                )
            result["input_view"] = view
        return result

    async def _read(self, reference: AgentInputCaptureRef | TaskInvocationInputRef | TaskGraphCaptureRef | TaskGraphTemplateRef,
                    kind: str, principal: Principal) -> Mapping[str, JsonValue]:
        if reference.namespace != self._namespace or reference.tenant_id != principal.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        graph = kind in {"graph", "template"}
        await self._authorize(principal, ResourceKind.TASK_GRAPH if graph else ResourceKind.EXECUTION,
                              reference.capture_id, AuthorizationAction.TASK_CAPTURE_GRAPH if graph else AuthorizationAction.EXECUTION_CAPTURE_INPUT)
        key = self._key(principal.tenant_id, kind, reference.capture_id)
        if await self._get(input_capture_expiry_key(key)) is not None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        value = await self._get(key, reference.digest)
        if value is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        if not isinstance(value, Mapping) or value.get("kind") != kind or value.get("namespace") != self._namespace or value.get("tenant_id") != principal.tenant_id or not isinstance(value.get("payload"), Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if kind == "agent":
            identity_version = value.get("request_identity_version", 1)
            if type(identity_version) is not int or identity_version not in (1, 2):
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        owner = value.get("principal")
        if not isinstance(owner, Mapping) or not isinstance(owner.get("principal_id"), str) or owner.get("tenant_id") != principal.tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._authorize(principal, ResourceKind.TASK_GRAPH if graph else ResourceKind.EXECUTION,
                              reference.capture_id, AuthorizationAction.TASK_CAPTURE_GRAPH if graph else AuthorizationAction.EXECUTION_CAPTURE_INPUT,
                              owner["principal_id"])
        if value.get("expires_at") is not None:
            try:
                expires_at = datetime.fromisoformat(value["expires_at"])
                if expires_at.tzinfo is None:
                    raise ValueError("capture expiry must be aware")
            except (ValueError, TypeError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if expires_at <= datetime.now(timezone.utc):
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        required = {
            "agent": {"prompt", "binding", "original_input", "source_execution_id", "source_invocation_id", "repository_instructions", "task_input", "source_principal"},
            "task": {"contract"}, "graph": {"source_graph_id", "mode", "template"}, "template": {"template"},
        }[kind]
        if not required.issubset(value["payload"]):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value["payload"]

    async def canonicalize_files(self, files: Sequence[str]) -> tuple[str, ...]:
        """Resolve file names through this producer's borrowed input source."""
        return await self._materializer.canonicalize_files(files)

    async def materialize_input(self, prompt: UserPromptInput, files: Sequence[str] = ()) -> CanonicalUserInput:
        """Read accepted input from its producer without publishing another capture."""
        canonical = await self._materializer.canonicalize_input(prompt)
        return await self._materializer.materialize(canonical, await self.canonicalize_files(files))

    async def create_agent_input(self, prompt: UserPromptInput, *, files: Sequence[str] = (), principal: Principal, idempotency_key: str,
                                 expires_at: datetime | None = None) -> AgentInputCaptureRef:
        await self._authorize(principal, ResourceKind.EXECUTION, "input-capture", AuthorizationAction.EXECUTION_CAPTURE_INPUT)
        canonical = await self.materialize_input(prompt, files)
        payload = {"prompt": task_prompt_draft(canonical), "input_view": dict(canonical.view) if isinstance(canonical, MaterializedUserContent) else None, "binding": None,
                   "original_input": {}, "source_execution_id": None,
                   "source_invocation_id": None, "repository_instructions": None, "task_input": None, "source_principal": None, "input_context": None}
        identity, digest = await self._publish("agent", principal, idempotency_key, payload, expires_at=expires_at)
        return AgentInputCaptureRef(self._namespace, principal.tenant_id, identity, digest, None)

    async def expire_inputs(self, *, principal: Principal, now: datetime, limit: int = 100) -> tuple[str, ...]:
        """Tombstone owned captures at their explicit deletion deadline."""
        validate_page_limit(limit)
        if now.tzinfo is None:
            raise ValueError("capture expiry time must be timezone-aware")
        await self._authorize(principal, ResourceKind.EVALUATION, "retention", AuthorizationAction.EVALUATION_PURGE)
        owner_scope = scope_key(self._namespace, principal.tenant_id, "task", "input_capture",
                                   "owner", (principal.principal_id,))
        candidates = []
        for kind in ("expired", "agent", "template"):
            cursor = None
            while len(candidates) < limit:
                query = RecordQuery(kind="input_capture", scope_key=owner_scope,
                    sort_key_prefix=f"v1/input-capture/{kind}/".encode("ascii"), limit=1000,
                    after_sort_key=None if cursor is None else cursor.sort_key,
                    after_key=None if cursor is None else cursor.key)
                records = await self._state_store.read(lambda transaction: transaction.list_records(query))
                for record in records:
                    value = capture_record_value(record)
                    if (not isinstance(value, Mapping) or value.get("namespace") != self._namespace
                            or value.get("tenant_id") != principal.tenant_id
                            or not isinstance(value.get("principal"), Mapping)
                            or value["principal"].get("principal_id") != principal.principal_id
                            or value["principal"].get("tenant_id") != principal.tenant_id):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if kind == "expired":
                        key = value["capture_key"]
                        if value.get("cleanup_objects") is not None and key not in candidates:
                            candidates.append(key)
                            if len(candidates) == limit:
                                break
                        continue
                    if value.get("expires_at") is None:
                        continue
                    try:
                        expires_at = datetime.fromisoformat(value["expires_at"])
                        if expires_at.tzinfo is None:
                            raise ValueError("capture expiry must be aware")
                    except (ValueError, TypeError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    if expires_at > now:
                        continue
                    key = capture_record_key(record)
                    await self._authorize(principal, ResourceKind.EVALUATION, key,
                                          AuthorizationAction.EVALUATION_PURGE, principal.principal_id)
                    expiry_key = input_capture_expiry_key(key)
                    if await self._get(expiry_key) is None:
                        cleanup_objects = tuple(iter_runtime_object_refs(capture_retention_value(record), default_domain=PersistenceDomain.TASK))
                        expired = {"namespace": self._namespace, "tenant_id": principal.tenant_id,
                            "capture_key": key, "expired_at": expires_at.isoformat(),
                            "principal": principal_identity_payload(principal)}
                        if cleanup_objects:
                            expired["cleanup_objects"] = encode_domain(cleanup_objects)
                        await self._put(expiry_key, expired)
                    if key not in candidates:
                        candidates.append(key)
                    if len(candidates) == limit:
                        break
                if len(records) < 1000:
                    break
                cursor = records[-1]
        return tuple(candidates)

    async def acknowledge_expired_input_objects(
        self, keys: tuple[str, ...], objects: tuple[tuple[PersistenceDomain, ObjectRef], ...], *, principal: Principal,
    ) -> None:
        """Retain expiry fences while acknowledging durably recorded attachment cleanup."""
        if not objects:
            return
        await self._authorize(principal, ResourceKind.EVALUATION, "retention", AuthorizationAction.EVALUATION_PURGE)
        for key in keys:
            expiry_key = input_capture_expiry_key(key)
            identity = capture_record(self._namespace, principal.tenant_id, expiry_key, None).key

            async def acknowledge(transaction: StateTransaction) -> None:
                record = await transaction.get_record(identity)
                if record is None:
                    return
                value = capture_record_value(record)
                if not isinstance(value, Mapping) or value.get("cleanup_objects") is None:
                    return
                owner = value.get("principal")
                if not isinstance(owner, Mapping) or owner.get("principal_id") != principal.principal_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                pending = decode_domain(value["cleanup_objects"], tuple[tuple[PersistenceDomain, ObjectRef], ...])
                remaining = tuple(item for item in pending if item not in objects)
                if remaining == pending:
                    return
                updated = dict(value)
                if remaining:
                    updated["cleanup_objects"] = encode_domain(remaining)
                else:
                    updated.pop("cleanup_objects")
                replacement = capture_record(self._namespace, principal.tenant_id, expiry_key, updated,
                                             storage_revision=record.storage_revision + 1)
                if not await transaction.replace_record(replacement, expected_storage_revision=record.storage_revision):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            await self._state_store.mutate(acknowledge)

    async def task_input_records(
        self, references: tuple[TaskInvocationInputRef, ...], *, principal: Principal,
    ) -> tuple[str, ...]:
        """Resolve owned derivations for cleanup without revoking retained references."""
        candidates = []
        for reference in references:
            if reference.namespace != self._namespace or reference.tenant_id != principal.tenant_id:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorize(principal, ResourceKind.EVALUATION, reference.capture_id,
                                  AuthorizationAction.EVALUATION_PURGE, principal.principal_id)
            key = self._key(principal.tenant_id, "task", reference.capture_id)
            value = await self._get(key, reference.digest)
            if value is not None:
                if (not isinstance(value, Mapping) or value.get("kind") != "task"
                        or value.get("namespace") != self._namespace or value.get("tenant_id") != principal.tenant_id
                        or not isinstance(value.get("principal"), Mapping)
                        or not isinstance(value["principal"].get("principal_id"), str)
                        or value["principal"].get("tenant_id") != principal.tenant_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._authorize(principal, ResourceKind.EVALUATION, reference.capture_id,
                                      AuthorizationAction.EVALUATION_PURGE, value["principal"]["principal_id"])
            candidates.append(key)
        return tuple(candidates)

    async def expire_task_input_records(
        self, keys: tuple[str, ...], *, principal: Principal, now: datetime,
    ) -> None:
        """Fence deleted or missing inputs from authorized receipts under offline exclusivity."""
        if now.tzinfo is None:
            raise ValueError("capture expiry time must be timezone-aware")
        for key in keys:
            if not key.startswith("v1/input-capture/task/"):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._authorize(principal, ResourceKind.EVALUATION, key,
                                  AuthorizationAction.EVALUATION_PURGE, principal.principal_id)
            if await self._get(key) is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "input_capture_still_retained"})
            expiry_key = input_capture_expiry_key(key)
            if await self._get(expiry_key) is None:
                await self._put(expiry_key, {"namespace": self._namespace,
                    "tenant_id": principal.tenant_id, "capture_key": key, "expired_at": now.isoformat()})

    async def record_invocation(self, execution_id: str, invocation: TaskNodeInvocation) -> None:
        state = await self._task_repositories.tasks.graph_state(invocation.graph_id, tenant_id=invocation.principal.tenant_id)
        node = next((node for node in state.nodes if node.node_id == invocation.node.node_id), None) if state is not None else None
        if node is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = {"graph_id": invocation.graph_id, "node": encode_domain(node),
                 "dependency_states": encode_domain({name: invocation.dependency_states[name] for name in node.dependencies})}
        await self._put(self._key(invocation.principal.tenant_id, "invocation", execution_id), value)

    async def read_invocation_node(
        self, execution_id: str, *, principal: Principal,
    ) -> tuple[str, TaskNode] | None:
        """Read the retained graph/node binding for execution recovery."""
        header = await self._execution_repositories.executions.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
        value = await self._get(self._key(principal.tenant_id, "invocation", execution_id))
        if value is None:
            return None
        if not isinstance(value, Mapping) or not isinstance(value.get("graph_id"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value["graph_id"], decode_domain(value.get("node"), TaskNode)

    async def record_graph(self, graph: TaskGraph, admission: TaskGraphAdmission) -> None:
        await self._put(self._key(admission.principal.tenant_id, "declaration", graph.graph_id),
                        encode_domain(TaskGraphTemplate(graph.nodes, admission.limits)))

    async def _payload(
        self, payload: StoredPayload, domain: PersistenceDomain, *, text: bool = False,
    ) -> JsonValue:
        if payload.encoding == "json-attachments":
            raw = await read_attachment_payload(payload,
                read_object=lambda ref: read_runtime_object(self._object_stores[domain], ref))
        elif payload.kind == "inline":
            return payload.decode()
        else:
            if payload.ref is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw = await read_runtime_object(self._object_stores[domain], payload.ref)
        try:
            return raw.decode("utf-8") if text else json.loads(raw)
        except (UnicodeDecodeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _capture_dependencies(self, node: TaskNode, graph_id: str,
                                    states: Mapping[str, TaskDependencyState], principal: Principal) -> tuple[TaskDependencyCapture, ...]:
        entries: dict[str, tuple[str, str, TaskDependencyState]] = {
            name: (graph_id, name, state) for name, state in states.items()
        }
        for name, reference in node.input_refs.items():
            if isinstance(reference, TaskNodeResultRef):
                state = states[reference.node_id]
                entries[name] = (graph_id, reference.node_id, state)
            else:
                if reference.namespace != self._namespace or reference.tenant_id != principal.tenant_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                entries[name] = (reference.graph_id, reference.node_id, TaskDependencyState(TaskStatus.SUCCEEDED, reference.result_digest))
        captured = []
        for name, (source_graph, source_node, state) in entries.items():
            if state.status is not TaskStatus.SUCCEEDED:
                captured.append(TaskDependencyCapture(name, state))
                continue
            admission = await self._task_repositories.admissions.get(source_graph, tenant_id=principal.tenant_id)
            if admission is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            await self._authorize(principal, ResourceKind.TASK_GRAPH, source_graph, AuthorizationAction.TASK_READ, admission.principal.principal_id)
            records = await self._task_repositories.tasks.get_results(source_graph, (source_node,), tenant_id=principal.tenant_id)
            record = records.get(source_node)
            if record is None or record.result_digest != state.result_digest:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            source = await self._execution_repositories.executions.get(record.execution_id, tenant_id=principal.tenant_id)
            if source is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            await self._authorize(principal, ResourceKind.EXECUTION, record.execution_id, AuthorizationAction.EXECUTION_READ, source.principal_id)
            await self._authorize(principal, ResourceKind.EXECUTION, record.execution_id, AuthorizationAction.EXECUTION_CAPTURE_INPUT, source.principal_id)
            result = await self._execution.result(record.execution_id, principal=principal)
            if canonical_sha256(result.output) != state.result_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._put(self._key(principal.tenant_id, "result", state.result_digest), result.output)
            captured.append(TaskDependencyCapture(name, state, TaskResultRef(self._namespace, principal.tenant_id, source_graph, source_node, state.result_digest), record.execution_id, state.result_digest))
        if node.input_capture is not None:
            previous = await self.read_task(node.input_capture, principal=principal)
            occupied = {item.name for item in captured}
            if occupied.intersection(item.name for item in previous.dependencies):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            captured.extend(previous.dependencies)
        return tuple(captured)

    async def capture_input(self, execution_id: str, request: CaptureInputRequest) -> ExecutionInputCaptureRef:
        principal = request.principal
        repository = self._execution_repositories.executions
        header = await repository.get_header(execution_id, tenant_id=principal.tenant_id)
        if header is None and self._prepare_pending_capture is not None:
            await self._prepare_pending_capture(execution_id, principal)
            header = await repository.state_store.read(
                lambda _transaction: repository.get_header(execution_id, tenant_id=principal.tenant_id)
            )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT, header)
        await self._authorization.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
        hold_id = None
        if self._execution_retention is StorageRetentionMode.TRANSIENT:
            hold_id = "input-capture:" + uuid.uuid4().hex
            try:
                await self._execution.acquire_dependency_hold(execution_id, tenant_id=principal.tenant_id, hold_id=hold_id)
            except AIError as error:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE) from error
        try:
            return await self._capture_input(execution_id, request)
        finally:
            if hold_id is not None:
                await self._execution.release_dependency_hold(execution_id, tenant_id=principal.tenant_id, hold_id=hold_id)

    async def _admitted_repository_instructions(
        self, reference: RuntimePayloadRef | None,
    ) -> dict[str, JsonValue] | None:
        if reference is None:
            return None
        instructions = await self._context_archive.load_repository_instruction_payload(
            reference, source_domain=PersistenceDomain.EXECUTION,
        )
        return instructions.to_payload()

    async def _capture_input(self, execution_id: str, request: CaptureInputRequest) -> ExecutionInputCaptureRef:
        principal = request.principal
        repository = self._execution_repositories.executions
        record = await repository.read_content(
            lambda _group: repository.get(execution_id, tenant_id=principal.tenant_id)
        )
        if record is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        await self._authorize(principal, ResourceKind.EXECUTION, execution_id, AuthorizationAction.EXECUTION_READ, record.principal_id)
        await self._authorize(principal, ResourceKind.EXECUTION, execution_id, AuthorizationAction.EXECUTION_CAPTURE_INPUT, record.principal_id)
        is_agent = isinstance(record.binding, AgentBindingContract)
        context = None
        if is_agent and request.context_policy == "captured":
            if record.input_context is not None:
                context = await self._context_archive.load(record.input_context)
                if context.unavailable_reason is not None:
                    raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE, safe_details={"reason": context.unavailable_reason})
            elif any(value is not None for value in (record.session_id, record.memory_scope, record.previous_execution_id, record.fork_base_execution_id)):
                raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE, safe_details={"reason": "pre_input_context_not_retained"})
            else:
                instructions = await self._admitted_repository_instructions(record.repository_instructions)
                context = ExecutionInputContext.from_messages((), repository_instructions=instructions)
        invocation = await self._get(self._key(principal.tenant_id, "invocation", execution_id))
        if invocation is None and record.requires_task_invocation_capture:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE,
                          safe_details={"reason": "task_invocation_not_retained"})
        task_input = None
        if isinstance(invocation, Mapping):
            node = decode_domain(invocation["node"], TaskNode)
            states = decode_domain(invocation["dependency_states"], dict[str, TaskDependencyState])
            dependencies = await self._capture_dependencies(node, invocation["graph_id"], states, principal)
            admission = await self._task_repositories.admissions.get(invocation["graph_id"], tenant_id=principal.tenant_id)
            if admission is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            bindings = await TaskGraphBindingCaptureStore(self._namespace, self._state_store).load(admission)
            declaration = bindings.tasks.get((node.task.id, node.task.revision))
            if declaration is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous = None if node.input_capture is None else await self.read_task(node.input_capture, principal=principal)
            task_input = TaskInvocationInputContract(execution_id, node.task,
                node.input if previous is None else previous.input,
                (node.original_input if node.original_input is not None else node.input) if previous is None else previous.original_input,
                declaration, dependencies)
        if is_agent:
            prompt = await self._materializer.restore(record.stored_user_input)
            if context is not None:
                instructions = context.repository_instructions
            elif record.input_context is not None:
                instructions = await self._context_archive.load_repository_instructions(record.input_context)
            else:
                instructions = await self._admitted_repository_instructions(record.repository_instructions)
            input_view = record.stored_user_input.view
            payload = {"prompt": task_prompt_draft(prompt), "input_view": None if input_view is None else dict(input_view), "binding": record.binding.to_payload(),
                       "original_input": dict(record.stored_user_input.view or {"prompt": task_prompt_draft(prompt)}) if task_input is None else dict(task_input.original_input),
                       "source_execution_id": execution_id, "source_invocation_id": record.parent_invocation_id,
                       "repository_instructions": dict(instructions) if isinstance(instructions, Mapping) else instructions,
                       "input_context": None if context is None else context.to_payload(),
                       **({"tool_response_ref": record.tool_response_ref.to_payload()} if record.tool_response_ref is not None else {}),
                       "source_principal": {"principal_id": record.principal_id, "tenant_id": principal.tenant_id, "kind": record.principal_kind},
                       "task_input": None if task_input is None else encode_domain(task_input)}
            identity, digest = await self._publish("agent", principal, request.idempotency_key, payload)
            return AgentInputCaptureRef(self._namespace, principal.tenant_id, identity, digest, execution_id)
        normalized = await self._payload(record.stored_user_input.payload, PersistenceDomain.EXECUTION)
        if not isinstance(normalized, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if task_input is None:
            task_input = TaskInvocationInputContract(execution_id, TaskRef(record.binding.id, record.binding.revision), normalized, normalized, encode_domain(record.binding))
        else:
            task_input = replace(task_input, input=normalized)
        identity, digest = await self._publish("task", principal, request.idempotency_key, {"contract": encode_domain(task_input)})
        return TaskInvocationInputRef(self._namespace, principal.tenant_id, identity, digest, execution_id)

    async def read_agent(self, reference: AgentInputCaptureRef, *, principal: Principal) -> AgentInputCapture:
        payload = dict(await self._read(reference, "agent", principal))
        instructions = payload["repository_instructions"]
        if instructions is not None:
            payload["repository_instructions"] = await self._context_archive.decode_repository_instructions(instructions)
        context = payload.get("input_context")
        if context is not None:
            context = dict(context)
            if context["repository_instructions"] is not None:
                context["repository_instructions"] = (
                    payload["repository_instructions"] if context["repository_instructions"] == instructions
                    else await self._context_archive.decode_repository_instructions(context["repository_instructions"])
                )
            payload["input_context"] = context
        source = payload["source_execution_id"]
        if source != reference.source_execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prompt = decode_task_prompt_draft(payload["prompt"])
        if not isinstance(prompt, str) and payload.get("input_view") is not None:
            prompt = MaterializedUserContent(prompt, payload["input_view"])
        return AgentInputCapture(prompt,
                                 None if payload["binding"] is None else AgentBindingContract.from_payload(payload["binding"]),
                                 ImmutableJsonMapping(payload["original_input"]), source,
                                 payload["source_invocation_id"], payload["repository_instructions"],
                                 None if payload["task_input"] is None else decode_domain(payload["task_input"], TaskInvocationInputContract),
                                 None if payload["source_principal"] is None else ImmutableJsonMapping(payload["source_principal"]),
                                 None if payload.get("input_context") is None else ExecutionInputContext.from_payload(payload["input_context"]),
                                 None if payload.get("tool_response_ref") is None else decode_domain(payload["tool_response_ref"], AssetVersionRef))

    async def read_task(self, reference: TaskInvocationInputRef, *, principal: Principal) -> TaskInvocationInputContract:
        payload = await self._read(reference, "task", principal)
        contract = decode_domain(payload["contract"], TaskInvocationInputContract)
        if contract.source_execution_id != reference.source_execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return contract

    async def read_execution_input(self, execution_id: str, *, principal: Principal) -> Mapping[str, JsonValue]:
        await self._execution.inspect(execution_id, principal=principal)
        record = await self._execution_repositories.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None or record.stored_user_input.codec != "task-input-v1":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        body = await self._payload(record.stored_user_input.payload, PersistenceDomain.EXECUTION)
        if not isinstance(body, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return body

    async def read_execution_prompt(self, execution_id: str, *, principal: Principal) -> CanonicalUserInput:
        """Read an Agent's accepted prompt without publishing a new capture."""
        await self._execution.inspect(execution_id, principal=principal)
        record = await self._execution_repositories.executions.get(execution_id, tenant_id=principal.tenant_id)
        if record is None or not isinstance(record.binding, AgentBindingContract):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._materializer.restore(record.stored_user_input)

    async def read_dependency(self, reference: TaskInvocationInputRef, name: str, *, principal: Principal) -> JsonValue:
        contract = await self.read_task(reference, principal=principal)
        item = next((item for item in contract.dependencies if item.name == name), None)
        if item is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if item.state.status is not TaskStatus.SUCCEEDED:
            raise AIError(ErrorCode.TASK_DEPENDENCY_FAILED)
        key = self._key(principal.tenant_id, "result", item.body_digest)
        value = await self._get(key, item.body_digest)
        if value is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        if canonical_sha256(value) != item.body_digest:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        return value

    def _captured_original_input(
        self, original: Mapping[str, JsonValue], accepted: CanonicalUserInput,
        previous: Mapping[str, JsonValue], context: ExecutionInputContext | None,
    ) -> dict[str, JsonValue]:
        normalized = dict(original)
        prompt = normalized.get("prompt")
        if isinstance(prompt, Mapping) and prompt.get("kind") != "stored-user-content-v1":
            original_prompt = decode_task_prompt_draft(prompt)
            frozen_prompt = captured_input_prompt(original_prompt, accepted)
            if frozen_prompt is not original_prompt:
                normalized["prompt"] = task_prompt_draft(frozen_prompt)
        if normalized.get("files"):
            normalized["capture_files"] = previous.get("capture_files") or task_prompt_draft(
                captured_input_files(accepted, len(normalized["files"])))
            normalized["files"] = []
        normalized["session_id"] = None
        normalized.pop("capture_context", None)
        if context is not None:
            normalized["capture_context"] = context.to_payload()
        return normalized

    def require_reprojectable_input(self, value: Mapping[str, JsonValue]) -> None:
        if value.get("kind") != "agent-task-input":
            return
        prompt = value.get("prompt")
        if not isinstance(prompt, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        original = decode_task_prompt_draft(prompt)
        if value.get("files") or (not isinstance(original, str)
                and any(isinstance(item, WorkspaceFileInput) for item in original)):
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE,
                          safe_details={"reason": "accepted_workspace_input_not_retained"})

    async def resolve_task_input(self, reference: ExecutionInputCaptureRef, *, principal: Principal,
                                 input_mode: str = "fixed_input", exclude_dependencies: tuple[str, ...] = ()) -> TaskInvocationInputContract:
        if input_mode not in {"fixed_input", "reproject_input"}:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if isinstance(reference, AgentInputCaptureRef):
            agent = await self.read_agent(reference, principal=principal)
            agent.require_importable()
            if agent.task_input is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            contract = agent.task_input
            original_input = self._captured_original_input(
                contract.original_input, agent.prompt, contract.input, agent.input_context)
            contract = replace(contract, original_input=original_input)
            if input_mode == "fixed_input":
                from ._agent_task_input import AgentTaskInput
                normalized = dict(AgentTaskInput(agent.prompt, planning=False, thinking=None,
                                                input_context=agent.input_context))
                normalized["capture_fixed_input"] = True
            else:
                normalized = original_input
        else:
            contract = await self.read_task(reference, principal=principal)
            normalized = dict(contract.input if input_mode == "fixed_input" else contract.original_input)
        agent_input = isinstance(reference, AgentInputCaptureRef) or contract.binding.get("type") == "agent"
        if agent_input and input_mode == "reproject_input":
            self.require_reprojectable_input(normalized)
        # A new Agent invocation never inherits its source session.
        if agent_input and normalized.get("kind") == "agent-task-input":
            normalized["session_id"] = None
        excluded = tuple(sorted(set(contract.excluded_dependencies) | set(exclude_dependencies)))
        contract = replace(contract, input=normalized, input_mode=input_mode,
                           dependencies=tuple(item for item in contract.dependencies if item.name not in excluded),
                           excluded_dependencies=excluded)
        return contract

    async def _validate_task_input_derivation(
        self, contract: TaskInvocationInputContract, source_capture: ExecutionInputCaptureRef,
        principal: Principal,
    ) -> None:
        if isinstance(source_capture, AgentInputCaptureRef):
            source = (await self.read_agent(source_capture, principal=principal)).task_input
            if source is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        else:
            source = await self.read_task(source_capture, principal=principal)
        dependencies = {item.name: item for item in source.dependencies}
        if (contract.source_execution_id != source.source_execution_id or contract.task_ref != source.task_ref
                or dict(contract.binding) != dict(source.binding)
                or any(dependencies.get(item.name) != item for item in contract.dependencies)):
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, safe_details={"reason": "input_capture_derivation_changed"})

    async def describe_task_input(self, contract: TaskInvocationInputContract, *, principal: Principal,
                                  source_capture: ExecutionInputCaptureRef, idempotency_key: str) -> TaskInvocationInputRef:
        await self._validate_task_input_derivation(contract, source_capture, principal)
        identity, value = self._publication("task", principal, idempotency_key, {"contract": encode_domain(contract)})
        return TaskInvocationInputRef(self._namespace, principal.tenant_id, identity, canonical_sha256(value),
                                      contract.source_execution_id)

    async def create_task_input(self, contract: TaskInvocationInputContract, *, principal: Principal,
                                source_capture: ExecutionInputCaptureRef, idempotency_key: str) -> TaskInvocationInputRef:
        await self._validate_task_input_derivation(contract, source_capture, principal)
        identity, digest = await self._publish("task", principal, idempotency_key, {"contract": encode_domain(contract)})
        return TaskInvocationInputRef(self._namespace, principal.tenant_id, identity, digest, contract.source_execution_id)

    async def task_input(self, reference: ExecutionInputCaptureRef, *, principal: Principal,
                         input_mode: str = "fixed_input", exclude_dependencies: tuple[str, ...] = (),
                         idempotency_key: str | None = None) -> TaskInvocationInputRef:
        contract = await self.resolve_task_input(reference, principal=principal, input_mode=input_mode,
                                                exclude_dependencies=exclude_dependencies)
        key = idempotency_key if idempotency_key is not None else deterministic_id(
            reference.digest, "capture", input_mode, canonical_sha256(list(contract.excluded_dependencies)))
        return await self.create_task_input(contract, principal=principal, source_capture=reference, idempotency_key=key)

    async def capture_graph(self, graph_id: str, request: CaptureGraphRequest) -> TaskGraphCaptureRef:
        principal = request.principal
        await self._authorize(principal, ResourceKind.TASK_GRAPH, graph_id, AuthorizationAction.TASK_READ)
        await self._authorize(principal, ResourceKind.TASK_GRAPH, graph_id, AuthorizationAction.TASK_CAPTURE_GRAPH)
        admission = await self._task_repositories.admissions.get(graph_id, tenant_id=principal.tenant_id)
        if admission is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        await self._authorize(principal, ResourceKind.TASK_GRAPH, graph_id, AuthorizationAction.TASK_READ, admission.principal.principal_id)
        await self._authorize(principal, ResourceKind.TASK_GRAPH, graph_id, AuthorizationAction.TASK_CAPTURE_GRAPH, admission.principal.principal_id)
        state = await self._task_repositories.tasks.graph_state(graph_id, tenant_id=principal.tenant_id)
        if state is None:
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        if request.mode == "materialized_graph":
            if state.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                raise AIError(ErrorCode.TASK_NOT_READY)
            nodes = state.nodes
        else:
            original = await self._get(self._key(principal.tenant_id, "declaration", graph_id))
            if original is None:
                # Static retained graphs have no expansion ambiguity.
                if any(node.expander is not None for node in state.nodes):
                    raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
                nodes = state.nodes
            else:
                nodes = decode_domain(original, TaskGraphTemplate).nodes
        bindings = await TaskGraphBindingCaptureStore(self._namespace, self._state_store).load(admission)
        selected = {node.node_id for node in nodes}
        sources = {item.node_id: item.execution_id for item in state.node_states}
        captured_nodes = []
        for node in nodes:
            input_refs = {}
            frozen_refs = {}
            for name, reference in node.input_refs.items():
                if isinstance(reference, TaskNodeResultRef):
                    input_refs[name] = reference
                elif reference.graph_id == graph_id and reference.node_id in selected:
                    if reference.node_id not in node.dependencies:
                        raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, safe_details={"reason": "result_source_not_scheduled"})
                    input_refs[name] = TaskNodeResultRef(reference.node_id)
                else:
                    frozen_refs[name] = reference
            body = dict(node.input)
            original_input = node.original_input
            input_capture = node.input_capture
            declaration = {} if node.task is None else bindings.tasks.get((node.task.id, node.task.revision), {})
            source = sources.get(node.node_id)
            if declaration.get("type") == "agent":
                if source is None:
                    if request.context_policy == "captured":
                        raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE, safe_details={"reason": "graph_node_never_started"})
                    previous = None if input_capture is None else await self.read_task(input_capture, principal=principal)
                    effective_input = body if previous is None else previous.input
                    if (declaration["config"]["input_mode"] == "projected"
                            and effective_input.get("capture_fixed_input") is not True):
                        raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, safe_details={"reason": "graph_node_never_started"})
                    original_input = dict(node.original_input if node.original_input is not None else node.input)
                    original_input.pop("capture_context", None)
                    original_input["session_id"] = None
                    if previous is not None:
                        clean_input = dict(previous.input)
                        original_input = dict(previous.original_input)
                        for value in (clean_input, original_input):
                            value.pop("capture_context", None)
                            value["session_id"] = None
                        contract = replace(previous, input=clean_input, original_input=original_input)
                        input_capture = await self.create_task_input(contract, principal=principal,
                            source_capture=input_capture,
                            idempotency_key=deterministic_id(request.idempotency_key, "context-input", node.node_id))
                else:
                    from ._agent_task_input import AgentTaskInput
                    reference = await self.capture_input(source, CaptureInputRequest(
                        principal, deterministic_id(request.idempotency_key, "input", node.node_id), request.context_policy))
                    agent = await self.read_agent(reference, principal=principal)
                    agent.require_importable()
                    original_input = self._captured_original_input(
                        agent.task_input.original_input if agent.task_input is not None
                        else node.original_input if node.original_input is not None else node.input,
                        agent.prompt, agent.task_input.input if agent.task_input is not None else node.input,
                        agent.input_context)
                    body = dict(AgentTaskInput(agent.prompt, planning=False, thinking=None,
                                              input_context=agent.input_context))
                    body["capture_fixed_input"] = True
                    if input_capture is not None:
                        previous = await self.read_task(input_capture, principal=principal)
                        contract = replace(previous, input=body, original_input=original_input, input_mode="fixed_input")
                        identity, digest = await self._publish("task", principal,
                            deterministic_id(request.idempotency_key, "context-input", node.node_id), {"contract": encode_domain(contract)})
                        input_capture = TaskInvocationInputRef(self._namespace, principal.tenant_id, identity, digest, contract.source_execution_id)
                        body = {}
            elif declaration.get("type") == "function" and input_capture is not None:
                if source is None:
                    previous = await self.read_task(input_capture, principal=principal)
                    if previous.input_mode == "reproject_input":
                        raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, safe_details={"reason": "graph_node_never_started"})
                else:
                    reference = await self.capture_input(source, CaptureInputRequest(
                        principal, deterministic_id(request.idempotency_key, "input", node.node_id), request.context_policy))
                    input_capture = await self.task_input(reference, principal=principal,
                        exclude_dependencies=(*node.dependencies, *input_refs),
                        idempotency_key=deterministic_id(request.idempotency_key, "accepted-input", node.node_id))
                    # Internal dependencies run again; external results belong to the captured invocation.
                    body = {}
                    original_input = None
                    frozen_refs = {}
            if declaration.get("type") == "agent" and body.get("kind") == "agent-task-input":
                if request.context_policy == "clean":
                    body.pop("capture_context", None)
                body["session_id"] = None
                prompt = body.get("prompt")
                if isinstance(prompt, Mapping) and prompt.get("kind") == "stored-user-content-v1":
                    stored = decode_domain(prompt["value"], StoredUserInput)
                    value = await self._payload(stored.payload, PersistenceDomain.TASK, text=stored.codec == "text")
                    stored = replace(stored, payload=StoredPayload.inline_text(value) if stored.codec == "text" else StoredPayload.inline_json(value))
                    accepted_prompt = await self._materializer.restore(stored)
                    body["prompt"] = task_prompt_draft(accepted_prompt)
                    body["files"] = []
                    if original_input is not None:
                        original_input = self._captured_original_input(original_input, accepted_prompt, body, None)
            if frozen_refs:
                frozen_node = TaskNode(node.node_id, task=node.task, input_refs=frozen_refs, input_capture=input_capture)
                dependencies = await self._capture_dependencies(frozen_node, graph_id, {}, principal)
                if input_capture is not None:
                    previous = await self.read_task(input_capture, principal=principal)
                    body = dict(previous.input)
                source_id = source or "graph:" + graph_id + ":" + node.node_id
                contract = TaskInvocationInputContract(source_id, node.task, body,
                    original_input if original_input is not None else body, declaration, dependencies)
                identity, digest = await self._publish("task", principal, deterministic_id(request.idempotency_key, "task-input", node.node_id),
                                                       {"contract": encode_domain(contract)})
                input_capture = TaskInvocationInputRef(self._namespace, principal.tenant_id, identity, digest, source_id)
                body = {}
            node_type = AgentTaskNode if declaration.get("type") == "agent" else TaskNode
            captured_nodes.append(node_type.from_resolved(node.node_id, node.dependencies, task=node.task, input=body,
                input_capture=input_capture, original_input=original_input, budget_cost=node.budget_cost,
                expander=None if request.mode == "materialized_graph" else node.expander,
                input_refs=input_refs, timeout_seconds=node.timeout_seconds, max_attempts=node.max_attempts,
                retry_delay_seconds=node.retry_delay_seconds, output_contract=node.output_contract,
                effect_policy=node.effect_policy, reconcile=node.reconcile,
                dependency_policy=node.dependency_policy, failure_policy=node.failure_policy))
        if request.mode == "materialized_graph":
            required_tasks = {(node.task.id, node.task.revision) for node in captured_nodes if node.task is not None}
            task_contracts = tuple(value for identity, value in bindings.tasks.items() if identity in required_tasks)
            expander_contracts = ()
        else:
            task_contracts = tuple(bindings.tasks.values())
            expander_contracts = tuple(bindings.expanders.values())
        template = TaskGraphTemplate(tuple(captured_nodes), admission.limits,
                                     task_contracts, expander_contracts, request.context_policy)
        identity, digest = await self._publish("graph", principal, request.idempotency_key,
            {"source_graph_id": graph_id, "mode": request.mode, "template": encode_domain(template)})
        return TaskGraphCaptureRef(self._namespace, principal.tenant_id, identity, digest, graph_id)

    async def create_graph_template(self, template: TaskGraphTemplate, *, principal: Principal,
                                    idempotency_key: str) -> TaskGraphTemplateRef:
        await self._authorize(principal, ResourceKind.TASK_GRAPH, "graph-template", AuthorizationAction.TASK_CAPTURE_GRAPH)
        identity, digest = await self._publish("template", principal, idempotency_key, {"template": encode_domain(template)})
        return TaskGraphTemplateRef(self._namespace, principal.tenant_id, identity, digest)

    async def read_graph(self, reference: TaskGraphCaptureRef | TaskGraphTemplateRef, *, principal: Principal) -> TaskGraphTemplate:
        kind = "graph" if isinstance(reference, TaskGraphCaptureRef) else "template"
        payload = await self._read(reference, kind, principal)
        if kind == "graph" and payload["source_graph_id"] != reference.source_graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return decode_domain(payload["template"], TaskGraphTemplate)


__all__ = ["AgentInputCapture", "CaptureInputRequest", "CaptureGraphRequest",
           "ExecutionInputCaptureRef", "InputCaptures", "InputCaptureExecutionPort"]
