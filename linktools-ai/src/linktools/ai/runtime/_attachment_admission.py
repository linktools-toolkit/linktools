#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime integration for managed input admission and projection."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, cast

import linktools.ai.runtime._attachment as attachment_runtime
import linktools.ai.runtime._execution as execution_runtime
import linktools.ai.runtime._local as local_runtime
from pydantic_ai.messages import BinaryContent, UserContent

from ..core import canonical_sha256, idempotency_key_digest
from ..errors import AIError, ErrorCode
from ._attachment import DefaultAttachmentService, InputPreparer
from ._input import (
    UserPromptTransport,
    _decode_user_content,
    _restore_input_v2,
    _restore_user_prompt,
    input_intent_digest,
    managed_user_prompt_draft,
    prepared_user_prompt_transport,
)
from ._object import read_runtime_object
from ._runtime_service import Runtime
from .state import (
    AttachmentEntry,
    ExecutionRepositoryImpl,
    InputPrepareRecord,
    InputTarget,
    Locator,
    PreparedInput,
    RuntimeDomain,
    managed_attachment_locator,
)
from .state._attachment_repository import (
    AttachmentRepository,
    _project_owner_record,
)
from .state._repositories import replace_checked

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core import Principal
    from .service_api import ExecutionRequest
    from .state import ExecutionRecord
    from .state._contracts import (
        ExecutionStartReservation,
        ExecutionStartReservationResult,
        RecoveryExecutionInput,
    )
    from .state._store import StateTransaction


@dataclass(frozen=True, slots=True)
class _ManagedAdmission:
    scope: str
    idempotency_key: str
    prepared: PreparedInput


_managed_admission: ContextVar[_ManagedAdmission | None] = ContextVar(
    "linktools_ai_managed_attachment_admission",
    default=None,
)
_model_prompt: ContextVar[tuple[UserContent, ...] | None] = ContextVar(
    "linktools_ai_managed_model_prompt",
    default=None,
)
_installed = False
_original_request_digest = execution_runtime._request_digest
_original_reserve_start = ExecutionRepositoryImpl.reserve_start
_original_prepare_start = local_runtime.LocalExecutionBackend.prepare_start
_original_validate_recovery_identity = local_runtime.LocalExecutionBackend._validate_recovery_identity
_original_run = local_runtime.LocalExecutionBackend._run
_original_user_prompt_transport = local_runtime.user_prompt_transport
_original_start_for_agent = Runtime._start_for_agent
_original_retry_execution = Runtime._retry_execution
_original_fork_execution = Runtime._fork_execution


@contextmanager
def _admission_scope(value: _ManagedAdmission) -> Iterator[None]:
    token = _managed_admission.set(value)
    try:
        yield
    finally:
        _managed_admission.reset(token)


def _prepared_owner(value: PreparedInput) -> str:
    if not value.attachment_manifest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    owner: str | None = None
    for slot, entry in enumerate(value.attachment_manifest):
        try:
            kind, current_owner, current_slot = managed_attachment_locator(entry.path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if kind != "p" or current_slot != slot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if owner is None:
            owner = current_owner
        elif owner != current_owner:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if owner is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return owner


def _managed_request_digest(
    request: "ExecutionRequest",
    binding_digest: str,
    *,
    session_id: str | None,
    source_execution_id: str | None,
    base_execution_id: str | None,
    parent_execution_id: str | None,
    root_execution_id: str | None,
    lineage_kind,
) -> str:
    base = _original_request_digest(
        request,
        binding_digest,
        session_id=session_id,
        source_execution_id=source_execution_id,
        base_execution_id=base_execution_id,
        parent_execution_id=parent_execution_id,
        root_execution_id=root_execution_id,
        lineage_kind=lineage_kind,
    )
    admission = _managed_admission.get()
    if admission is None:
        return base
    if (
        request.idempotency_key != admission.idempotency_key
        or request.user_prompt_codec != "linktools-input-v2"
    ):
        return base
    return canonical_sha256(
        {
            "version": 1,
            "request_digest": base,
            "input_digest": admission.prepared.input_digest,
        }
    )


def _validate_execution_input(execution: "ExecutionRecord", prepared: PreparedInput) -> None:
    if (
        execution.attachment_manifest != prepared.attachment_manifest
        or execution.input_digest != prepared.input_digest
        or execution.path_origin != prepared.path_origin
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _adopt_prepare_in_transaction(
    transaction: "StateTransaction",
    repository: AttachmentRepository,
    owner_key: str,
    prepared: PreparedInput,
    *,
    execution_id: str,
) -> None:
    stored = await transaction.get_record(bytes.fromhex(owner_key))
    if stored is None or stored.kind != "input_prepare":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    current = await repository._decode_prepare(stored)
    if (
        current.intent_digest != prepared.intent_digest
        or current.path_origin != prepared.path_origin
    ):
        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    target = InputTarget(
        Locator("state:execution", "records", execution_id),
        None,
    )
    if current.status == "ADOPTED":
        if current.target != target:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return
    if current.status != "READY" or current.input != prepared:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    adopted = InputPrepareRecord(
        1,
        current.intent_digest,
        current.path_origin,
        "ADOPTED",
        (),
        None,
        target,
        None,
    )
    candidate = _project_owner_record(stored, adopted, state="ADOPTED")
    await replace_checked(transaction, candidate, stored.storage_version)


async def _managed_reserve_start(
    self: ExecutionRepositoryImpl,
    reservation: "ExecutionStartReservation",
) -> "ExecutionStartReservationResult":
    admission = _managed_admission.get()
    if admission is None or reservation.idempotency.scope != admission.scope:
        return await _original_reserve_start(self, reservation)
    if (
        reservation.idempotency.idempotency_key_digest
        != idempotency_key_digest(admission.idempotency_key)
    ):
        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    prepared = admission.prepared
    owner_key = _prepared_owner(prepared)
    execution = replace(
        reservation.execution,
        attachment_manifest=prepared.attachment_manifest,
        input_digest=prepared.input_digest,
        path_origin=prepared.path_origin,
    )
    effective = replace(reservation, execution=execution)
    attachments = AttachmentRepository(
        self.state_store,
        namespace=self._namespace,
        tenant_id=self._tenant_id,
    )

    async def mutate(transaction: "StateTransaction") -> "ExecutionStartReservationResult":
        result = await _original_reserve_start(self, effective)
        _validate_execution_input(result.execution, prepared)
        await _adopt_prepare_in_transaction(
            transaction,
            attachments,
            owner_key,
            prepared,
            execution_id=result.execution.execution_id,
        )
        return result

    return await self.state_store.mutate(mutate)


def _validate_managed_request(
    request: "ExecutionRequest",
    execution: "ExecutionRecord",
) -> None:
    if not execution.attachment_manifest:
        if request.user_prompt_codec == "linktools-input-v2":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return
    if request.user_prompt_codec != "linktools-input-v2":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    prompt = _restore_input_v2(request.user_prompt)
    from .state import input_v2_digest

    if input_v2_digest(prompt, execution.attachment_manifest) != execution.input_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _managed_prepare_start(
    self: local_runtime.LocalExecutionBackend,
    request: "ExecutionRequest",
    execution: "ExecutionRecord",
    identity: execution_runtime.ExecutionStartIdentity,
) -> "ExecutionRecord":
    if not execution.attachment_manifest:
        return await _original_prepare_start(self, request, execution, identity)
    _validate_managed_request(request, execution)
    await self._validate_start(request, execution)
    if execution.status is not local_runtime.ExecutionStatus.PENDING_START:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding = self._catalog.binding(execution.binding_digest)
    if execution.binding != binding.snapshot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    now = local_runtime.datetime.now(local_runtime.timezone.utc)
    recovery_input = local_runtime.RecoveryExecutionInput(
        user_prompt=local_runtime._recovery_prompt_payload(request.user_prompt),
        user_prompt_codec=request.user_prompt_codec,
        principal_id=request.principal.principal_id,
        principal_kind=request.principal.kind,
        session_id=execution.session_id,
        memory_scope=execution.memory_scope,
        binding_digest=execution.binding_digest,
        lineage_kind=execution.lineage_kind.value,
        parent_execution_id=execution.parent_execution_id,
        root_execution_id=execution.root_execution_id,
        source_execution_id=execution.source_execution_id,
        base_execution_id=execution.base_execution_id,
        conversation_step_run_id=execution.conversation_step_run_id,
        idempotency=local_runtime.RecoveryIdempotencyInput(
            scope=identity.scope,
            idempotency_key_digest=identity.idempotency_key_digest,
            request_digest=identity.request_digest,
        ),
        mode=execution.mode,
        planning=execution.planning,
        thinking=execution.thinking,
        binding=execution.binding,
        repository_instructions=execution.repository_instructions,
        attachment_manifest=execution.attachment_manifest,
        input_digest=execution.input_digest,
        path_origin=execution.path_origin,
        correlation=execution.correlation,
    )
    candidate = local_runtime.RecoveryCheckpoint(
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        input=recovery_input,
        step_run_id=None,
        agent_run_sequence=execution.agent_run_sequence,
        state=local_runtime.RecoveryCheckpointState.ADMITTED,
        handoff_phase=local_runtime.RecoveryHandoffPhase.NONE,
        terminal_handoff=None,
        handoff_contract_digest=None,
        pending_operation_id=None,
        revision=0,
        created_at=now,
        updated_at=now,
    )
    expected = (
        await self._expected_session_cursor(execution)
        if execution.session_id is not None
        else None
    )
    started = await self._runtime_commands.commit_start_attempt_checkpoint(
        local_runtime.ExecutionStartClaim(
            execution.execution_id,
            execution.tenant_id,
            execution.revision,
            execution.event_sequence,
            identity.scope,
            identity.idempotency_key_digest,
            identity.request_digest,
            now,
        ),
        recovery_checkpoint=candidate if self._recovery_enabled else None,
        session_id=execution.session_id,
        expected_cursor=expected,
    )
    local_runtime._logger.info(
        "execution start checkpoint committed: execution=%s",
        execution.execution_id,
    )
    return started


def _managed_validate_recovery_identity(
    self: local_runtime.LocalExecutionBackend,
    execution: "ExecutionRecord",
    recovery_input: "RecoveryExecutionInput",
) -> None:
    _original_validate_recovery_identity(self, execution, recovery_input)
    if (
        execution.attachment_manifest != recovery_input.attachment_manifest
        or execution.input_digest != recovery_input.input_digest
        or execution.path_origin != recovery_input.path_origin
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _read_attachment_body(
    backend: local_runtime.LocalExecutionBackend,
    entry: AttachmentEntry,
) -> bytes:
    try:
        domain = RuntimeDomain(entry.content.domain)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if domain is RuntimeDomain.EXECUTION:
        store = backend._execution_objects
    elif domain is RuntimeDomain.RECOVERY:
        store = backend._recovery_objects
    else:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    return await read_runtime_object(store, entry.content.object)


async def _project_model_prompt(
    backend: local_runtime.LocalExecutionBackend,
    execution: "ExecutionRecord",
    user_prompt: str,
) -> tuple[UserContent, ...]:
    prompt = _restore_input_v2(user_prompt)
    if execution.input_digest is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    from .state import input_v2_digest

    if input_v2_digest(prompt, execution.attachment_manifest) != execution.input_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    projected: list[UserContent] = []
    loaded: dict[int, BinaryContent] = {}
    for part in prompt.parts:
        if part.kind == "text":
            projected.append(cast("UserContent", part.text))
            continue
        if part.kind == "native":
            projected.extend(_decode_user_content(dict(part.value)))
            continue
        if part.kind != "attachment" or part.index >= len(execution.attachment_manifest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = loaded.get(part.index)
        if content is None:
            entry = execution.attachment_manifest[part.index]
            body = await _read_attachment_body(backend, entry)
            content = BinaryContent(
                body,
                media_type=entry.media_type,
                identifier=entry.presentation.identifier,
                vendor_metadata=(
                    None
                    if entry.presentation.vendor_metadata is None
                    else dict(entry.presentation.vendor_metadata)
                ),
            )
            loaded[part.index] = content
        projected.append(content)
    return tuple(projected)


async def _managed_run(
    self: local_runtime.LocalExecutionBackend,
    request: "ExecutionRequest",
    original: "ExecutionRecord",
) -> None:
    if not original.attachment_manifest:
        await _original_run(self, request, original)
        return
    _validate_managed_request(request, original)
    prompt = await _project_model_prompt(self, original, request.user_prompt)
    token = _model_prompt.set(prompt)
    try:
        await _original_run(self, request, original)
    finally:
        _model_prompt.reset(token)


def _managed_user_prompt_transport(value: str, codec: str = "text") -> UserPromptTransport:
    if codec != "linktools-input-v2":
        return _original_user_prompt_transport(value, codec)
    prompt = _model_prompt.get()
    if prompt is None:
        return _original_user_prompt_transport(value, codec)
    return UserPromptTransport("managed-input", "linktools-managed-draft", prompt)


def _runtime_attachment_service(runtime: Runtime) -> DefaultAttachmentService:
    service = runtime.attachments
    if not isinstance(service, DefaultAttachmentService):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return service


async def _replay_adopted_input(
    runtime: Runtime,
    raw_prompt,
    attachments: "Sequence[str]",
    *,
    principal: "Principal",
    scope: str,
    idempotency_key: str,
) -> PreparedInput:
    service = _runtime_attachment_service(runtime)
    repository = service._repository
    owner_key = repository.prepare_key(scope, idempotency_key)
    record = await repository.get_prepare(owner_key, tenant_id=principal.tenant_id)
    if record is None or record.status != "ADOPTED" or record.target is None:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if (
        record.target.node_id is not None
        or record.target.at.resource != "state:execution"
        or record.target.at.space != "records"
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    intent_digest = input_intent_digest(raw_prompt, attachments)
    if record.intent_digest != intent_digest:
        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    execution = await service._state.execution.executions.get(
        record.target.at.key,
        tenant_id=principal.tenant_id,
    )
    if execution is None or not execution.attachment_manifest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if execution.path_origin != record.path_origin or execution.input_digest is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    plan = attachment_runtime._build_input_plan(
        attachment_runtime._prompt_items(raw_prompt),
        attachment_runtime._attachment_sources(attachments),
    )
    prompt = attachment_runtime._build_input_v2(plan, execution.attachment_manifest)
    return PreparedInput(
        1,
        "linktools-input-v2",
        prompt,
        execution.attachment_manifest,
        intent_digest,
        execution.input_digest,
        execution.path_origin,
    )


async def _prepare_runtime_input(
    runtime: Runtime,
    user_prompt: UserPromptTransport,
    attachments: "Sequence[str]",
    *,
    principal: "Principal",
    scope: str,
    idempotency_key: str,
) -> PreparedInput | None:
    draft = managed_user_prompt_draft(user_prompt)
    raw_prompt = draft if draft is not None else _restore_user_prompt(user_prompt)
    if not attachments and draft is None:
        return None
    service = _runtime_attachment_service(runtime)
    preparer = InputPreparer(
        service._repository,
        service._state,
        service._object_key_factory,
        runtime.workspace,
    )
    if not preparer.requires_managed_input(raw_prompt, attachments):
        return None
    try:
        return await preparer.prepare(
            raw_prompt,
            attachments,
            principal=principal,
            scope=scope,
            idempotency_key=idempotency_key,
        )
    except AIError as error:
        if error.code is not ErrorCode.STORAGE_CONFLICT:
            raise
    return await _replay_adopted_input(
        runtime,
        raw_prompt,
        attachments,
        principal=principal,
        scope=scope,
        idempotency_key=idempotency_key,
    )


async def _managed_start_for_agent(
    self: Runtime,
    agent_digest: str,
    user_prompt: UserPromptTransport,
    *,
    attachments: "Sequence[str]",
    output,
    principal,
    session_id,
    idempotency_key,
    memory_scope,
    mode,
    planning,
    thinking,
    correlation=None,
):
    resolved_principal = self._resolve_principal(principal)
    key = idempotency_key or secrets.token_urlsafe(32)
    scope = "execution.run" if session_id is None else "session.resume"
    prepared = await _prepare_runtime_input(
        self,
        user_prompt,
        attachments,
        principal=resolved_principal,
        scope=scope,
        idempotency_key=key,
    )
    if prepared is None:
        return await _original_start_for_agent(
            self,
            agent_digest,
            user_prompt,
            attachments=attachments,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode=mode,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
        )
    transport = prepared_user_prompt_transport(prepared)
    with _admission_scope(_ManagedAdmission(scope, key, prepared)):
        return await _original_start_for_agent(
            self,
            agent_digest,
            transport,
            attachments=(),
            output=output,
            principal=resolved_principal,
            session_id=session_id,
            idempotency_key=key,
            memory_scope=memory_scope,
            mode=mode,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
        )


async def _managed_retry_execution(
    self: Runtime,
    binding_digest: str,
    execution_id: str,
    user_prompt: UserPromptTransport,
    *,
    attachments: "Sequence[str]",
    principal: "Principal",
    idempotency_key,
    correlation=None,
):
    key = idempotency_key or secrets.token_urlsafe(32)
    scope = "execution.retry"
    prepared = await _prepare_runtime_input(
        self,
        user_prompt,
        attachments,
        principal=principal,
        scope=scope,
        idempotency_key=key,
    )
    if prepared is None:
        return await _original_retry_execution(
            self,
            binding_digest,
            execution_id,
            user_prompt,
            attachments=attachments,
            principal=principal,
            idempotency_key=idempotency_key,
            correlation=correlation,
        )
    with _admission_scope(_ManagedAdmission(scope, key, prepared)):
        return await _original_retry_execution(
            self,
            binding_digest,
            execution_id,
            prepared_user_prompt_transport(prepared),
            attachments=(),
            principal=principal,
            idempotency_key=key,
            correlation=correlation,
        )


async def _managed_fork_execution(
    self: Runtime,
    binding_digest: str,
    execution_id: str,
    user_prompt: UserPromptTransport,
    *,
    attachments: "Sequence[str]",
    principal: "Principal",
    idempotency_key,
    correlation=None,
):
    key = idempotency_key or secrets.token_urlsafe(32)
    scope = "execution.fork"
    prepared = await _prepare_runtime_input(
        self,
        user_prompt,
        attachments,
        principal=principal,
        scope=scope,
        idempotency_key=key,
    )
    if prepared is None:
        return await _original_fork_execution(
            self,
            binding_digest,
            execution_id,
            user_prompt,
            attachments=attachments,
            principal=principal,
            idempotency_key=idempotency_key,
            correlation=correlation,
        )
    with _admission_scope(_ManagedAdmission(scope, key, prepared)):
        return await _original_fork_execution(
            self,
            binding_digest,
            execution_id,
            prepared_user_prompt_transport(prepared),
            attachments=(),
            principal=principal,
            idempotency_key=key,
            correlation=correlation,
        )


def install_attachment_admission() -> None:
    """Install managed-input hooks on existing Runtime owners exactly once."""
    global _installed
    if _installed:
        return
    execution_runtime._request_digest = _managed_request_digest
    ExecutionRepositoryImpl.reserve_start = _managed_reserve_start
    local_runtime.LocalExecutionBackend.prepare_start = _managed_prepare_start
    local_runtime.LocalExecutionBackend._validate_recovery_identity = (
        _managed_validate_recovery_identity
    )
    local_runtime.LocalExecutionBackend._run = _managed_run
    local_runtime.user_prompt_transport = _managed_user_prompt_transport
    Runtime._start_for_agent = _managed_start_for_agent
    Runtime._retry_execution = _managed_retry_execution
    Runtime._fork_execution = _managed_fork_execution
    _installed = True


__all__ = ["install_attachment_admission"]
