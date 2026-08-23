#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize reconstructable pre-composition Runtime V1 state exactly once.

The normal Runtime codec remains strict. This module recognizes only frozen
legacy shapes written before AgentDefinition and AgentBinding identities were
split, reconstructs the current exact binding, and rewrites those records to
the single current V1 shape before normal Runtime reads begin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ...agent import AgentBinding, AgentBindingSnapshot, AgentCatalog, AgentCompiler
from ...capability import RuntimeCapability
from ...core import JsonValue, canonical_sha256
from ...errors import AIError, ErrorCode
from ...spec import AgentSpec, AgentUsageLimits
from ._codec import (
    _CURRENT_CODEC,
    _decode_domain,
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_domain,
    encode_envelope,
    parse_envelope,
    wire_type_id,
)
from ._contracts import (
    ContextProjection,
    EvaluationRecord,
    ExecutionRecord,
    RecoveryAdmissionRecord,
    SessionRecord,
)
from ._plan import RuntimeDomain
from ._repositories import (
    EvaluationRepositoryImpl,
    ExecutionRepositoryImpl,
    RecoveryCheckpointRepositoryImpl,
    SessionRepositoryImpl,
    _domain_data,
)
from ._root import RuntimeState
from ._steps import StateStepArchive
from ._store import RecordQuery, StateStore, StateTransaction, StoredRecord, sequence_key

_PAGE_SIZE = 128
_MIGRATION_ID = "agent_identity_v1"
_LEGACY_BINDING_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "output_type_module",
        "output_type_qualname",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
    }
)
_LEGACY_AGENT_SPEC_FIELDS = frozenset(
    {
        "id",
        "revision",
        "model",
        "system_prompt",
        "instructions",
        "allow_tools",
        "metadata",
        "usage_limits",
    }
)
# Frozen shapes owned by this one-time semantic migration.  They describe
# the exact unversioned V1 wire before/after the Agent identity split and must
# never be derived from the current Runtime dataclasses or codec implementation.
_POST_COMPOSITION_EXECUTION_V1_FIELDS = frozenset(
    {
        "execution_id",
        "tenant_id",
        "session_id",
        "binding_digest",
        "parent_execution_id",
        "root_execution_id",
        "source_execution_id",
        "base_execution_id",
        "lineage_kind",
        "status",
        "revision",
        "event_sequence",
        "agent_run_sequence",
        "error_code",
        "safe_error_details",
        "created_at",
        "updated_at",
        "planning",
        "thinking",
        "binding",
        "memory_scope",
        "conversation_step_run_id",
        "result",
    }
)
_POST_COMPOSITION_SESSION_V1_FIELDS = frozenset(
    {
        "session_id",
        "tenant_id",
        "owner_principal_id",
        "agent_digest",
        "status",
        "revision",
        "resource_generation",
        "cwd",
        "metadata",
        "created_at",
        "updated_at",
        "closed_at",
        "active_execution_id",
        "continuation",
        "history_quality",
        "history_id",
    }
)
_PRE_COMPOSITION_SESSION_V1_FIELDS = (
    _POST_COMPOSITION_SESSION_V1_FIELDS - {"agent_digest"}
) | {"binding_digest"}
_POST_COMPOSITION_PROJECTION_V1_FIELDS = frozenset(
    {"agent_digest", "items", "digest"}
)
_PRE_COMPOSITION_PROJECTION_V1_FIELDS = (
    _POST_COMPOSITION_PROJECTION_V1_FIELDS - {"agent_digest"}
) | {"binding_digest"}
_POST_COMPOSITION_RECOVERY_ADMISSION_V1_FIELDS = frozenset(
    {"execution_id", "tenant_id", "input", "created_at"}
)
_POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS = frozenset(
    {
        "user_prompt",
        "principal_id",
        "principal_kind",
        "session_id",
        "memory_scope",
        "binding_digest",
        "lineage_kind",
        "parent_execution_id",
        "root_execution_id",
        "source_execution_id",
        "base_execution_id",
        "conversation_step_run_id",
        "idempotency",
        "planning",
        "thinking",
        "binding",
    }
)
_PRE_COMPOSITION_RECOVERY_INPUT_V1_FIELDS = (
    _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS | {"agent_id"}
)


async def migrate_v1_agent_identity_state(
    state: RuntimeState,
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    *,
    tenant_id: str,
) -> int:
    """Normalize known legacy V1 Agent identity shapes in place.

    Per-domain StateStore sequences are completion markers, so normal startup
    does not rescan history. Markers are written only after every recognized
    record has been normalized. A crash before that point is safe because the
    migration is idempotent and current records are never rewritten.
    """
    if not state.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    sessions = state.conversation.sessions
    executions = state.execution.executions
    recovery = state.recovery.checkpoints
    evaluations = state.evaluation.records
    if not isinstance(sessions, SessionRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not isinstance(executions, ExecutionRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not isinstance(recovery, RecoveryCheckpointRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not isinstance(evaluations, EvaluationRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    repositories = {
        RuntimeDomain.CONVERSATION: sessions,
        RuntimeDomain.EXECUTION: executions,
        RuntimeDomain.RECOVERY: recovery,
        RuntimeDomain.EVALUATION: evaluations,
    }
    completed = [
        await _migration_complete(repository, domain)
        for domain, repository in repositories.items()
    ]
    if all(completed):
        return 0

    # Validate every old-digest -> current-binding relation before any write.
    # This prevents a malformed execution from partially migrating its Session.
    legacy_bindings: dict[str, AgentBinding] = {}
    legacy_agents: dict[str, str] = {}
    execution_targets: dict[str, tuple[str, str]] = {}
    execution_records = await _records(executions, "execution")
    recovery_records = await _records(recovery, "recovery_admission")
    for record in execution_records:
        _collect_execution_binding(
            record,
            compiler,
            catalog,
            legacy_bindings,
            legacy_agents,
            execution_targets,
        )
    for record in recovery_records:
        _collect_recovery_binding(
            record, compiler, catalog, legacy_bindings, legacy_agents
        )

    migrated = 0
    for record in await _records(sessions, "session"):
        current = _migrate_session_record(record, sessions, legacy_agents)
        if current is not None:
            await _replace_data(sessions.state_store, record, _domain_data(current))
            migrated += 1

    # Projection digests are historical references from StoredStepSnapshot and
    # execution seals. Replace only the identity field and preserve `digest`.
    for domain in (
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.RECOVERY,
    ):
        archive = state.steps.read_store(domain)
        if not isinstance(archive, StateStepArchive):
            continue
        history = archive.transcript_repository
        for record in await _records(history, "context_projection"):
            data = _migrate_projection_data(record, legacy_agents)
            if data is not None:
                await _replace_data(history._store, record, data)
                migrated += 1

    # Evaluation binds to the exact executable identity of its linked
    # execution. Reconcile it before rewriting Execution so a crash at any
    # boundary remains restartable. Historical evaluation idempotency request
    # digests are intentionally not rewritten because the original request is
    # not fully reconstructable.
    for record in await _records(evaluations, "evaluation"):
        data = _migrate_evaluation_data(record, execution_targets)
        if data is not None:
            await _replace_data(evaluations.state_store, record, data)
            migrated += 1

    # Exact execution/recovery records are migrated last. Their old snapshots
    # are the evidence used above to migrate Session/projection identity.
    for record in execution_records:
        data = _migrate_execution_data(
            record, compiler, catalog, legacy_bindings, legacy_agents
        )
        if data is not None:
            await _replace_data(executions.state_store, record, data)
            migrated += 1
    for record in recovery_records:
        data = _migrate_recovery_data(
            record, compiler, catalog, legacy_bindings, legacy_agents
        )
        if data is not None:
            await _replace_data(recovery.state_store, record, data)
            migrated += 1

    for domain, repository in repositories.items():
        await _mark_migration_complete(repository, domain)
    return migrated


async def _migration_complete(repository: object, domain: RuntimeDomain) -> bool:
    key = _migration_key(repository, domain)
    value = await repository.state_store.read(
        lambda transaction: transaction.get_sequence(key)
    )
    return value > 0


async def _mark_migration_complete(repository: object, domain: RuntimeDomain) -> None:
    key = _migration_key(repository, domain)

    async def mutate(transaction: StateTransaction) -> None:
        if await transaction.get_sequence(key) == 0:
            await transaction.next_sequence(key)

    await repository.state_store.mutate(mutate)


def _migration_key(repository: object, domain: RuntimeDomain) -> bytes:
    return sequence_key(
        repository._namespace,
        repository._tenant_id,
        domain.value,
        "migration",
        _MIGRATION_ID,
    )


async def _records(repository: object, kind: str) -> tuple[StoredRecord, ...]:
    store = repository._store
    partition = repository._partition(kind)
    values: list[StoredRecord] = []
    after_sort_key: str | None = None
    after_key_digest: bytes | None = None
    while True:
        page = await store.read(
            lambda transaction, sort_key=after_sort_key, key_digest=after_key_digest: transaction.list_records(
                RecordQuery(
                    partition_digest=partition,
                    kind=kind,
                    after_sort_key=sort_key,
                    after_key_digest=key_digest,
                    limit=_PAGE_SIZE,
                )
            )
        )
        if not page:
            return tuple(values)
        values.extend(page)
        if len(page) < _PAGE_SIZE:
            return tuple(values)
        last = page[-1]
        after_sort_key = last.sort_key
        after_key_digest = last.key_digest


async def _replace_data(
    store: StateStore,
    current: StoredRecord,
    data: Mapping[str, JsonValue],
) -> None:
    candidate = replace(
        current,
        data=dict(data),
        storage_version=current.storage_version + 1,
    )
    replaced = await store.mutate(
        lambda transaction: transaction.replace_record(
            candidate,
            expected_storage_version=current.storage_version,
        )
    )
    if not replaced:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _domain_fields(
    data: Mapping[str, JsonValue],
    *,
    type_name: str,
    wire_id: str,
) -> Mapping[str, object]:
    envelope = parse_envelope(data)
    if envelope.version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    value = envelope.value
    if set(value) != {"type", "payload"} or value.get("type") != type_name:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    payload = value.get("payload")
    return _persisted_dataclass_fields(payload, wire_id)


def _decode_current_record(
    data: Mapping[str, JsonValue],
    target: type[object],
) -> object | None:
    try:
        return _decode_enveloped_domain(data, target)
    except AIError as error:
        if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
            return None
        raise


def _collect_execution_binding(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    bindings: dict[str, AgentBinding],
    agents: dict[str, str],
    targets: dict[str, tuple[str, str]],
) -> None:
    current = _decode_current_record(record.data, ExecutionRecord)
    if isinstance(current, ExecutionRecord):
        _remember_execution_target(
            targets,
            current.execution_id,
            current.binding_digest,
            current.binding.output_schema_fingerprint,
        )
        return
    fields = _domain_fields(
        record.data,
        type_name="execution_record",
        wire_id="execution_record",
    )
    if frozenset(fields) != _POST_COMPOSITION_EXECUTION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding = fields["binding"]
    if _is_current_binding(binding):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if binding is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={"record": "execution", "reason": "missing_exact_binding"},
        )
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    if _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    execution_id = _decode_domain(fields["execution_id"], str, _CURRENT_CODEC)
    if not isinstance(execution_id, str) or not execution_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _remember_binding(bindings, agents, legacy_digest, migrated, catalog)
    _remember_execution_target(
        targets,
        execution_id,
        migrated.digest,
        migrated.snapshot.output_schema_fingerprint,
    )


def _remember_execution_target(
    targets: dict[str, tuple[str, str]],
    execution_id: str,
    binding_digest: str,
    output_schema_fingerprint: str,
) -> None:
    target = (binding_digest, output_schema_fingerprint)
    previous = targets.get(execution_id)
    if previous is not None and previous != target:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    targets[execution_id] = target

def _collect_recovery_binding(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    bindings: dict[str, AgentBinding],
    agents: dict[str, str],
) -> None:
    current = _decode_current_record(record.data, RecoveryAdmissionRecord)
    if isinstance(current, RecoveryAdmissionRecord):
        return
    fields = _domain_fields(
        record.data,
        type_name="recovery_admission",
        wire_id="recovery_admission",
    )
    if frozenset(fields) != _POST_COMPOSITION_RECOVERY_ADMISSION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    input_fields = _nested_dataclass_fields(
        fields.get("input"), "recovery_execution_input"
    )
    keys = frozenset(input_fields)
    binding = input_fields.get("binding")
    if keys == _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if keys != _PRE_COMPOSITION_RECOVERY_INPUT_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if binding is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={
                "record": "recovery_admission",
                "reason": "missing_exact_binding",
            },
        )
    if _is_current_binding(binding):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    if _decode_domain(input_fields["binding_digest"], str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_id = _decode_domain(input_fields["agent_id"], str, _CURRENT_CODEC)
    if agent_id != migrated.definition.spec.id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _remember_binding(bindings, agents, legacy_digest, migrated, catalog)


def _remember_binding(
    bindings: dict[str, AgentBinding],
    agents: dict[str, str],
    legacy_digest: str,
    binding: AgentBinding,
    catalog: AgentCatalog,
) -> None:
    previous = bindings.get(legacy_digest)
    if previous is not None and previous.snapshot != binding.snapshot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_digest = binding.definition.digest
    previous_agent = agents.get(legacy_digest)
    if previous_agent is not None and previous_agent != agent_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    bindings[legacy_digest] = binding
    agents[legacy_digest] = agent_digest
    catalog.register_definition(binding.definition)
    catalog.register_binding(binding)


def _migrate_evaluation_data(
    record: StoredRecord,
    targets: Mapping[str, tuple[str, str]],
) -> Mapping[str, JsonValue] | None:
    value = _decode_enveloped_domain(record.data, EvaluationRecord)
    target = targets.get(value.execution_id)
    if target is None:
        # The linked Execution may have been retained elsewhere or already
        # released. Without that authority there is no safe identity rewrite.
        return None
    binding_digest, output_schema_fingerprint = target
    if value.output_schema_fingerprint != output_schema_fingerprint:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value.binding_digest == binding_digest:
        return None
    return _domain_data(replace(value, binding_digest=binding_digest))


def _migrate_execution_data(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    bindings: dict[str, AgentBinding],
    agents: dict[str, str],
) -> Mapping[str, JsonValue] | None:
    current = _decode_current_record(record.data, ExecutionRecord)
    if isinstance(current, ExecutionRecord):
        return None
    fields = _domain_fields(
        record.data,
        type_name="execution_record",
        wire_id="execution_record",
    )
    if frozenset(fields) != _POST_COMPOSITION_EXECUTION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding = fields["binding"]
    if _is_current_binding(binding):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if binding is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    if _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _remember_binding(bindings, agents, legacy_digest, migrated, catalog)
    current_fields = dict(fields)
    current_fields["binding_digest"] = encode_domain(migrated.digest)
    current_fields["binding"] = migrated.snapshot.to_payload()
    data = _schema1_data("execution_record", current_fields)
    value = _decode_enveloped_domain(data, ExecutionRecord)
    return _domain_data(value)


def _migrate_recovery_data(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    bindings: dict[str, AgentBinding],
    agents: dict[str, str],
) -> Mapping[str, JsonValue] | None:
    current = _decode_current_record(record.data, RecoveryAdmissionRecord)
    if isinstance(current, RecoveryAdmissionRecord):
        return None
    fields = _domain_fields(
        record.data,
        type_name="recovery_admission",
        wire_id="recovery_admission",
    )
    if frozenset(fields) != _POST_COMPOSITION_RECOVERY_ADMISSION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    input_fields = _nested_dataclass_fields(
        fields.get("input"), "recovery_execution_input"
    )
    keys = frozenset(input_fields)
    binding = input_fields.get("binding")
    if keys == _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if keys != _PRE_COMPOSITION_RECOVERY_INPUT_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if binding is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if _is_current_binding(binding):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    if _decode_domain(input_fields["binding_digest"], str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_id = _decode_domain(input_fields["agent_id"], str, _CURRENT_CODEC)
    if agent_id != migrated.definition.spec.id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _remember_binding(bindings, agents, legacy_digest, migrated, catalog)
    next_input_fields = {
        key: value for key, value in input_fields.items() if key != "agent_id"
    }
    next_input_fields["binding_digest"] = encode_domain(migrated.digest)
    next_input_fields["binding"] = migrated.snapshot.to_payload()
    if frozenset(next_input_fields) != _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    next_fields = dict(fields)
    next_fields["input"] = _schema1_dataclass(
        "recovery_execution_input", next_input_fields
    )
    data = _schema1_data("recovery_admission", next_fields)
    value = _decode_enveloped_domain(data, RecoveryAdmissionRecord)
    return _domain_data(value)


def _migrate_session_record(
    record: StoredRecord,
    repository: SessionRepositoryImpl,
    agents: Mapping[str, str],
) -> SessionRecord | None:
    current = _decode_current_record(record.data, SessionRecord)
    if isinstance(current, SessionRecord):
        if current.tenant_id != repository._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return None
    fields = _domain_fields(
        record.data,
        type_name="session_record",
        wire_id="session_record",
    )
    keys = frozenset(fields)
    if keys == _POST_COMPOSITION_SESSION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if keys != _PRE_COMPOSITION_SESSION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest = _decode_domain(
        fields["binding_digest"], str, _CURRENT_CODEC
    )
    agent_digest = agents.get(legacy_digest)
    if agent_digest is None:
        session_id = _decode_domain(fields["session_id"], str, _CURRENT_CODEC)
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={
                "record": "session",
                "reason": "binding_evidence_unavailable",
                "session_id": session_id,
            },
        )
    current_fields = {
        key: value
        for key, value in fields.items()
        if key != "binding_digest"
    }
    current_fields["agent_digest"] = encode_domain(agent_digest)
    if frozenset(current_fields) != _POST_COMPOSITION_SESSION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = _decode_enveloped_domain(
        _schema1_data("session_record", current_fields),
        SessionRecord,
    )
    if value.tenant_id != repository._tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    return value


def _migrate_projection_data(
    record: StoredRecord,
    agents: Mapping[str, str],
) -> Mapping[str, JsonValue] | None:
    current = _decode_current_record(record.data, ContextProjection)
    if isinstance(current, ContextProjection):
        return None
    fields = _domain_fields(
        record.data,
        type_name="context_projection",
        wire_id="context_projection",
    )
    keys = frozenset(fields)
    if keys == _POST_COMPOSITION_PROJECTION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if keys != _PRE_COMPOSITION_PROJECTION_V1_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest = _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC)
    agent_digest = agents.get(legacy_digest)
    if agent_digest is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={
                "record": "context_projection",
                "reason": "binding_evidence_unavailable",
            },
        )
    current_fields = {
        "agent_digest": encode_domain(agent_digest),
        "items": fields["items"],
        "digest": fields["digest"],
    }
    data = _schema1_data("context_projection", current_fields)
    value = _decode_enveloped_domain(data, ContextProjection)
    return encode_envelope(
        {
            "type": wire_type_id(value),
            "payload": _encode_persisted_domain(value),
        }
    )


def _migrate_legacy_binding(
    value: object,
    compiler: AgentCompiler,
) -> tuple[str, AgentBinding]:
    if not isinstance(value, Mapping) or frozenset(value) != _LEGACY_BINDING_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    legacy_digest = _digest(value.get("binding_digest"))
    spec = _legacy_agent_spec(value.get("agent_spec"))
    descriptors = value.get("local_runtime_capability_descriptors")
    if not isinstance(descriptors, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    descriptor_values = tuple(_json_mapping(descriptor) for descriptor in descriptors)
    module_name = _string(value.get("output_type_module"))
    qualname = _string(value.get("output_type_qualname"))
    schema_id = _string(value.get("output_schema_id"))
    schema_revision = _positive_int(value.get("output_schema_revision"))
    schema_fingerprint = _digest(value.get("output_schema_fingerprint"))
    try:
        capabilities = tuple(
            RuntimeCapability.restore(descriptor)
            for descriptor in descriptor_values
        )
        definition = compiler.compile(spec, capabilities=capabilities)
        output_binding_fingerprint = canonical_sha256(
            {
                "schema_id": schema_id,
                "schema_revision": schema_revision,
                "schema_fingerprint": schema_fingerprint,
                "module": module_name,
                "qualname": qualname,
            }
        )
        binding_digest = canonical_sha256(
            {
                "version": 1,
                "agent_digest": definition.digest,
                "output_binding_fingerprint": output_binding_fingerprint,
            }
        )
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=definition.spec,
            agent_digest=definition.digest,
            output_type_module=module_name,
            output_type_qualname=qualname,
            output_schema_id=schema_id,
            output_schema_revision=schema_revision,
            output_schema_fingerprint=schema_fingerprint,
            local_runtime_capability_descriptors=(
                definition.local_runtime_capability_descriptors
            ),
            binding_digest=binding_digest,
        )
        binding = compiler.restore(snapshot)
    except AIError as error:
        if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
            raise
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
    if binding.snapshot != snapshot:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return legacy_digest, binding

def _legacy_agent_spec(value: object) -> AgentSpec:
    if not isinstance(value, Mapping) or frozenset(value) != _LEGACY_AGENT_SPEC_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(value.get("metadata"), Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    instructions = value.get("instructions")
    allow_tools = value.get("allow_tools")
    if not isinstance(instructions, list) or not isinstance(allow_tools, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AgentSpec(
            id=_string(value.get("id")),
            revision=_positive_int(value.get("revision")),
            model=_string(value.get("model")),
            system_prompt=_string(value.get("system_prompt"), allow_empty=True),
            instructions=tuple(
                _string(item, allow_empty=True) for item in instructions
            ),
            allow_tools=tuple(_string(item) for item in allow_tools),
            usage_limits=_usage_limits(value.get("usage_limits")),
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _usage_limits(value: object) -> AgentUsageLimits | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    names = {
        "model_requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    if set(value) != names:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    kwargs: dict[str, int | None] = {}
    for name in names:
        item = value[name]
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kwargs[name] = item
    try:
        return AgentUsageLimits(**kwargs)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _persisted_dataclass_fields(
    value: object,
    wire_id: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(value) != {"$dataclass", "fields"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value.get("$dataclass") != wire_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return raw_fields


def _nested_dataclass_fields(value: object, wire_id: str) -> Mapping[str, object]:
    return _persisted_dataclass_fields(value, wire_id)


def _schema1_dataclass(
    wire_id: str,
    fields: Mapping[str, object],
) -> dict[str, JsonValue]:
    return {
        "$dataclass": wire_id,
        "schema": 1,
        "fields": dict(fields),
    }  # type: ignore[return-value]


def _schema1_data(
    wire_id: str,
    fields: Mapping[str, object],
) -> Mapping[str, JsonValue]:
    return encode_envelope(
        {
            "type": wire_id,
            "payload": _schema1_dataclass(wire_id, fields),
        }
    )


def _is_current_binding(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        AgentBindingSnapshot.from_payload(value)
    except AIError as error:
        if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
            raise
        return False
    return True


def _json_mapping(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value  # type: ignore[return-value]


def _string(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _digest(value: object) -> str:
    text = _string(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return text


__all__ = ["migrate_v1_agent_identity_state"]
