from pathlib import Path

ROOT = Path('.')


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise SystemExit(f'{label}: expected exactly one anchor, found {text.count(old)}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


# 1. Complete durable identity migration, including Evaluation records.
path = ROOT / 'linktools-ai/src/linktools/ai/runtime/state/_migration.py'
text = path.read_text(encoding='utf-8')
text = text.replace(
    '    ContextProjection,\n    ExecutionRecord,\n    RecoveryAdmissionRecord,\n',
    '    ContextProjection,\n    EvaluationRecord,\n    ExecutionRecord,\n    RecoveryAdmissionRecord,\n',
    1,
)
text = text.replace(
    '    ExecutionRepositoryImpl,\n    RecoveryCheckpointRepositoryImpl,\n',
    '    EvaluationRepositoryImpl,\n    ExecutionRepositoryImpl,\n    RecoveryCheckpointRepositoryImpl,\n',
    1,
)
text = text.replace(
    '    recovery = state.recovery.checkpoints\n',
    '    recovery = state.recovery.checkpoints\n    evaluations = state.evaluation.records\n',
    1,
)
text = text.replace(
    '    if not isinstance(recovery, RecoveryCheckpointRepositoryImpl):\n        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n\n    repositories = {\n',
    '    if not isinstance(recovery, RecoveryCheckpointRepositoryImpl):\n        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n    if not isinstance(evaluations, EvaluationRepositoryImpl):\n        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n\n    repositories = {\n',
    1,
)
text = text.replace(
    '        RuntimeDomain.RECOVERY: recovery,\n    }\n',
    '        RuntimeDomain.RECOVERY: recovery,\n        RuntimeDomain.EVALUATION: evaluations,\n    }\n',
    1,
)
text = text.replace(
    '    legacy_bindings: dict[str, AgentBinding] = {}\n    legacy_agents: dict[str, str] = {}\n    execution_records = await _records(executions, "execution")\n',
    '    legacy_bindings: dict[str, AgentBinding] = {}\n    legacy_agents: dict[str, str] = {}\n    execution_targets: dict[str, tuple[str, str]] = {}\n    execution_records = await _records(executions, "execution")\n',
    1,
)
text = text.replace(
    '        _collect_execution_binding(\n            record, compiler, catalog, legacy_bindings, legacy_agents\n        )\n',
    '        _collect_execution_binding(\n            record,\n            compiler,\n            catalog,\n            legacy_bindings,\n            legacy_agents,\n            execution_targets,\n        )\n',
    1,
)
needle = '''    # Exact execution/recovery records are migrated last. Their old snapshots\n    # are the evidence used above to migrate Session/projection identity.\n'''
insert = '''    # Evaluation binds to the exact executable identity of its linked\n    # execution. Reconcile it before rewriting Execution so a crash at any\n    # boundary remains restartable. Historical evaluation idempotency request\n    # digests are intentionally not rewritten because the original request is\n    # not fully reconstructable.\n    for record in await _records(evaluations, "evaluation"):\n        data = _migrate_evaluation_data(record, execution_targets)\n        if data is not None:\n            await _replace_data(evaluations.state_store, record, data)\n            migrated += 1\n\n    # Exact execution/recovery records are migrated last. Their old snapshots\n    # are the evidence used above to migrate Session/projection identity.\n'''
if text.count(needle) != 1:
    raise SystemExit('evaluation migration insertion anchor not found')
text = text.replace(needle, insert, 1)

start = text.index('def _collect_execution_binding(')
end = text.index('\ndef _collect_recovery_binding(', start)
new_collect = '''def _collect_execution_binding(\n    record: StoredRecord,\n    compiler: AgentCompiler,\n    catalog: AgentCatalog,\n    bindings: dict[str, AgentBinding],\n    agents: dict[str, str],\n    targets: dict[str, tuple[str, str]],\n) -> None:\n    fields = _domain_fields(\n        record.data,\n        type_name="execution_record",\n        wire_id="execution_record",\n    )\n    if frozenset(fields) != _EXECUTION_V1_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    binding = fields["binding"]\n    if _is_current_binding(binding):\n        current = _decode_enveloped_domain(record.data, ExecutionRecord)\n        _remember_execution_target(\n            targets,\n            current.execution_id,\n            current.binding_digest,\n            current.binding.output_schema_fingerprint,\n        )\n        return\n    if binding is None:\n        raise AIError(\n            ErrorCode.STORAGE_VERSION_UNSUPPORTED,\n            safe_details={"record": "execution", "reason": "missing_exact_binding"},\n        )\n    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)\n    if _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC) != legacy_digest:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    execution_id = _decode_domain(fields["execution_id"], str, _CURRENT_CODEC)\n    if not isinstance(execution_id, str) or not execution_id:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    _remember_binding(bindings, agents, legacy_digest, migrated, catalog)\n    _remember_execution_target(\n        targets,\n        execution_id,\n        migrated.digest,\n        migrated.snapshot.output_schema_fingerprint,\n    )\n\n\ndef _remember_execution_target(\n    targets: dict[str, tuple[str, str]],\n    execution_id: str,\n    binding_digest: str,\n    output_schema_fingerprint: str,\n) -> None:\n    target = (binding_digest, output_schema_fingerprint)\n    previous = targets.get(execution_id)\n    if previous is not None and previous != target:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    targets[execution_id] = target\n\n'''
text = text[:start] + new_collect + text[end + 1:]

insert_at = text.index('def _migrate_execution_data(')
new_eval = '''def _migrate_evaluation_data(\n    record: StoredRecord,\n    targets: Mapping[str, tuple[str, str]],\n) -> Mapping[str, JsonValue] | None:\n    value = _decode_enveloped_domain(record.data, EvaluationRecord)\n    target = targets.get(value.execution_id)\n    if target is None:\n        # The linked Execution may have been retained elsewhere or already\n        # released. Without that authority there is no safe identity rewrite.\n        return None\n    binding_digest, output_schema_fingerprint = target\n    if value.output_schema_fingerprint != output_schema_fingerprint:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    if value.binding_digest == binding_digest:\n        return None\n    return _domain_data(replace(value, binding_digest=binding_digest))\n\n\n'''
text = text[:insert_at] + new_eval + text[insert_at:]
path.write_text(text, encoding='utf-8')

# 2. Keep migration internal to Runtime state implementation.
path = ROOT / 'linktools-ai/src/linktools/ai/runtime/state/__init__.py'
text = path.read_text(encoding='utf-8')
text = text.replace('from ._migration import migrate_v1_agent_identity_state\n', '', 1)
text = text.replace('    "migrate_v1_agent_identity_state",\n', '', 1)
path.write_text(text, encoding='utf-8')

path = ROOT / 'linktools-ai/src/linktools/ai/runtime/_factory.py'
text = path.read_text(encoding='utf-8')
text = text.replace('    migrate_v1_agent_identity_state,\n', '', 1)
anchor = 'from .state import (\n'
if text.count(anchor) != 1:
    raise SystemExit('runtime factory state import anchor not found')
# Import from the implementation module without expanding the public state API.
block_end = text.index(')\n\n_logger', text.index(anchor))
text = text[:block_end + 2] + 'from .state._migration import migrate_v1_agent_identity_state\n\n' + text[block_end + 2:]
path.write_text(text, encoding='utf-8')

# 3. Close a stale Session identity reference in fork replay.
path = ROOT / 'linktools-ai/src/linktools/ai/runtime/state/_repositories.py'
text = path.read_text(encoding='utf-8')
old = '            or existing_target.binding_digest != target.binding_digest\n'
new = '            or existing_target.agent_digest != target.agent_digest\n'
if text.count(old) != 1:
    raise SystemExit(f'session fork identity anchor count={text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

# 4. Extend migration conformance with Evaluation and crash/restart convergence.
path = ROOT / 'tests/ai/test_runtime_state_identity_migration.py'
text = path.read_text(encoding='utf-8')
text = text.replace(
    'from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, SessionStatus\n',
    'from linktools.ai.core import (\n    EvaluationStatus,\n    ExecutionLineageKind,\n    ExecutionStatus,\n    SessionStatus,\n)\n',
    1,
)
text = text.replace(
    'from linktools.ai.runtime.state import (\n    RuntimeDomain,\n    SessionRecord,\n    migrate_v1_agent_identity_state,\n)\n',
    'from linktools.ai.runtime.state import RuntimeDomain, SessionRecord\nfrom linktools.ai.runtime.state._migration import migrate_v1_agent_identity_state\n',
    1,
)
text = text.replace(
    '    ContextProjection,\n    ExecutionRecord,\n',
    '    ContextProjection,\n    EvaluationRecord,\n    ExecutionRecord,\n',
    1,
)
append = r'''

@pytest.mark.asyncio
async def test_migration_rewrites_evaluation_from_linked_legacy_execution() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="legacy-evaluation", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    legacy_digest = "a" * 64
    execution = _execution(binding)
    executions = state.execution.executions
    stored = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    stored = replace(
        stored,
        data=_legacy_execution_data(execution, binding, legacy_digest),
    )
    await executions.state_store.mutate(lambda tx: tx.insert_record(stored))
    now = datetime.now(timezone.utc)
    evaluation = EvaluationRecord(
        evaluation_id="evaluation",
        tenant_id="tenant",
        execution_id="execution",
        dataset_id="dataset",
        dataset_revision=1,
        evaluator_id="default",
        evaluator_revision=1,
        binding_digest=legacy_digest,
        output_schema_fingerprint=binding.snapshot.output_schema_fingerprint,
        artifact_digest=None,
        status=EvaluationStatus.SUCCEEDED,
        revision=0,
        metrics={},
        created_at=now,
        updated_at=now,
    )
    await state.evaluation.records.create(evaluation)
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 2
        migrated = await state.evaluation.records.get(
            "evaluation", tenant_id="tenant"
        )
        assert migrated is not None
        assert migrated.binding_digest == binding.digest
        assert migrated.output_schema_fingerprint == binding.snapshot.output_schema_fingerprint
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_converges_evaluation_after_execution_was_already_migrated() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="partial-evaluation", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    execution = _execution(binding)
    executions = state.execution.executions
    await executions.state_store.mutate(
        lambda tx: tx.insert_record(
            executions._stored(
                "execution",
                execution.execution_id,
                execution,
                state=execution.status.value,
            )
        )
    )
    now = datetime.now(timezone.utc)
    evaluation = EvaluationRecord(
        evaluation_id="evaluation",
        tenant_id="tenant",
        execution_id=execution.execution_id,
        dataset_id="dataset",
        dataset_revision=1,
        evaluator_id="default",
        evaluator_revision=1,
        binding_digest="f" * 64,
        output_schema_fingerprint=binding.snapshot.output_schema_fingerprint,
        artifact_digest=None,
        status=EvaluationStatus.SUCCEEDED,
        revision=0,
        metrics={},
        created_at=now,
        updated_at=now,
    )
    await state.evaluation.records.create(evaluation)
    try:
        assert await migrate_v1_agent_identity_state(
            state, catalog, compiler, tenant_id="tenant"
        ) == 1
        migrated = await state.evaluation.records.get(
            "evaluation", tenant_id="tenant"
        )
        assert migrated is not None
        assert migrated.binding_digest == binding.digest
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_migration_rejects_evaluation_output_contract_drift() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="evaluation-contract-drift", tenant_id="tenant")
    compiler, catalog = _compiler_catalog()
    binding = compiler.bind(catalog.root_definition("default"))
    execution = _execution(binding)
    executions = state.execution.executions
    await executions.state_store.mutate(
        lambda tx: tx.insert_record(
            executions._stored(
                "execution",
                execution.execution_id,
                execution,
                state=execution.status.value,
            )
        )
    )
    now = datetime.now(timezone.utc)
    await state.evaluation.records.create(
        EvaluationRecord(
            evaluation_id="evaluation",
            tenant_id="tenant",
            execution_id=execution.execution_id,
            dataset_id="dataset",
            dataset_revision=1,
            evaluator_id="default",
            evaluator_revision=1,
            binding_digest="e" * 64,
            output_schema_fingerprint="d" * 64,
            artifact_digest=None,
            status=EvaluationStatus.SUCCEEDED,
            revision=0,
            metrics={},
            created_at=now,
            updated_at=now,
        )
    )
    try:
        with pytest.raises(AIError) as raised:
            await migrate_v1_agent_identity_state(
                state, catalog, compiler, tenant_id="tenant"
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
'''
if 'test_migration_rewrites_evaluation_from_linked_legacy_execution' in text:
    raise SystemExit('evaluation migration tests already present')
path.write_text(text.rstrip() + append + '\n', encoding='utf-8')

# 5. SQLite regression uses the internal migration indirectly through Runtime startup;
# no migration helper is part of the public API.

print('final migration and identity closure patch applied')
