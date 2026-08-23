from pathlib import Path

path = Path('tests/ai/test_runtime_composition_regressions.py')
text = path.read_text(encoding='utf-8')
text = text.replace(
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        output_type_module=output.value_type.__module__,\n''',
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n''', 1)

start = text.index('def _execution(*, binding: AgentBindingSnapshot | None = None) -> ExecutionRecord:\n')
end = text.index('\n\ndef _recovery(', start)
text = text[:start] + '''def _execution(*, binding: AgentBindingSnapshot | None = None) -> ExecutionRecord:\n    now = datetime.now(timezone.utc)\n    selected = binding or _binding()\n    return ExecutionRecord(\n        execution_id="execution",\n        tenant_id="tenant",\n        session_id=None,\n        binding_digest=selected.binding_digest,\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        lineage_kind=ExecutionLineageKind.RUN,\n        status=ExecutionStatus.PENDING_START,\n        revision=0,\n        event_sequence=0,\n        agent_run_sequence=0,\n        error_code=None,\n        safe_error_details={},\n        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=selected,\n    )\n''' + text[end:]

start = text.index('def _recovery(\n')
end = text.index('\n\n@pytest.mark.asyncio\nasync def test_approval_list_authorizes_before_reading_pending_records', start)
text = text[:start] + '''def _recovery(\n    *,\n    binding: AgentBindingSnapshot | None = None,\n) -> RecoveryExecutionInput:\n    selected = binding or _binding()\n    return RecoveryExecutionInput(\n        user_prompt="prompt",\n        principal_id="principal",\n        principal_kind="service",\n        session_id=None,\n        memory_scope=None,\n        binding_digest=selected.binding_digest,\n        lineage_kind=ExecutionLineageKind.RUN.value,\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        conversation_step_run_id=None,\n        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),\n        planning=False,\n        thinking=False,\n        binding=selected,\n    )\n''' + text[end:]

start = text.index('@pytest.mark.parametrize(\n    ("factory", "target"),\n    (\n        (_execution, ExecutionRecord),\n        (_recovery, RecoveryExecutionInput),\n    ),\n)\ndef test_binding_codec_accepts_legacy_and_current_exact_v1_shapes')
end = text.index('\n\n@pytest.mark.parametrize(\n    ("factory", "target"),', start)
text = text[:start] + '''@pytest.mark.parametrize(\n    ("factory", "target"),\n    (\n        (_execution, ExecutionRecord),\n        (_recovery, RecoveryExecutionInput),\n    ),\n)\ndef test_binding_codec_round_trips_mandatory_exact_v1_shape(\n    factory: object,\n    target: type[object],\n) -> None:\n    current = factory(binding=_binding())\n    assert decode_domain(encode_domain(current), target) == current\n''' + text[end:]

path.write_text(text, encoding='utf-8')
print('runtime composition regressions migrated to mandatory exact binding')
