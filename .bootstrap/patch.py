from pathlib import Path

path = Path('tests/ai/test_agent_composition.py')
text = path.read_text(encoding='utf-8')
text = text.replace(
'''from linktools.ai.agent import (\n    AgentBindingSnapshot,\n    AgentDefinition,\n    AgentCatalog,\n)\n''',
'''from linktools.ai.agent import AgentBindingSnapshot\n''')
text = text.replace(
'from linktools.ai.runtime._factory import _restore_recovery_definitions\n',
'from linktools.ai.runtime._factory import _restore_recovery_bindings\n')
text = text.replace(
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        output_type_module="example.output",\n''',
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="c" * 64,\n        output_type_module="example.output",\n''', 1)

start = text.index('def test_catalog_uses_durable_semantics_for_restored_capability_instances() -> None:\n')
end = text.index('\n\nclass _SessionService:', start)
text = text[:start] + '''def test_agent_definition_no_longer_owns_output_binding() -> None:\n    from dataclasses import fields\n    from linktools.ai.agent import AgentDefinition\n\n    names = {field.name for field in fields(AgentDefinition)}\n    assert "digest" in names\n    assert "spec" in names\n    assert "output_binding" not in names\n    assert "binding_snapshot" not in names\n\n''' + text[end+2:]

start = text.index('@pytest.mark.asyncio\nasync def test_runtime_session_definition_reports_binding_mismatch() -> None:\n')
end = text.index('\n\n@pytest.mark.asyncio\nasync def test_runtime_existing_session_reports_binding_mismatch', start)
text = text[:start] + text[end+2:]
text = text.replace('async def test_runtime_existing_session_reports_binding_mismatch() -> None:', 'async def test_runtime_existing_session_reports_agent_mismatch() -> None:')

text = text.replace(
'''    async def _authorized(session_id: str, principal: object, action: object) -> object:\n        del session_id, principal, action\n        return SimpleNamespace(binding_digest="a" * 64)\n''',
'''    async def _authorized(session_id: str, principal: object, action: object) -> object:\n        del session_id, principal, action\n        return SimpleNamespace(agent_digest="a" * 64)\n''')
text = text.replace(
'''    await service.resume(\n        "a" * 64,\n        "session",\n''',
'''    await service.resume(\n        "a" * 64,\n        "b" * 64,\n        "session",\n''')

start = text.index('@pytest.mark.asyncio\nasync def test_recovery_handoff_schema_must_match_restored_definition() -> None:\n')
text = text[:start] + '''@pytest.mark.asyncio\nasync def test_recovery_handoff_schema_must_match_restored_binding() -> None:\n    binding_digest = "a" * 64\n    snapshot = SimpleNamespace(binding_digest=binding_digest)\n    recovery_input = SimpleNamespace(\n        binding_digest=binding_digest,\n        planning=False,\n        thinking=False,\n        binding=snapshot,\n    )\n    checkpoint = SimpleNamespace(\n        execution_id="execution",\n        state=RecoveryCheckpointState.HANDOFF,\n        input=recovery_input,\n        terminal_handoff=SimpleNamespace(\n            outcome=SimpleNamespace(\n                output=object(),\n                output_schema_id="wrong",\n                output_schema_revision=1,\n                output_schema_fingerprint="c" * 64,\n            )\n        ),\n    )\n\n    async def _list_recoverable_page(**kwargs: object) -> object:\n        del kwargs\n        return SimpleNamespace(items=(checkpoint,), next_cursor=None)\n\n    async def _get_execution(*args: object, **kwargs: object) -> None:\n        del args, kwargs\n        return None\n\n    state = SimpleNamespace(\n        recovery=SimpleNamespace(\n            checkpoints=SimpleNamespace(list_recoverable_page=_list_recoverable_page)\n        ),\n        execution=SimpleNamespace(\n            executions=SimpleNamespace(get=_get_execution),\n        ),\n    )\n    definition = SimpleNamespace(digest="d" * 64)\n    binding = SimpleNamespace(\n        digest=binding_digest,\n        definition=definition,\n        output_binding=SimpleNamespace(\n            schema_id="expected",\n            schema_revision=1,\n            schema_fingerprint="b" * 64,\n        ),\n    )\n    catalog = SimpleNamespace(\n        register_definition=lambda value: value,\n        register_binding=lambda value: value,\n    )\n    compiler = SimpleNamespace(restore=lambda value: binding)\n\n    with pytest.raises(AIError) as error:\n        await _restore_recovery_bindings(\n            catalog,\n            compiler,\n            state,\n            tenant_id="tenant",\n        )\n\n    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR\n'''

path.write_text(text, encoding='utf-8')
print('agent composition tests migrated to final identity model')
