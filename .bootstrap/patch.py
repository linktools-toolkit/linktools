from pathlib import Path

ROOT = Path('.')

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')

def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding='utf-8')

def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

path = 'linktools-ai/src/linktools/ai/runtime/_local.py'
text = read(path)
text = once(
    text,
    '        definition: AgentDefinition | None,\n        run_id: str | None,',
    '        binding: AgentBinding | None,\n        run_id: str | None,',
    'same-group terminal binding parameter',
)
text = once(
    text,
    '''                terminal_plan = await self._step_lifecycle.prepare_execution_terminal_seal(\n                    execution_id=current.execution_id,\n                    run_ids=candidate_run_ids,\n                    agent_digest=binding.definition.digest,\n                )\n''',
    '''                terminal_plan = await self._step_lifecycle.prepare_execution_terminal_seal(\n                    execution_id=current.execution_id,\n                    run_ids=candidate_run_ids,\n                    binding_digest=current.binding_digest,\n                )\n''',
    'execution terminal seal exact binding',
)
write(path, text)

path = 'tests/ai/test_asset_repository.py'
text = read(path)
text = once(text,
    '''    registry = AssetTypeRegistry()\n    registry.register(_binding())\n    return store, AssetRepository(store, registry.freeze())\n''',
    '''    return store, AssetRepository(store, (_binding(),))\n''',
    'asset repo helper')
text = once(text,
    '''    registry = AssetTypeRegistry()\n    registry.register(_multi_directory_binding())\n    repository = AssetRepository(store, registry.freeze())\n''',
    '''    repository = AssetRepository(store, (_multi_directory_binding(),))\n''',
    'multi directory repository')
common = '''    registry = AssetTypeRegistry()\n    registry.register(_binding())\n    repository = AssetRepository(store, registry.freeze())\n'''
for label in ('counting repository', 'race repository', 'recovery race repository', 'readonly repository'):
    if common not in text:
        raise RuntimeError(f'{label}: pattern missing')
    text = text.replace(common, '''    repository = AssetRepository(store, (_binding(),))\n''', 1)

start = text.index('def test_registry_freeze_rejects_overlapping_single_file_layouts() -> None:\n')
end = text.index('\n\n@pytest.mark.parametrize("layout"', start)
text = text[:start] + '''def test_repository_rejects_overlapping_single_file_layouts() -> None:\n    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))\n    with pytest.raises(AIError) as error:\n        AssetRepository(\n            store,\n            (\n                AssetTypeBinding(\n                    "sample",\n                    _Value,\n                    (\n                        AssetVariantBinding("short", SingleFileLayout(".md"), _Codec(), "short", 1),\n                        AssetVariantBinding("long", SingleFileLayout(".agent.md"), _Codec(), "long", 1),\n                    ),\n                    "short",\n                ),\n            ),\n        )\n    assert error.value.code is ErrorCode.ASSET_CODEC_CONFLICT\n''' + text[end:]

start = text.index('def test_registry_accepts_concrete_protocol_subclass_value_type() -> None:\n')
end = text.index('\n\n@pytest.mark.asyncio\nasync def test_concrete_protocol_subclass_keeps_typed_runtime_exact_type', start)
text = text[:start] + '''def test_repository_accepts_concrete_protocol_subclass_value_type() -> None:\n    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))\n    repository = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "concrete-protocol",\n                _ConcreteProtocolValue,\n                (AssetVariantBinding("file", SingleFileLayout(""), _ConcreteProtocolCodec(), "concrete-protocol", 1),),\n                "file",\n            ),\n        ),\n    )\n    assert repository.kinds == ("concrete-protocol",)\n''' + text[end:]

text = once(text,
    '''    registry = AssetTypeRegistry()\n    registry.register(\n        AssetTypeBinding(\n            "concrete-protocol-runtime",\n            _ConcreteProtocolValue,\n            (\n                AssetVariantBinding(\n                    "file",\n                    SingleFileLayout(""),\n                    _ConcreteProtocolCodec(),\n                    "concrete-protocol-runtime",\n                    1,\n                ),\n            ),\n            "file",\n        )\n    )\n    repository = AssetRepository(store, registry.freeze())\n''',
    '''    repository = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "concrete-protocol-runtime",\n                _ConcreteProtocolValue,\n                (\n                    AssetVariantBinding(\n                        "file",\n                        SingleFileLayout(""),\n                        _ConcreteProtocolCodec(),\n                        "concrete-protocol-runtime",\n                        1,\n                    ),\n                ),\n                "file",\n            ),\n        ),\n    )\n''',
    'concrete protocol runtime repository')
text = once(text,
    '''    registry = AssetTypeRegistry()\n    registry.register(\n        AssetTypeBinding(\n            "wrong",\n            _Value,\n            (AssetVariantBinding("wrong", SingleFileLayout(""), _WrongTypeCodec(), "wrong", 1),),\n            "wrong",\n        )\n    )\n    registry.register(\n        AssetTypeBinding(\n            "broken",\n            _Value,\n            (AssetVariantBinding("broken", SingleFileLayout(""), _BrokenCodec(), "broken", 1),),\n            "broken",\n        )\n    )\n    extra = AssetRepository(store, registry.freeze())\n''',
    '''    extra = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "wrong",\n                _Value,\n                (AssetVariantBinding("wrong", SingleFileLayout(""), _WrongTypeCodec(), "wrong", 1),),\n                "wrong",\n            ),\n            AssetTypeBinding(\n                "broken",\n                _Value,\n                (AssetVariantBinding("broken", SingleFileLayout(""), _BrokenCodec(), "broken", 1),),\n                "broken",\n            ),\n        ),\n    )\n''',
    'codec error repositories')
if 'AssetTypeRegistry' in text:
    raise RuntimeError('AssetTypeRegistry residue remains in public repository tests')
write(path, text)

path = 'tests/ai/test_workspace_runtime_regressions.py'
text = read(path)
text = text.replace('created = await runtime.create_session("remember")', 'created = await runtime.agent().create_session("remember")')
text = text.replace('await runtime.create_session("custom-tenant")', 'await runtime.agent().create_session("custom-tenant")')
write(path, text)

path = 'tests/ai/test_contracts.py'
text = read(path)
text = once(text,
    '''        version=1,\n        agent_spec=spec,\n        output_type_module=output.value_type.__module__,\n''',
    '''        version=1,\n        agent_spec=spec,\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n''',
    'contract snapshot agent digest')
text = once(text,
    '''    recovery_input = RecoveryExecutionInput(\n        user_prompt="prompt",\n        principal_id="principal",\n        principal_kind="user",\n        session_id=None,\n        memory_scope=None,\n        agent_id="default",\n        binding_digest="binding",\n        lineage_kind="RUN",\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        conversation_step_run_id=None,\n        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),\n    )\n''',
    '''    snapshot = _binding_snapshot(digest="c" * 64)\n    recovery_input = RecoveryExecutionInput(\n        user_prompt="prompt",\n        principal_id="principal",\n        principal_kind="user",\n        session_id=None,\n        memory_scope=None,\n        binding_digest=snapshot.binding_digest,\n        lineage_kind="RUN",\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        conversation_step_run_id=None,\n        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),\n        planning=False,\n        thinking=False,\n        binding=snapshot,\n    )\n''',
    'mandatory recovery binding contract test')
write(path, text)

print('exact binding and public asset test fixes applied')
