from pathlib import Path

root = Path('.')

# Migration owns shape conversion only. Output type restoration belongs to AgentCompiler.
path = root / 'linktools-ai/src/linktools/ai/runtime/state/_migration.py'
text = path.read_text(encoding='utf-8')
text = text.replace('import importlib\n', '', 1)
text = text.replace('from pydantic import BaseModel\n\n', '', 1)
text = text.replace(
    'from ...core import JsonValue\n',
    'from ...core import JsonValue, canonical_sha256\n',
    1,
)
text = text.replace(
    'from ._root import RuntimeState\n',
    'from ._root import RuntimeState\nfrom ._steps import StateStepArchive\n',
    1,
)
old = '''    for domain in (\n        RuntimeDomain.CONVERSATION,\n        RuntimeDomain.EXECUTION,\n        RuntimeDomain.RECOVERY,\n    ):\n        history = state.steps.read_store(domain).transcript_repository\n        for record in await _records(history, "context_projection"):\n            data = _migrate_projection_data(record, legacy_agents)\n            if data is not None:\n                await _replace_data(history._store, record, data)\n                migrated += 1\n'''
new = '''    for domain in (\n        RuntimeDomain.CONVERSATION,\n        RuntimeDomain.EXECUTION,\n        RuntimeDomain.RECOVERY,\n    ):\n        archive = state.steps.read_store(domain)\n        if not isinstance(archive, StateStepArchive):\n            continue\n        history = archive.transcript_repository\n        for record in await _records(history, "context_projection"):\n            data = _migrate_projection_data(record, legacy_agents)\n            if data is not None:\n                await _replace_data(history._store, record, data)\n                migrated += 1\n'''
if text.count(old) != 1:
    raise SystemExit(f'projection archive anchor count={text.count(old)}')
text = text.replace(old, new, 1)

start = text.index('def _migrate_legacy_binding(')
end = text.index('\ndef _legacy_agent_spec(', start)
replacement = '''def _migrate_legacy_binding(\n    value: object,\n    compiler: AgentCompiler,\n) -> tuple[str, AgentBinding]:\n    if not isinstance(value, Mapping) or frozenset(value) != _LEGACY_BINDING_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    if value.get("version") != 1 or isinstance(value.get("version"), bool):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    legacy_digest = _digest(value.get("binding_digest"))\n    spec = _legacy_agent_spec(value.get("agent_spec"))\n    descriptors = value.get("local_runtime_capability_descriptors")\n    if not isinstance(descriptors, list):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    try:\n        capabilities = tuple(\n            RuntimeCapability.restore(_json_mapping(descriptor))\n            for descriptor in descriptors\n        )\n        definition = compiler.compile(spec, capabilities=capabilities)\n        module_name = _string(value.get("output_type_module"))\n        qualname = _string(value.get("output_type_qualname"))\n        schema_id = _string(value.get("output_schema_id"))\n        schema_revision = _positive_int(value.get("output_schema_revision"))\n        schema_fingerprint = _digest(value.get("output_schema_fingerprint"))\n        output_binding_fingerprint = canonical_sha256(\n            {\n                "schema_id": schema_id,\n                "schema_revision": schema_revision,\n                "schema_fingerprint": schema_fingerprint,\n                "module": module_name,\n                "qualname": qualname,\n            }\n        )\n        binding_digest = canonical_sha256(\n            {\n                "version": 1,\n                "agent_digest": definition.digest,\n                "output_binding_fingerprint": output_binding_fingerprint,\n            }\n        )\n        snapshot = AgentBindingSnapshot(\n            version=1,\n            agent_spec=definition.spec,\n            agent_digest=definition.digest,\n            output_type_module=module_name,\n            output_type_qualname=qualname,\n            output_schema_id=schema_id,\n            output_schema_revision=schema_revision,\n            output_schema_fingerprint=schema_fingerprint,\n            local_runtime_capability_descriptors=(\n                definition.local_runtime_capability_descriptors\n            ),\n            binding_digest=binding_digest,\n        )\n        binding = compiler.restore(snapshot)\n    except AIError as error:\n        if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:\n            raise\n        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error\n    if binding.snapshot != snapshot:\n        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)\n    return legacy_digest, binding\n\n'''
text = text[:start] + replacement + text[end + 1:]

start = text.find('def _output_type(')
if start != -1:
    end = text.index('\ndef _nested_dataclass_fields(', start)
    text = text[:start] + text[end + 1:]
path.write_text(text, encoding='utf-8')

# Sibling packages import through capability's package boundary. Keep the helper
# available there without expanding the documented __all__ surface.
path = root / 'linktools-ai/src/linktools/ai/capability/__init__.py'
text = path.read_text(encoding='utf-8')
old = 'from ._mcp import MCPCapabilityProvider, MCPRuntime, mcp_server_selector\n'
new = 'from ._mcp import (\n    MCPCapabilityProvider,\n    MCPRuntime,\n    mcp_server_selector as mcp_server_selector,\n)\n'
if text.count(old) != 1:
    raise SystemExit(f'MCP re-export anchor count={text.count(old)}')
text = text.replace(old, new, 1)
text = text.replace('    "mcp_server_selector",\n', '', 1)
path.write_text(text, encoding='utf-8')

# Projection persistence exists only on durable StateStepArchive. Exercise the
# projection migration with the filesystem backend rather than an in-memory archive.
path = root / 'tests/ai/test_runtime_state_identity_migration.py'
text = path.read_text(encoding='utf-8')
old = '''@pytest.mark.asyncio\nasync def test_migration_preserves_historical_projection_digest() -> None:\n    state = RuntimeState.in_memory()\n    await state.initialize(namespace="legacy-projection", tenant_id="tenant")\n'''
new = '''@pytest.mark.asyncio\nasync def test_migration_preserves_historical_projection_digest(tmp_path) -> None:\n    state = RuntimeState.filesystem(tmp_path / "runtime")\n    await state.initialize(namespace="legacy-projection", tenant_id="tenant")\n'''
if text.count(old) != 1:
    raise SystemExit(f'projection test anchor count={text.count(old)}')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')

print('final migration architecture closure applied')
