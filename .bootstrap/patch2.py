from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

path = Path('tests/ai/test_local_execution_recovery.py')
text = path.read_text(encoding='utf-8')
if 'from linktools.ai.agent import AgentBindingSnapshot\n' not in text:
    text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
if 'from linktools.ai.spec import AgentSpec\n' not in text:
    text = text.replace('from linktools.ai.runtime.state import ExecutionRecord\n', 'from linktools.ai.runtime.state import ExecutionRecord\nfrom linktools.ai.spec import AgentSpec\n')
if 'def _binding_snapshot() -> AgentBindingSnapshot:' not in text:
    marker = '\n\nclass _Executions:\n'
    helper = '''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("default", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n\n\ndef _binding() -> object:\n    snapshot = _binding_snapshot()\n    definition = SimpleNamespace(\n        digest=snapshot.agent_digest,\n        spec=SimpleNamespace(id="default", allow_tools=()),\n    )\n    return SimpleNamespace(\n        digest=snapshot.binding_digest, snapshot=snapshot, definition=definition\n    )\n'''
    if marker not in text:
        raise RuntimeError('local helper marker missing')
    text = text.replace(marker, helper + marker, 1)
if 'binding=_binding_snapshot(),' not in text:
    text = once(text,
'''        created_at=now,\n        updated_at=now,\n    )\n''',
'''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n''',
'local execution binding')
if 'binding=lambda digest: binding,' not in text:
    text = once(text,
'''    backend._catalog = SimpleNamespace(\n        root_ids=("default",),\n        definition=lambda digest: SimpleNamespace(\n            digest=digest,\n            spec=SimpleNamespace(\n                id="default",\n                allow_tools=(),\n            ),\n        )\n    )\n''',
'''    binding = _binding()\n    backend._catalog = SimpleNamespace(\n        root_ids=("default",),\n        binding=lambda digest: binding,\n    )\n''',
'local fake catalog')
path.write_text(text, encoding='utf-8')

path = Path('tests/ai/test_runtime_storage_io_convergence.py')
text = path.read_text(encoding='utf-8')
if 'from linktools.ai.agent import AgentBindingSnapshot\n' not in text:
    text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
if 'from linktools.ai.spec import AgentSpec\n' not in text:
    text = text.replace('from linktools.ai.storage import (\n', 'from linktools.ai.spec import AgentSpec\nfrom linktools.ai.storage import (\n', 1)
if 'def _binding_snapshot() -> AgentBindingSnapshot:' not in text:
    marker = '\npytestmark = pytest.mark.asyncio\n'
    helper = '''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "default"),\n        agent_digest="d" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
    if marker not in text:
        raise RuntimeError('storage IO marker missing')
    text = text.replace(marker, marker + helper, 1)
if 'agent_id="agent",' in text:
    text = text.replace('            agent_id="agent",\n', '', 1)
old = '''            idempotency=RecoveryIdempotencyInput(\n                scope="execution.start",\n                idempotency_key_digest="b" * 64,\n                request_digest="c" * 64,\n            ),\n        ),\n'''
new = '''            idempotency=RecoveryIdempotencyInput(\n                scope="execution.start",\n                idempotency_key_digest="b" * 64,\n                request_digest="c" * 64,\n            ),\n            planning=False,\n            thinking=False,\n            binding=_binding_snapshot(),\n        ),\n'''
if 'binding=_binding_snapshot(),' not in text:
    text = once(text, old, new, 'SQL recovery binding')
path.write_text(text, encoding='utf-8')

path = Path('tests/ai/test_architecture.py')
text = path.read_text(encoding='utf-8')
if '        "build_workspace_assets",\n' not in text:
    text = once(text,
'''        "Workspace",\n        "WorkspacePolicy",\n        "open_workspace_runtime",\n        "trusted_workspace_principal",\n''',
'''        "Workspace",\n        "WorkspacePolicy",\n        "build_workspace_assets",\n        "open_workspace_runtime",\n        "trusted_workspace_principal",\n''',
'workspace public surface')
path.write_text(text, encoding='utf-8')
print('local/recovery/public surface fixtures migrated')
