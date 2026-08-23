from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

# Local backend tests use one exact binding shared by the execution and fake catalog.
path = Path('tests/ai/test_local_execution_recovery.py')
text = path.read_text(encoding='utf-8')
text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
text = text.replace('from linktools.ai.runtime.state import ExecutionRecord\n', 'from linktools.ai.runtime.state import ExecutionRecord\nfrom linktools.ai.spec import AgentSpec\n')
marker = '\n\nclass _Executions:\n'
helper = '''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("default", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n\n\ndef _binding() -> object:\n    snapshot = _binding_snapshot()\n    definition = SimpleNamespace(\n        digest=snapshot.agent_digest,\n        spec=SimpleNamespace(id="default", allow_tools=()),\n    )\n    return SimpleNamespace(\n        digest=snapshot.binding_digest,\n        snapshot=snapshot,\n        definition=definition,\n    )\n'''
if marker not in text:
    raise RuntimeError('local binding helper marker missing')
text = text.replace(marker, helper + marker, 1)
text = once(text,
'''        created_at=now,\n        updated_at=now,\n    )\n''',
'''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n''',
'local execution mandatory binding')
old_catalog = '''    backend._catalog = SimpleNamespace(\n        root_ids=("default",),\n        definition=lambda digest: SimpleNamespace(\n            digest=digest,\n            spec=SimpleNamespace(\n                id="default",\n                allow_tools=(),\n            ),\n        )\n    )\n'''
new_catalog = '''    binding = _binding()\n    backend._catalog = SimpleNamespace(\n        root_ids=("default",),\n        binding=lambda digest: binding,\n    )\n'''
text = once(text, old_catalog, new_catalog, 'local fake binding catalog')
path.write_text(text, encoding='utf-8')

# SQL recovery checkpoint fixture follows mandatory exact binding snapshot.
path = Path('tests/ai/test_runtime_storage_io_convergence.py')
text = path.read_text(encoding='utf-8')
text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
text = text.replace('from linktools.ai.storage import (\n', 'from linktools.ai.spec import AgentSpec\nfrom linktools.ai.storage import (\n', 1)
marker = '\npytestmark = pytest.mark.asyncio\n'
helper = '''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "default"),\n        agent_digest="d" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
if marker not in text:
    raise RuntimeError('storage IO pytest marker missing')
text = text.replace(marker, marker + helper, 1)
old = '''        input=RecoveryExecutionInput(\n            user_prompt="prompt",\n            principal_id="principal",\n            principal_kind="local_trusted",\n            session_id=None,\n            memory_scope=None,\n            agent_id="agent",\n            binding_digest="a" * 64,\n            lineage_kind="RUN",\n            parent_execution_id=None,\n            root_execution_id="execution",\n            source_execution_id=None,\n            base_execution_id=None,\n            conversation_step_run_id=None,\n            idempotency=RecoveryIdempotencyInput(\n                scope="execution.start",\n                idempotency_key_digest="b" * 64,\n                request_digest="c" * 64,\n            ),\n        ),\n'''
new = '''        input=RecoveryExecutionInput(\n            user_prompt="prompt",\n            principal_id="principal",\n            principal_kind="local_trusted",\n            session_id=None,\n            memory_scope=None,\n            binding_digest="a" * 64,\n            lineage_kind="RUN",\n            parent_execution_id=None,\n            root_execution_id="execution",\n            source_execution_id=None,\n            base_execution_id=None,\n            conversation_step_run_id=None,\n            idempotency=RecoveryIdempotencyInput(\n                scope="execution.start",\n                idempotency_key_digest="b" * 64,\n                request_digest="c" * 64,\n            ),\n            planning=False,\n            thinking=False,\n            binding=_binding_snapshot(),\n        ),\n'''
text = once(text, old, new, 'SQL recovery exact binding fixture')
path.write_text(text, encoding='utf-8')

# Public workspace surface now intentionally includes the public Asset builder.
path = Path('tests/ai/test_architecture.py')
text = path.read_text(encoding='utf-8')
old = '''        "Workspace",\n        "WorkspacePolicy",\n        "open_workspace_runtime",\n        "trusted_workspace_principal",\n'''
new = '''        "Workspace",\n        "WorkspacePolicy",\n        "build_workspace_assets",\n        "open_workspace_runtime",\n        "trusted_workspace_principal",\n'''
text = once(text, old, new, 'workspace public surface expectation')
path.write_text(text, encoding='utf-8')

print('local/recovery/public surface fixtures migrated')
