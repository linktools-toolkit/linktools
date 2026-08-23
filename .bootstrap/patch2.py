from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

# Execution service fake catalog now exposes exact AgentBinding lookup.
path = Path('tests/ai/persistence/test_execution_start_service.py')
text = path.read_text(encoding='utf-8')
text = once(text,
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        output_type_module="builtins",\n''',
'''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="d" * 64,\n        output_type_module="builtins",\n''',
'execution start snapshot agent digest')
text = once(text,
'''class _DefinitionCatalog:\n    def definition(self, digest: str) -> object:\n        return SimpleNamespace(digest=digest, binding_snapshot=_binding(digest))\n''',
'''class _DefinitionCatalog:\n    def binding(self, digest: str) -> object:\n        return SimpleNamespace(digest=digest, snapshot=_binding(digest))\n''',
'execution start fake binding catalog')
path.write_text(text, encoding='utf-8')

# Filesystem mutation fixture gets a mandatory exact binding snapshot.
path = Path('tests/ai/test_filesystem_mutation_invariants.py')
text = path.read_text(encoding='utf-8')
text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
text = text.replace('from linktools.ai.runtime.state._contracts import ExecutionRecord\n', 'from linktools.ai.runtime.state._contracts import ExecutionRecord\nfrom linktools.ai.spec import AgentSpec\n')
marker = '\n\ndef _tool_record() -> ToolOperationRecord:\n'
helper = '''\n\ndef _binding() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
if marker not in text:
    raise RuntimeError('filesystem binding helper marker missing')
text = text.replace(marker, helper + marker, 1)
old = '''    execution = ExecutionRecord(\n        "execution", "tenant", None, "binding", None, "execution", None, None,\n        ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 0, 0, 0, None, {}, now, now,\n    )\n'''
new = '''    execution = ExecutionRecord(\n        "execution", "tenant", None, "a" * 64, None, "execution", None, None,\n        ExecutionLineageKind.RUN, ExecutionStatus.STARTED, 0, 0, 0, None, {}, now, now,\n        False, False, _binding(),\n    )\n'''
text = once(text, old, new, 'filesystem execution record')
path.write_text(text, encoding='utf-8')

# History projection uses one central ExecutionRecord fixture; migrate it once.
path = Path('tests/ai/test_history_projection_conformance.py')
text = path.read_text(encoding='utf-8')
text = text.replace('from linktools.ai.adapter import StepExecutionHistoryReader\n', 'from linktools.ai.adapter import StepExecutionHistoryReader\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
text = text.replace('from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock\n', 'from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock\nfrom linktools.ai.spec import AgentSpec\n')
marker = '\n\ndef _record(status: ExecutionStatus, sequence: int) -> ExecutionRecord:\n'
helper = '''\n\ndef _binding() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("default", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
if marker not in text:
    raise RuntimeError('history binding helper marker missing')
text = text.replace(marker, helper + marker, 1)
text = once(text,
'''        binding_digest="binding",\n''',
'''        binding_digest="a" * 64,\n''',
'history execution digest')
text = once(text,
'''        created_at=now,\n        updated_at=now,\n    )\n''',
'''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding(),\n    )\n''',
'history execution mandatory binding')
text = text.replace('binding_digest="binding",', 'binding_digest="a" * 64,')
text = text.replace('ExecutionTerminalSealPlan("execution", "binding", (), ())', 'ExecutionTerminalSealPlan("execution", "a" * 64, (), ())')
path.write_text(text, encoding='utf-8')

print('execution start/filesystem/history fixtures migrated to exact bindings')
