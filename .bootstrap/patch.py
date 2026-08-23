from pathlib import Path

# Session admission fake catalog uses exact AgentBinding lookup.
path = Path('tests/ai/test_session_admission_conformance.py')
text = path.read_text(encoding='utf-8')
if '        agent_digest="d" * 64,\n' not in text:
    old = '''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        output_type_module="builtins",\n'''
    new = '''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="d" * 64,\n        output_type_module="builtins",\n'''
    if text.count(old) != 1:
        raise RuntimeError('session admission binding snapshot pattern mismatch')
    text = text.replace(old, new, 1)
old = '''class _DefinitionCatalog:\n    def definition(self, digest: str) -> object:\n        return SimpleNamespace(digest=digest, binding_snapshot=_binding(digest))\n'''
new = '''class _DefinitionCatalog:\n    def binding(self, digest: str) -> object:\n        return SimpleNamespace(digest=digest, snapshot=_binding(digest))\n'''
if old in text:
    text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')

# Session fork validates Agent identity, not execution binding identity.
path = Path('tests/ai/test_session_history_projection.py')
text = path.read_text(encoding='utf-8')
old = '''        await service.fork(\n            "binding",\n            "session",\n'''
new = '''        await service.fork(\n            "a" * 64,\n            "session",\n'''
if text.count(old) != 1:
    raise RuntimeError(f'session history fork identity pattern: {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

# One snapshot fixture drives all Temporal TaskGraph conformance tests.
path = Path('tests/ai/test_temporal_taskgraph_conformance.py')
text = path.read_text(encoding='utf-8')
old = '''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        output_type_module="builtins",\n'''
new = '''        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="c" * 64,\n        output_type_module="builtins",\n'''
if text.count(old) != 1:
    raise RuntimeError(f'Temporal snapshot pattern: {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

# Final V1 closure persistence fixture also carries the mandatory exact binding.
path = Path('tests/ai/test_v1_final_closure.py')
text = path.read_text(encoding='utf-8')
if 'from linktools.ai.agent import AgentBindingSnapshot\n' not in text:
    text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n', 1)
if 'from linktools.ai.spec import AgentSpec\n' not in text:
    text = text.replace('from linktools.ai.task import TaskNode\n', 'from linktools.ai.spec import AgentSpec\nfrom linktools.ai.task import TaskNode\n', 1)
if 'def _binding_snapshot() -> AgentBindingSnapshot:' not in text:
    marker = '\n\ndef _assert_integrity('
    helper = '''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
    if marker not in text:
        raise RuntimeError('V1 closure helper marker missing')
    text = text.replace(marker, helper + marker, 1)
# Only the stale durable-authority execution fixture uses this literal binding field.
text = text.replace('        binding_digest="binding",\n', '        binding_digest="a" * 64,\n', 1)
old = '''        created_at=now,\n        updated_at=now,\n    )\n'''
new = '''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n'''
# Restrict insertion to the late ExecutionRecord by using the last occurrence.
if 'binding=_binding_snapshot(),' not in text:
    index = text.rfind(old)
    if index < 0:
        raise RuntimeError('V1 closure ExecutionRecord end missing')
    text = text[:index] + new + text[index + len(old):]
path.write_text(text, encoding='utf-8')

print('remaining exact identity fixtures closed')
