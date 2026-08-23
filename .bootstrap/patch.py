from pathlib import Path


def add_binding_support(path_value: str, agent_id: str = "agent") -> str:
    path = Path(path_value)
    text = path.read_text(encoding='utf-8')
    if 'from linktools.ai.agent import AgentBindingSnapshot\n' not in text:
        text = text.replace('import pytest\n', 'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n', 1)
    if 'from linktools.ai.spec import AgentSpec\n' not in text:
        anchor = 'from linktools.ai.runtime'
        index = text.index(anchor)
        text = text[:index] + 'from linktools.ai.spec import AgentSpec\n' + text[index:]
    if 'def _binding_snapshot() -> AgentBindingSnapshot:' not in text:
        marker_candidates = ('\n\ndef _execution(', '\n\n@pytest.mark.asyncio')
        marker = next((candidate for candidate in marker_candidates if candidate in text), None)
        if marker is None:
            raise RuntimeError(f'{path_value}: fixture insertion marker missing')
        helper = f'''\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("{agent_id}", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
        text = text.replace(marker, helper + marker, 1)
    path.write_text(text, encoding='utf-8')
    return text

# One helper covers all stream-order execution fixtures.
path = Path('tests/ai/test_runtime_stream_order.py')
text = add_binding_support(str(path), 'default')
text = text.replace('        binding_digest="binding",\n', '        binding_digest="a" * 64,\n', 1)
old = '''        created_at=now,\n        updated_at=now,\n    )\n'''
new = '''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n'''
if text.count(old) < 1:
    raise RuntimeError('runtime stream execution helper end missing')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')

# Memory terminal has a single positional ExecutionRecord construction.
path = Path('tests/ai/test_memory_terminal.py')
text = add_binding_support(str(path), 'default')
old = '''            "binding",\n            None,\n            "execution",\n            None,\n            None,\n            ExecutionLineageKind.RUN,\n            ExecutionStatus.STARTED,\n            1,\n            1,\n            1,\n            None,\n            {},\n            now,\n            now,\n        )\n'''
new = '''            "a" * 64,\n            None,\n            "execution",\n            None,\n            None,\n            ExecutionLineageKind.RUN,\n            ExecutionStatus.STARTED,\n            1,\n            1,\n            1,\n            None,\n            {},\n            now,\n            now,\n            False,\n            False,\n            _binding_snapshot(),\n        )\n'''
if text.count(old) != 1:
    raise RuntimeError(f'memory terminal record pattern: {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

# Step IO uses one helper.
path = Path('tests/ai/test_step_io_final_fix.py')
text = add_binding_support(str(path), 'agent')
text = text.replace('        binding_digest="binding",\n', '        binding_digest="a" * 64,\n', 1)
if 'binding=_binding_snapshot(),' not in text:
    if text.count(old := '''        created_at=now,\n        updated_at=now,\n    )\n''') < 1:
        raise RuntimeError('step IO execution helper end missing')
    text = text.replace(old, '''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n''', 1)
path.write_text(text, encoding='utf-8')

print('remaining execution fixtures migrated')
