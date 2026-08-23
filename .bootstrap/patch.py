from pathlib import Path

path = Path('tests/ai/test_history_projection_conformance.py')
text = path.read_text(encoding='utf-8')
if 'from linktools.ai.agent import AgentBindingSnapshot\n' not in text:
    text = text.replace('from linktools.ai.adapter import StepExecutionHistoryReader\n', 'from linktools.ai.adapter import StepExecutionHistoryReader\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
if 'from linktools.ai.spec import AgentSpec\n' not in text:
    text = text.replace('from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock\n', 'from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock\nfrom linktools.ai.spec import AgentSpec\n')
if 'def _binding() -> AgentBindingSnapshot:' not in text:
    marker = '\n\ndef _record(status: ExecutionStatus, sequence: int) -> ExecutionRecord:\n'
    helper = '''\n\ndef _binding() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("default", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
    if marker not in text:
        raise RuntimeError('history record marker missing')
    text = text.replace(marker, helper + marker, 1)
text = text.replace('binding_digest="binding",', 'binding_digest="a" * 64,')
record_end = '''        created_at=now,\n        updated_at=now,\n    )\n'''
record_new = '''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding(),\n    )\n'''
if 'binding=_binding(),' not in text:
    if text.count(record_end) < 1:
        raise RuntimeError('history record end missing')
    text = text.replace(record_end, record_new, 1)
text = text.replace('ExecutionTerminalSealPlan("execution", "binding", (), ())', 'ExecutionTerminalSealPlan("execution", "a" * 64, (), ())')
path.write_text(text, encoding='utf-8')
print('history exact binding fixture completed')
