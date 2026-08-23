from pathlib import Path

path = Path('tests/ai/test_recovery_integrity_validator.py')
text = path.read_text(encoding='utf-8')
text = text.replace(
'import pytest\n',
'import pytest\nfrom linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.agent._output import bind_output\n')
text = text.replace(
'from linktools.ai.runtime.state._store import (\n',
'from linktools.ai.spec import AgentSpec\nfrom linktools.ai.runtime.state._store import (\n')
insert = '''\n\ndef _binding() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("default", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n'''
marker = '\n\ndef _checkpoint(\n'
if marker not in text:
    raise RuntimeError('checkpoint marker missing')
text = text.replace(marker, insert + marker, 1)
old = '''        RecoveryExecutionInput(\n            user_prompt="prompt",\n            principal_id="owner",\n            principal_kind="user",\n            session_id=None,\n            memory_scope=None,\n            agent_id="default",\n            binding_digest="binding",\n            lineage_kind="run",\n            parent_execution_id=None,\n            root_execution_id=execution_id,\n            source_execution_id=None,\n            base_execution_id=None,\n            conversation_step_run_id=None,\n            idempotency=RecoveryIdempotencyInput("scope", "key", "digest"),\n        ),\n'''
new = '''        RecoveryExecutionInput(\n            user_prompt="prompt",\n            principal_id="owner",\n            principal_kind="user",\n            session_id=None,\n            memory_scope=None,\n            binding_digest="a" * 64,\n            lineage_kind="run",\n            parent_execution_id=None,\n            root_execution_id=execution_id,\n            source_execution_id=None,\n            base_execution_id=None,\n            conversation_step_run_id=None,\n            idempotency=RecoveryIdempotencyInput("scope", "key", "digest"),\n            planning=False,\n            thinking=False,\n            binding=_binding(),\n        ),\n'''
if text.count(old) != 1:
    raise RuntimeError('RecoveryExecutionInput fixture pattern mismatch')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('recovery integrity tests migrated to mandatory snapshot')
