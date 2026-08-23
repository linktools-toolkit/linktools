from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one match in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


matrix = Path("tests/ai/test_error_contract_matrix.py")
replace_once(
    matrix,
    '''    async def run(\n        self,\n        prompt: str,\n        *,\n        session_id: str,\n        memory_scope: str,\n    ) -> ExecutionResult:\n        del prompt, session_id, memory_scope\n        return self._result\n''',
    '''    async def run(\n        self,\n        prompt: str,\n        *,\n        session_id: str,\n        memory_scope: str,\n        planning: bool,\n        thinking: bool,\n    ) -> ExecutionResult:\n        del prompt, session_id, memory_scope, planning, thinking\n        return self._result\n''',
)
replace_once(
    matrix,
    '''    def agent(self, *, planning: bool, thinking: bool) -> _FailedAgent:\n        del planning, thinking\n        return self._agent\n''',
    '''    def agent(self) -> _FailedAgent:\n        return self._agent\n''',
)

review = Path("tests/ai/test_error_review_regressions.py")
replace_once(
    review,
    "from linktools.ai.agent._capabilities import AgentRunScope\n",
    "from linktools.ai.agent import AgentBindingSnapshot\n"
    "from linktools.ai.agent._capabilities import AgentRunScope\n",
)
replace_once(
    review,
    "from linktools.ai.agent._executor import AgentExecutor\n",
    "from linktools.ai.agent._executor import AgentExecutor\n"
    "from linktools.ai.agent._output import bind_output\n",
)
replace_once(
    review,
    "from linktools.ai.runtime.state import ExecutionRecord\n",
    "from linktools.ai.runtime.state import ExecutionRecord\n"
    "from linktools.ai.spec import AgentSpec\n",
)
replace_once(
    review,
    "from pydantic_ai_harness.compaction import DeduplicateFileReads\n\n\n",
    '''from pydantic_ai_harness.compaction import DeduplicateFileReads\n\n\ndef _binding_snapshot() -> AgentBindingSnapshot:\n    output = bind_output()\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "default"),\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n        output_type_qualname=output.value_type.__qualname__,\n        output_schema_id=output.schema_id,\n        output_schema_revision=output.schema_revision,\n        output_schema_fingerprint=output.schema_fingerprint,\n        local_runtime_capability_descriptors=(),\n        binding_digest="a" * 64,\n    )\n\n\n''',
)
replace_once(
    review,
    '''    executor._execute = cancelled  # type: ignore[method-assign]\n    definition = SimpleNamespace(spec=SimpleNamespace(usage_limits=None))\n\n    with pytest.raises(asyncio.CancelledError):\n        await executor.execute(\n            definition,  # type: ignore[arg-type]\n''',
    '''    executor._execute = cancelled  # type: ignore[method-assign]\n    binding = SimpleNamespace(\n        definition=SimpleNamespace(spec=SimpleNamespace(usage_limits=None))\n    )\n\n    with pytest.raises(asyncio.CancelledError):\n        await executor.execute(\n            binding,  # type: ignore[arg-type]\n''',
)
replace_once(
    review,
    '''        created_at=now,\n        updated_at=now,\n    )\n''',
    '''        created_at=now,\n        updated_at=now,\n        planning=False,\n        thinking=False,\n        binding=_binding_snapshot(),\n    )\n''',
)
