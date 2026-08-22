#!/usr/bin/env python3
from pathlib import Path


def update(rel: str, fn) -> None:
    path = Path(rel)
    text = path.read_text(encoding="utf-8")
    new = fn(text)
    if new == text:
        raise RuntimeError(f"no change applied: {rel}")
    path.write_text(new, encoding="utf-8")


# Execution service resolves exact AgentBinding and persists its mandatory snapshot.
def patch_execution(text: str) -> str:
    text = text.replace(
        "from ..agent import AgentBindingSnapshot, AgentCompiler, AgentDefinition, AgentCatalog",
        "from ..agent import AgentBinding, AgentBindingSnapshot, AgentCatalog, AgentCompiler",
    )
    start = text.index("    def _definition(\n")
    end = text.index("    async def acquire_dependency_hold(\n", start)
    replacement = '''    def _binding(
        self,
        binding_digest: str,
        snapshot: "AgentBindingSnapshot | None" = None,
    ) -> AgentBinding:
        try:
            binding = self._catalog.binding(binding_digest)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE or snapshot is None:
                raise
            if snapshot.binding_digest != binding_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        if snapshot is not None and binding.snapshot != snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return binding

    def _validate_replayed_execution(
        self,
        execution: ExecutionRecord,
        binding: AgentBinding,
        request: ExecutionRequest,
    ) -> None:
        if (
            execution.binding_digest != binding.digest
            or execution.planning is not request.planning
            or execution.thinking is not request.thinking
            or execution.binding != binding.snapshot
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

'''
    text = text[:start] + replacement + text[end:]
    text = text.replace("        definition = self._definition(binding_digest)\n", "        binding = self._binding(binding_digest)\n", 1)
    text = text.replace(
        "            conversation_run_id = None if session.continuation is None else session.continuation.step_run_id\n            base_execution_id = None",
        "            if session.agent_digest != binding.definition.digest:\n                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)\n            conversation_run_id = None if session.continuation is None else session.continuation.step_run_id\n            base_execution_id = None",
        1,
    )
    text = text.replace("self._validate_replayed_execution(pending, definition, request)", "self._validate_replayed_execution(pending, binding, request)")
    text = text.replace("self._validate_replayed_execution(started, definition, request)", "self._validate_replayed_execution(started, binding, request)")
    text = text.replace("self._validate_replayed_execution(reservation.execution, definition, request)", "self._validate_replayed_execution(reservation.execution, binding, request)")
    text = text.replace("            binding=definition.binding_snapshot,", "            binding=binding.snapshot,", 1)
    text = text.replace("        definition = self._definition(previous.binding_digest, previous.binding)\n        if definition.digest != binding_digest:", "        binding = self._binding(previous.binding_digest, previous.binding)\n        if binding.digest != binding_digest:")
    return text

update("linktools-ai/src/linktools/ai/runtime/_execution.py", patch_execution)


# Persistence records have one current V1 shape: exact binding snapshot and modes are mandatory.
def patch_contracts(text: str) -> str:
    old = '''    planning: bool = False
    thinking: bool = False
    binding: AgentBindingSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise ValueError("execution modes must be boolean")
        if self.binding is None:
            if self.planning or self.thinking:
                raise ValueError("legacy execution without binding cannot enable modes")
        elif self.binding.binding_digest != self.binding_digest:
            raise ValueError("execution binding snapshot does not match binding digest")
'''
    new = '''    planning: bool
    thinking: bool
    binding: AgentBindingSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise ValueError("execution modes must be boolean")
        if not isinstance(self.binding, AgentBindingSnapshot) or self.binding.binding_digest != self.binding_digest:
            raise ValueError("execution binding snapshot does not match binding digest")
'''
    if old not in text:
        raise RuntimeError("ExecutionRecord legacy block not found")
    text = text.replace(old, new, 1)
    old = '''    memory_scope: str | None
    agent_id: str
    binding_digest: str
    lineage_kind: str
'''
    new = '''    memory_scope: str | None
    binding_digest: str
    lineage_kind: str
'''
    if old not in text:
        raise RuntimeError("Recovery agent_id block not found")
    text = text.replace(old, new, 1)
    old = '''    planning: bool = False
    thinking: bool = False
    binding: AgentBindingSnapshot | None = None

    def __post_init__(self) -> None:
        prompt = self.user_prompt
        if isinstance(prompt, str):
            object.__setattr__(self, "user_prompt", StoredPayload.inline_text(prompt))
        elif not isinstance(prompt, StoredPayload):
            raise ValueError("recovery prompt payload is invalid")
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise ValueError("recovery execution modes must be boolean")
        if self.binding is None:
            if self.planning or self.thinking:
                raise ValueError("legacy recovery input without binding cannot enable modes")
        elif (
            self.binding.binding_digest != self.binding_digest
            or self.binding.agent_spec.id != self.agent_id
        ):
            raise ValueError("recovery binding snapshot does not match execution identity")
'''
    new = '''    planning: bool
    thinking: bool
    binding: AgentBindingSnapshot

    def __post_init__(self) -> None:
        prompt = self.user_prompt
        if isinstance(prompt, str):
            object.__setattr__(self, "user_prompt", StoredPayload.inline_text(prompt))
        elif not isinstance(prompt, StoredPayload):
            raise ValueError("recovery prompt payload is invalid")
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise ValueError("recovery execution modes must be boolean")
        if not isinstance(self.binding, AgentBindingSnapshot) or self.binding.binding_digest != self.binding_digest:
            raise ValueError("recovery binding snapshot does not match execution identity")
'''
    if old not in text:
        raise RuntimeError("Recovery legacy block not found")
    return text.replace(old, new, 1)

update("linktools-ai/src/linktools/ai/runtime/state/_contracts.py", patch_contracts)


# Local backend consumes exact binding; projections use agent_digest, terminal identity stays binding_digest.
def patch_local(text: str) -> str:
    text = text.replace(
        "    MEMORY_TOOL_NAMES,\n    AgentDefinition,\n    AgentCatalog,",
        "    MEMORY_TOOL_NAMES,\n    AgentBinding,\n    AgentCatalog,",
    )
    old = '''        definition = self._catalog.definition(execution.binding_digest)
        if (
            request.planning is not execution.planning
            or request.thinking is not execution.thinking
            or (
                execution.binding is not None
                and execution.binding != definition.binding_snapshot
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
'''
    new = '''        binding = self._catalog.binding(execution.binding_digest)
        if (
            request.planning is not execution.planning
            or request.thinking is not execution.thinking
            or execution.binding != binding.snapshot
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
'''
    if old not in text:
        raise RuntimeError("local validate block not found")
    text = text.replace(old, new, 1)
    old = '''        if execution.status is not ExecutionStatus.PENDING_START or execution.binding is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        definition = self._catalog.definition(execution.binding_digest)
        if execution.binding != definition.binding_snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
'''
    new = '''        if execution.status is not ExecutionStatus.PENDING_START:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = self._catalog.binding(execution.binding_digest)
        if execution.binding != binding.snapshot:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
'''
    if old not in text:
        raise RuntimeError("local prepare block not found")
    text = text.replace(old, new, 1)
    text = text.replace("            agent_id=definition.spec.id,\n", "", 1)
    old = '''            definition = self._catalog.definition(current.binding_digest)
            if current.binding is not None and current.binding != definition.binding_snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
'''
    new = '''            binding = self._catalog.binding(current.binding_digest)
            if current.binding != binding.snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definition = binding.definition
'''
    if old not in text:
        raise RuntimeError("local worker binding block not found")
    text = text.replace(old, new, 1)
    # Model-context projection identity is the Agent, never the output binding.
    text = text.replace("binding_digest=current.binding_digest", "agent_digest=binding.definition.digest")
    text = text.replace("binding_digest=execution.binding_digest", "agent_digest=execution.binding.agent_digest")
    text = text.replace("                    definition,\n                    request.user_prompt,", "                    binding,\n                    request.user_prompt,", 1)
    text = text.replace("                definition,\n                result.output,", "                binding,\n                result.output,", 1)
    # Success/terminal output schema belongs to AgentBinding.
    text = text.replace("        definition: AgentDefinition,\n        output:", "        binding: AgentBinding,\n        output:")
    text = text.replace("            definition=definition,", "            binding=binding,")
    text = text.replace("        definition: AgentDefinition | None = None,", "        binding: AgentBinding | None = None,")
    text = text.replace("            if definition is None or output is None:", "            if binding is None or output is None:")
    text = text.replace("            schema_id = definition.output_binding.schema_id", "            schema_id = binding.output_binding.schema_id")
    text = text.replace("            schema_revision = definition.output_binding.schema_revision", "            schema_revision = binding.output_binding.schema_revision")
    text = text.replace("            schema_fingerprint = definition.output_schema_fingerprint", "            schema_fingerprint = binding.output_schema_fingerprint")
    return text

update("linktools-ai/src/linktools/ai/runtime/_local.py", patch_local)


# Restore durable exact bindings at startup; no legacy/null fallback.
def patch_factory(text: str) -> str:
    text = text.replace("dispatcher = SubagentDispatcher(catalog, execution)", "dispatcher = SubagentDispatcher(catalog, compiler, execution)", 1)
    text = text.replace("await _restore_recovery_definitions(catalog, compiler, state, tenant_id=tenant_id)", "await _restore_recovery_bindings(catalog, compiler, state, tenant_id=tenant_id)", 1)
    start = text.index("async def _restore_recovery_definitions(\n")
    end = text.index("\n\ndef _cursor_signer", start)
    replacement = '''async def _restore_recovery_bindings(
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    state: RuntimeState,
    *,
    tenant_id: str,
) -> None:
    cursor: str | None = None
    while True:
        page = await state.recovery.checkpoints.list_recoverable_page(
            tenant_id=tenant_id,
            cursor=cursor,
            limit=128,
        )
        for checkpoint in page.items:
            if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recovery_input = checkpoint.input
            execution = await state.execution.executions.get(
                checkpoint.execution_id,
                tenant_id=tenant_id,
            )
            if execution is not None and (
                execution.binding_digest != recovery_input.binding_digest
                or execution.planning is not recovery_input.planning
                or execution.thinking is not recovery_input.thinking
                or execution.binding != recovery_input.binding
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                binding = compiler.restore(recovery_input.binding)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    raise
                raise AIError(
                    ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                    safe_details={"execution_id": checkpoint.execution_id},
                ) from error
            if binding.digest != recovery_input.binding_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            catalog.register_definition(binding.definition)
            binding = catalog.register_binding(binding)
            handoff = checkpoint.terminal_handoff
            if handoff is not None and handoff.outcome.output is not None:
                output = binding.output_binding
                outcome = handoff.outcome
                if (
                    outcome.output_schema_id != output.schema_id
                    or outcome.output_schema_revision != output.schema_revision
                    or outcome.output_schema_fingerprint != output.schema_fingerprint
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if page.next_cursor is None:
            return
        cursor = page.next_cursor
'''
    return text[:start] + replacement + text[end:]

update("linktools-ai/src/linktools/ai/runtime/_factory.py", patch_factory)


# Subagents resolve a root AgentDefinition and bind the default output at dispatch time.
def patch_subagent(text: str) -> str:
    text = text.replace("from ..agent import AgentCatalog, SubagentDelegate", "from ..agent import AgentCatalog, AgentCompiler, SubagentDelegate")
    text = text.replace(
        "        catalog: AgentCatalog,\n        execution: DefaultExecutionService,",
        "        catalog: AgentCatalog,\n        compiler: AgentCompiler,\n        execution: DefaultExecutionService,",
        1,
    )
    text = text.replace("        self._catalog = catalog\n        self._execution = execution", "        self._catalog = catalog\n        self._compiler = compiler\n        self._execution = execution", 1)
    text = text.replace("            definition = self._catalog.subagent_definition(agent_id)", "            definition = self._catalog.root_definition(agent_id)", 1)
    marker = '''        idempotency_key = "subagent:" + canonical_sha256(
'''
    insert = '''        binding = self._catalog.register_binding(self._compiler.bind(definition))
'''
    text = text.replace(marker, insert + marker, 1)
    text = text.replace("            definition.digest,\n            request,", "            binding.digest,\n            request,", 1)
    return text

update("linktools-ai/src/linktools/ai/runtime/_subagent.py", patch_subagent)


# TaskNode has exactly one V1 form; restore exact binding from its snapshot.
def patch_planner(text: str) -> str:
    text = text.replace(
        '''_AGENT_TASK_FIELDS = frozenset(
    {
        "type",
        "version",
        "agent_id",
        "binding_digest",
        "binding",
        "user_prompt",
        "planning",
        "thinking",
    }
)
''',
        '''_AGENT_TASK_FIELDS = frozenset(
    {"type", "version", "binding", "user_prompt", "planning", "thinking"}
)
''',
        1,
    )
    start = text.index("        agent_id = payload[\"agent_id\"]")
    end = text.index("        if set(dependency_results) != set(node.dependencies):", start)
    replacement = '''        base_user_prompt = payload["user_prompt"]
        planning = payload["planning"]
        thinking = payload["thinking"]
        if (
            not isinstance(base_user_prompt, str)
            or not isinstance(planning, bool)
            or not isinstance(thinking, bool)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            snapshot = AgentBindingSnapshot.from_payload(payload["binding"])
            binding = self._catalog.register_binding(self._compiler.restore(snapshot))
        except AIError as error:
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"binding_digest": snapshot.binding_digest if "snapshot" in locals() else None},
            ) from error
        if binding.snapshot != snapshot or binding.digest != snapshot.binding_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        validate_agent_id(binding.definition.spec.id)
        validate_user_prompt(base_user_prompt)
'''
    text = text[:start] + replacement + text[end:]
    text = text.replace('                "agent_id": agent_id,\n                "binding_digest": binding_digest,\n', '                "binding_digest": binding.digest,\n', 1)
    text = text.replace("        return binding_digest, request", "        return binding.digest, request", 1)
    return text

update("linktools-ai/src/linktools/ai/runtime/_planner.py", patch_planner)


# Projection state commands derive the stable Agent identity from the mandatory snapshot.
def patch_commands(text: str) -> str:
    text = text.replace("binding_digest=commit.execution.binding_digest", "binding_digest=commit.execution.binding.agent_digest")
    return text

update("linktools-ai/src/linktools/ai/runtime/state/_commands.py", patch_commands)


# Update tests to the final public Asset/Workspace surfaces; do not restore private compatibility aliases.
def patch_asset_test(text: str) -> str:
    text = text.replace("    AssetTypeRegistry,\n", "")
    old = '''    registry = AssetTypeRegistry()
    for binding in builtin_asset_bindings():
        registry.register(binding)
    for binding in extra_bindings:
        registry.register(binding)
    return AssetRepository(store, registry.freeze())
'''
    new = '''    return AssetRepository(store, (*builtin_asset_bindings(), *extra_bindings))
'''
    if old not in text:
        raise RuntimeError("asset test helper not found")
    return text.replace(old, new, 1)

update("tests/ai/test_asset_repository.py", patch_asset_test)


def patch_workspace_test(text: str) -> str:
    text = text.replace("from linktools.ai.workspace import Workspace, open_workspace_runtime", "from linktools.ai.workspace import Workspace, build_workspace_assets, open_workspace_runtime")
    text = text.replace("from linktools.ai.workspace._factory import _build_asset_registry, _build_asset_repository\n", "")
    old = '''    snapshot = _build_asset_registry(())
    assets = await _build_asset_repository(
        Workspace.load(tmp_path),
        asset=None,
        snapshot=snapshot,
        path_adapter=None,
    )
'''
    new = '''    assets = await build_workspace_assets(Workspace.load(tmp_path))
'''
    if old not in text:
        raise RuntimeError("workspace private helper test block not found")
    text = text.replace(old, new, 1)
    text = text.replace('        created = await runtime.create_session("remember")\n        agent_created = await runtime.agent("default").create_session("remember-agent")', '        created = await runtime.agent("default").create_session("remember")\n        agent_created = await runtime.agent("default").create_session("remember-agent")', 1)
    text = text.replace('        await runtime.create_session("custom-tenant")', '        await runtime.agent("default").create_session("custom-tenant")', 1)
    return text

update("tests/ai/test_workspace_runtime_regressions.py", patch_workspace_test)

print("exact binding execution/recovery/task/subagent phase applied")
