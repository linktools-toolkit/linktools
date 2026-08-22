#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "linktools-ai" / "src"
TESTS = ROOT / "tests" / "ai"


def replace_span(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement + text[right:]


# Internal catalog naming is final and uniform.
for base in (SRC, TESTS):
    for path in base.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        updated = text.replace("AgentDefinitionCatalog", "AgentCatalog")
        if updated != text:
            path.write_text(updated, encoding="utf-8")

# AgentExecutor executes an exact AgentBinding; AgentDefinition remains the Agent-level view.
path = SRC / "linktools" / "ai" / "agent" / "_executor.py"
text = path.read_text(encoding="utf-8")
text = text.replace("from ._builder import build_pydantic_agent\n", "from ._binding import AgentBinding\nfrom ._builder import build_pydantic_agent\n")
text = text.replace(
    "        definition: AgentDefinition,\n        user_prompt: str,",
    "        binding: AgentBinding,\n        user_prompt: str,",
    1,
)
text = text.replace(
    "        if not isinstance(planning, bool) or not isinstance(thinking, bool):\n            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n        run_usage = RunUsage()",
    "        if not isinstance(binding, AgentBinding):\n            raise TypeError(\"binding must be AgentBinding\")\n        if not isinstance(planning, bool) or not isinstance(thinking, bool):\n            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n        definition = binding.definition\n        run_usage = RunUsage()",
    1,
)
text = text.replace("                definition,\n                user_prompt,", "                binding,\n                user_prompt,", 1)
text = text.replace(
    "        definition: AgentDefinition,\n        user_prompt: str,",
    "        binding: AgentBinding,\n        user_prompt: str,",
    1,
)
needle = "    ) -> AgentExecutionResult:\n        if not self._execution_root.is_dir():"
text = text.replace(
    needle,
    "    ) -> AgentExecutionResult:\n        definition = binding.definition\n        if not self._execution_root.is_dir():",
    1,
)
text = text.replace("        agent = build_pydantic_agent(definition, model=model)", "        agent = build_pydantic_agent(binding, model=model)", 1)
text = text.replace("        if not isinstance(output, definition.output_type):", "        if not isinstance(output, binding.output_type):", 1)
path.write_text(text, encoding="utf-8")

# AgentHandle contains only Agent identity; output and modes belong to each execution call.
path = SRC / "linktools" / "ai" / "runtime" / "_agent.py"
path.write_text('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-bound Agent execution handle."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..core import JsonValue, Principal, validate_user_prompt
from .service_api import (
    EvaluationHandle,
    ExecutionHandle,
    ExecutionResult,
    ExecutionStreamEvent,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    SessionView,
)

if TYPE_CHECKING:
    from ..task import TaskNode
    from ._runtime_service import Runtime


@dataclass(frozen=True, slots=True)
class AgentHandle:
    _runtime: "Runtime"
    agent_id: str
    _agent_digest: str

    async def start(
        self,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: bool = False,
        thinking: bool = False,
    ) -> ExecutionHandle:
        return await self._runtime._start_for_agent(
            self._agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )

    async def run(
        self,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: bool = False,
        thinking: bool = False,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        return await self._runtime._run_for_agent(
            self._agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            timeout_seconds=timeout_seconds,
        )

    def stream(
        self,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        planning: bool = False,
        thinking: bool = False,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        return self._runtime._stream_for_agent(
            self._agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> SessionView:
        return await self._runtime._create_session_for_agent(
            self._agent_digest,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
        )

    async def run_evaluation(
        self,
        request: RunEvaluationRequest,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> EvaluationHandle:
        return await self._runtime._run_evaluation_for_agent(
            self._agent_digest,
            request,
            output=output,
        )

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> ExecutionHandle:
        return await self._runtime._replay_evaluation_for_agent(
            self._agent_digest,
            snapshot_id,
            request,
            output=output,
        )

    def task(
        self,
        node_id: str,
        user_prompt: str,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: "type[BaseModel] | None" = None,
        planning: bool = False,
        thinking: bool = False,
    ) -> "TaskNode":
        from ..task import TaskNode

        validate_user_prompt(user_prompt)
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise TypeError("planning and thinking must be bool")
        binding = self._runtime._bind_agent(self._agent_digest, output=output)
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": "linktools.ai.agent",
                "version": 1,
                "binding": binding.snapshot.to_payload(),
                "user_prompt": user_prompt,
                "planning": planning,
                "thinking": thinking,
            },
            budget_cost=budget_cost,
        )


__all__ = ["AgentHandle"]
''', encoding="utf-8")

# Runtime public surface and binding flow.
path = SRC / "linktools" / "ai" / "runtime" / "_runtime_service.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from ..agent import (\n    AgentBindingSnapshot,\n    AgentCompiler,\n    AgentDefinition,\n    AgentCatalog,\n)",
    "from ..agent import (\n    AgentBinding,\n    AgentBindingSnapshot,\n    AgentCatalog,\n    AgentCompiler,\n    AgentDefinition,\n)",
)
start = text.index("_AGENT_TASK_LEGACY_V1_FIELDS")
end = text.index("\n\n\nclass _LocalRuntimeCoordinatorPort", start)
text = text[:start] + '_AGENT_TASK_V1_FIELDS = frozenset(\n    {"type", "version", "binding", "user_prompt", "planning", "thinking"}\n)' + text[end:]
text = text.replace(
    "    async def resume(\n        self,\n        binding_digest: str,\n        session_id: str,",
    "    async def resume(\n        self,\n        agent_digest: str,\n        binding_digest: str,\n        session_id: str,",
    1,
)
agent_block = '''    def agent(
        self,
        agent: "str | AgentSpec" = "default",
        *,
        capabilities: "Sequence[RuntimeCapability]" = (),
    ) -> AgentHandle:
        self._ensure_open()
        if any(not isinstance(capability, RuntimeCapability) for capability in capabilities):
            raise TypeError("capabilities must contain RuntimeCapability values")
        if isinstance(agent, str):
            validate_agent_id(agent)
            base = self._catalog.root_definition(agent)
            definition = (
                base
                if not capabilities
                else self._compiler.compile(base.spec, capabilities=capabilities)
            )
        elif isinstance(agent, AgentSpec):
            definition = self._compiler.compile(agent, capabilities=capabilities)
        else:
            raise TypeError("agent must be an Agent id or AgentSpec")
        definition = self._catalog.register_definition(definition)
        return AgentHandle(self, definition.spec.id, definition.digest)

'''
text = replace_span(text, "    def agent(\n", "    def _definition(\n", agent_block)
identity_block = '''    def _definition(self, agent_digest: str) -> AgentDefinition:
        self._ensure_open()
        return self._catalog.definition(agent_digest)

    def _bind_agent(
        self,
        agent_digest: str,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        self._ensure_open()
        definition = self._catalog.definition(agent_digest)
        return self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )

    def _restore_binding(self, snapshot: AgentBindingSnapshot) -> AgentBinding:
        self._ensure_open()
        try:
            current = self._catalog.binding(snapshot.binding_digest)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
        else:
            if current.snapshot != snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        restored = self._compiler.restore(snapshot)
        self._catalog.register_definition(restored.definition)
        return self._catalog.register_binding(restored)

'''
text = replace_span(text, "    def _definition(\n", "    async def _compile_agent(\n", identity_block)
execution_block = '''    async def _start_for_agent(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
        local_stream: bool = False,
    ) -> ExecutionHandle:
        self._ensure_open()
        principal = self._resolve_principal(principal)
        validate_user_prompt(user_prompt)
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        definition = self._catalog.definition(agent_digest)
        binding = self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            memory_scope=_validate_memory_scope(memory_scope),
            planning=planning,
            thinking=thinking,
        )
        if session_id is None:
            if local_stream and self._local_coordinator is not None:
                handle = await self._local_coordinator.run(binding.digest, request)
            else:
                handle = await self.execution.run(binding.digest, request)
        else:
            if not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            await self._ensure_session(definition, session_id, principal)
            resume_request = ResumeSessionRequest(
                principal=principal,
                user_prompt=user_prompt,
                idempotency_key=request.idempotency_key or "",
                memory_scope=request.memory_scope,
                planning=planning,
                thinking=thinking,
            )
            if local_stream and self._local_coordinator is not None:
                handle = await self._local_coordinator.resume(
                    definition.digest,
                    binding.digest,
                    session_id,
                    resume_request,
                )
            else:
                handle = await self.session.resume(
                    definition.digest,
                    binding.digest,
                    session_id,
                    resume_request,
                )
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s planning=%s thinking=%s",
            handle.execution_id,
            definition.spec.id,
            session_id,
            planning,
            thinking,
        )
        return handle

    async def _run_for_agent(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
        timeout_seconds: "float | None",
    ) -> ExecutionResult:
        principal = self._resolve_principal(principal)
        handle = await self._start_for_agent(
            agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )
        return await self.execution.wait(
            handle.execution_id,
            principal=principal,
            timeout_seconds=timeout_seconds,
        )

'''
text = replace_span(text, "    async def start(\n", "    async def cancel(\n", execution_block)
stream_eval_block = '''    def _stream_for_agent(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        principal = self._resolve_principal(principal)
        return self._stream(
            agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )

    async def _stream(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        handle = await self._start_for_agent(
            agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            local_stream=self._local_coordinator is not None,
        )
        stream = (
            self.event.stream(handle.execution_id, principal=principal)
            if self._local_coordinator is None
            else self._local_coordinator.stream(handle.execution_id, principal=principal)
        )
        async for event in stream:
            yield event

    async def _create_session_for_agent(
        self,
        agent_digest: str,
        session_id: str,
        *,
        principal: "Principal | None",
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
    ) -> SessionView:
        principal = self._resolve_principal(principal)
        definition = self._catalog.definition(agent_digest)
        values = dict(metadata or {})
        if any(key.startswith("linktools.ai.") for key in values):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values["linktools.ai.agent_id"] = definition.spec.id
        return await self.session.create(
            definition.digest,
            CreateSessionRequest(
                principal,
                session_id,
                secrets.token_urlsafe(32),
                cwd,
                values,
            ),
        )

    async def _run_evaluation_for_agent(
        self,
        agent_digest: str,
        request: RunEvaluationRequest,
        *,
        output: "type[BaseModel] | None",
    ) -> EvaluationHandle:
        binding = self._bind_agent(agent_digest, output=output)
        return await self.evaluation.run(
            binding.digest,
            binding.output_schema_fingerprint,
            request,
        )

    async def _replay_evaluation_for_agent(
        self,
        agent_digest: str,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        output: "type[BaseModel] | None",
    ) -> ExecutionHandle:
        binding = self._bind_agent(agent_digest, output=output)
        return await self.evaluation.replay(binding.digest, snapshot_id, request)

'''
text = replace_span(text, "    def stream(\n", "    async def run_graph(\n", stream_eval_block)
admit_block = '''    async def _admit_graph(
        self,
        graph: TaskGraph,
        *,
        principal: "Principal | None",
        idempotency_key: str,
        limits: "TaskGraphLimits | None",
    ) -> TaskGraphRequest:
        self._ensure_open()
        principal = self._resolve_principal(principal)
        selected_limits = limits or TaskGraphLimits()
        validate_idempotency_key(idempotency_key)
        graph.validate_limits(selected_limits)
        admitted_nodes: list[TaskNode] = []
        agent_ids: set[str] = set()
        for node in graph.nodes:
            payload = node.input
            if frozenset(payload) != _AGENT_TASK_V1_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if payload.get("type") != "linktools.ai.agent" or payload.get("version") != 1:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            user_prompt = payload.get("user_prompt")
            planning = payload.get("planning")
            thinking = payload.get("thinking")
            if (
                not isinstance(user_prompt, str)
                or not isinstance(planning, bool)
                or not isinstance(thinking, bool)
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                snapshot = AgentBindingSnapshot.from_payload(payload.get("binding"))
                binding = self._restore_binding(snapshot)
            except AIError as error:
                if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                    raise
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            validate_user_prompt(user_prompt)
            agent_ids.add(binding.definition.spec.id)
            admitted_nodes.append(
                TaskNode(
                    node.node_id,
                    node.dependencies,
                    input={
                        "type": "linktools.ai.agent",
                        "version": 1,
                        "binding": binding.snapshot.to_payload(),
                        "user_prompt": user_prompt,
                        "planning": planning,
                        "thinking": thinking,
                    },
                    budget_cost=node.budget_cost,
                )
            )
        admitted = TaskGraph(graph.graph_id, tuple(admitted_nodes))
        _logger.info(
            "agent task graph admitted: graph=%s tenant=%s agents=%s nodes=%s",
            graph.graph_id,
            principal.tenant_id,
            tuple(sorted(agent_ids)),
            len(admitted.nodes),
        )
        return TaskGraphRequest(admitted, principal, idempotency_key, selected_limits)

'''
text = replace_span(text, "    async def _admit_graph(\n", "    async def _ensure_session(\n", admit_block)
session_block = '''    async def _ensure_session(
        self,
        definition: AgentDefinition,
        session_id: str,
        principal: Principal,
    ) -> None:
        try:
            session = await self.session.get(session_id, principal=principal)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            try:
                await self.session.create(
                    definition.digest,
                    CreateSessionRequest(
                        principal,
                        session_id,
                        secrets.token_urlsafe(32),
                        None,
                        {"linktools.ai.agent_id": definition.spec.id},
                    ),
                )
                return
            except AIError as create_error:
                if create_error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
            session = await self.session.get(session_id, principal=principal)
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.agent_digest != definition.digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

'''
text = replace_span(text, "    async def _ensure_session(\n", "    def _ensure_open(\n", session_block)
path.write_text(text, encoding="utf-8")

print("phase-a applied")
