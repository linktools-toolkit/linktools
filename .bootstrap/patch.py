#!/usr/bin/env python3
import subprocess
from pathlib import Path

BASE = "a4ded79a152ccfbc635503ee7b7bd394bd50970e"
TARGET = "linktools-ai/src/linktools/ai/runtime/_runtime_service.py"
text = subprocess.check_output(("git", "show", f"{BASE}:{TARGET}"), text=True)


def replace_span(start: str, end: str, replacement: str) -> None:
    global text
    left = text.index(start)
    right = text.index(end, left)
    text = text[:left] + replacement + text[right:]


text = text.replace("from typing import Protocol, TypeGuard", "from typing import Protocol")
text = text.replace(
    "from ..agent import (\n    AgentBindingSnapshot,\n    AgentCompiler,\n    AgentDefinition,\n    AgentDefinitionCatalog,\n)",
    "from ..agent import (\n    AgentBinding,\n    AgentBindingSnapshot,\n    AgentCatalog,\n    AgentCompiler,\n    AgentDefinition,\n)",
)
start = text.index("_AGENT_TASK_LEGACY_V1_FIELDS")
end = text.index("\n\n\nclass _LocalRuntimeCoordinatorPort", start)
text = text[:start] + '_AGENT_TASK_V1_FIELDS = frozenset(\n    {"type", "version", "binding", "user_prompt", "planning", "thinking"}\n)' + text[end:]
text = text.replace("        binding_digest: str,\n        session_id: str,\n        request: ResumeSessionRequest,", "        agent_digest: str,\n        binding_digest: str,\n        session_id: str,\n        request: ResumeSessionRequest,", 1)
text = text.replace("        catalog: AgentDefinitionCatalog,", "        catalog: AgentCatalog,", 1)

replace_span(
    "    def agent(\n",
    "    def _definition(\n",
'''    def agent(
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

''')

replace_span(
    "    def _definition(\n",
    "    async def _compile_agent(\n",
'''    def _definition(self, agent_digest: str) -> AgentDefinition:
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

''')

replace_span(
    "    async def start(\n",
    "    async def cancel(\n",
'''    async def _start_for_agent(
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
            handle = (
                await self._local_coordinator.run(binding.digest, request)
                if local_stream and self._local_coordinator is not None
                else await self.execution.run(binding.digest, request)
            )
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

''')

replace_span(
    "    def stream(\n",
    "    async def run_graph(\n",
'''    def _stream_for_agent(
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
        return self._stream_agent(
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

    async def _stream_agent(
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

    def _task_for_agent(
        self,
        agent_digest: str,
        node_id: str,
        user_prompt: str,
        *,
        dependencies: tuple[str, ...],
        budget_cost: int,
        output: "type[BaseModel] | None",
        planning: bool,
        thinking: bool,
    ) -> TaskNode:
        self._ensure_open()
        validate_user_prompt(user_prompt)
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        binding = self._bind_agent(agent_digest, output=output)
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

''')

replace_span(
    "    async def _admit_graph(\n",
    "    async def _ensure_session(\n",
'''    async def _admit_graph(
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
            if (
                frozenset(payload) != _AGENT_TASK_V1_FIELDS
                or payload.get("type") != "linktools.ai.agent"
                or payload.get("version") != 1
            ):
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

''')

replace_span(
    "    async def _ensure_session(\n",
    "    def _ensure_open(\n",
'''    async def _ensure_session(
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

''')

Path(TARGET).write_text(text, encoding="utf-8")
