#!/usr/bin/env python3
import subprocess
from pathlib import Path

BASE = "a4ded79a152ccfbc635503ee7b7bd394bd50970e"


def baseline(path: str) -> str:
    return subprocess.check_output(("git", "show", f"{BASE}:{path}"), text=True)


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_span(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement + text[right:]


# Public Session contract pins Agent identity. Exact execution binding is supplied only to resume.
path = "linktools-ai/src/linktools/ai/runtime/service_api.py"
text = baseline(path)
text = text.replace("    binding_digest: str\n    status: SessionStatus", "    agent_digest: str\n    status: SessionStatus", 1)
text = text.replace(
    "    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView: ...",
    "    async def create(self, agent_digest: str, request: CreateSessionRequest) -> SessionView: ...",
    1,
)
text = text.replace(
    "    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...",
    "    async def resume(self, agent_digest: str, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...",
    1,
)
text = text.replace(
    "    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView: ...",
    "    async def fork(self, agent_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView: ...",
    1,
)
text = text.replace(
    "    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView: ...",
    "    async def update(self, agent_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView: ...",
    1,
)
write(path, text)

# State records use Agent identity for conversation/model-context semantics.
path = "linktools-ai/src/linktools/ai/runtime/state/_contracts.py"
text = baseline(path)
old_header = '''#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nfrom __future__ import annotations\n\n"""Runtime persistence contracts and immutable records.\n\nThis module contains no backend, filesystem, database, or workflow code.  It\nis the single semantic boundary shared by the local and SQL implementations.\n"""\n'''
new_header = '''#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Runtime persistence contracts and immutable records.\n\nThis module contains no backend, filesystem, database, or workflow code.  It\nis the single semantic boundary shared by the local and SQL implementations.\n"""\n\nfrom __future__ import annotations\n'''
if old_header not in text:
    raise RuntimeError("state contract header changed")
text = text.replace(old_header, new_header, 1)
text = text.replace(
    "class ContextProjection:\n    binding_digest: str\n",
    "class ContextProjection:\n    agent_digest: str\n",
    1,
)
text = text.replace(
    "    owner_principal_id: str\n    binding_digest: str\n    status: SessionStatus",
    "    owner_principal_id: str\n    agent_digest: str\n    status: SessionStatus",
    1,
)
write(path, text)

# Transcript projection identity is Agent identity throughout this module.
path = "linktools-ai/src/linktools/ai/runtime/state/_history.py"
text = baseline(path).replace("binding_digest", "agent_digest")
write(path, text)

# Session service validates Agent identity and uses binding_digest only when launching an execution.
path = "linktools-ai/src/linktools/ai/runtime/_session.py"
text = baseline(path)
text = text.replace(
    '''    async def load_session_model_context(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n        binding_digest: str | None = None,\n    ) -> tuple[object, ...]: ...''',
    '''    async def load_session_model_context(\n        self,\n        history_id: str,\n        *,\n        tenant_id: str,\n        agent_digest: str | None = None,\n    ) -> tuple[object, ...]: ...''',
    1,
)
text = text.replace(
    '''    async def load_model_context(\n        self,\n        *,\n        run_id: str,\n        binding_digest: str,\n    ) -> tuple[object, ...]: ...''',
    '''    async def load_model_context(\n        self,\n        *,\n        run_id: str,\n        agent_digest: str,\n    ) -> tuple[object, ...]: ...''',
    1,
)
create_start = text.index("    async def create(")
create_end = text.index("    async def get(", create_start)
create_block = text[create_start:create_end]
create_block = create_block.replace("binding_digest", "agent_digest")
create_block = create_block.replace('"binding": agent_digest', '"agent_digest": agent_digest')
text = text[:create_start] + create_block + text[create_end:]
text = text.replace(
    "                    binding_digest=record.binding_digest,",
    "                    agent_digest=record.agent_digest,",
    1,
)
text = text.replace(
    "                binding_digest=record.binding_digest,",
    "                agent_digest=record.agent_digest,",
    1,
)
resume_start = text.index("    async def resume(")
resume_end = text.index("    async def fork(", resume_start)
resume_block = '''    async def resume(\n        self,\n        agent_digest: str,\n        binding_digest: str,\n        session_id: str,\n        request: ResumeSessionRequest,\n    ) -> ExecutionHandle:\n        return await self._resume(\n            agent_digest,\n            binding_digest,\n            session_id,\n            request,\n            launch_gate=None,\n        )\n\n    async def _resume_with_launch_gate(\n        self,\n        agent_digest: str,\n        binding_digest: str,\n        session_id: str,\n        request: ResumeSessionRequest,\n        gate: _LaunchGate,\n    ) -> ExecutionHandle:\n        return await self._resume(\n            agent_digest,\n            binding_digest,\n            session_id,\n            request,\n            launch_gate=gate,\n        )\n\n    async def _resume(\n        self,\n        agent_digest: str,\n        binding_digest: str,\n        session_id: str,\n        request: ResumeSessionRequest,\n        *,\n        launch_gate: _LaunchGate | None,\n    ) -> ExecutionHandle:\n        async with self._session_consumer(session_id, request.principal.tenant_id):\n            record = await self._authorized(\n                session_id, request.principal, AuthorizationAction.SESSION_READ\n            )\n            record = await self._reconcile_terminal_admission(record)\n            await self._authorization.authorize(\n                request.principal,\n                AuthorizationAction.EXECUTION_RUN,\n                ResourceRef(\n                    ResourceKind.EXECUTION,\n                    session_id,\n                    request.principal.tenant_id,\n                ),\n            )\n            if record.agent_digest != agent_digest:\n                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)\n            execution_request = ExecutionRequest(\n                user_prompt=request.user_prompt,\n                principal=request.principal,\n                idempotency_key=request.idempotency_key,\n                memory_scope=request.memory_scope,\n                planning=request.planning,\n                thinking=request.thinking,\n            )\n            if self._gated_execution is None:\n                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n            if launch_gate is None:\n                try:\n                    return await self._gated_execution.run_for_session(\n                        binding_digest,\n                        session_id,\n                        execution_request,\n                    )\n                except AttributeError as error:\n                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error\n            try:\n                return await self._gated_execution._run_for_session_with_launch_gate(\n                    binding_digest,\n                    session_id,\n                    execution_request,\n                    launch_gate,\n                )\n            except AttributeError as error:\n                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error\n\n'''
text = text[:resume_start] + resume_block + text[resume_end:]
fork_start = text.index("    async def fork(")
fork_end = text.index("    async def update(", fork_start)
fork_block = text[fork_start:fork_end].replace("binding_digest", "agent_digest")
fork_block = fork_block.replace('"binding": source.agent_digest', '"agent_digest": source.agent_digest')
text = text[:fork_start] + fork_block + text[fork_end:]
update_start = text.index("    async def update(")
update_end = text.index("    async def close(", update_start)
update_block = text[update_start:update_end].replace("binding_digest", "agent_digest")
text = text[:update_start] + update_block + text[update_end:]
text = text.replace("            record.binding_digest,\n            record.status,", "            record.agent_digest,\n            record.status,", 1)
text = text.replace("Enforce session ownership, binding immutability, and revision CAS.", "Enforce session ownership, Agent identity immutability, and revision CAS.")
if ".binding_digest" in text:
    raise RuntimeError("Session service still reads a Session binding_digest")
write(path, text)
