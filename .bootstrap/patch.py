#!/usr/bin/env python3
import subprocess
from pathlib import Path

BASE = "a4ded79a152ccfbc635503ee7b7bd394bd50970e"


def baseline(rel: str) -> str:
    return subprocess.check_output(("git", "show", f"{BASE}:{rel}"), text=True)


def write(rel: str, text: str) -> None:
    Path(rel).write_text(text, encoding="utf-8")

# Public Session contract: Agent identity is stable, execution binding is per turn.
rel = "linktools-ai/src/linktools/ai/runtime/service_api.py"
text = baseline(rel)
text = text.replace("    binding_digest: str\n    status: SessionStatus", "    agent_digest: str\n    status: SessionStatus", 1)
text = text.replace(
    "    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView: ...",
    "    async def create(self, agent_digest: str, request: CreateSessionRequest) -> SessionView: ...",
)
text = text.replace(
    "    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...",
    "    async def resume(self, agent_digest: str, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...",
)
text = text.replace(
    "    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView: ...",
    "    async def fork(self, agent_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView: ...",
)
text = text.replace(
    "    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView: ...",
    "    async def update(self, agent_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView: ...",
)
write(rel, text)

# Local coordinator carries both stable Agent identity and exact execution binding.
rel = "linktools-ai/src/linktools/ai/runtime/_coordinator.py"
text = baseline(rel)
text = text.replace(
    "    async def resume(\n        self,\n        binding_digest: str,\n        session_id: str,",
    "    async def resume(\n        self,\n        agent_digest: str,\n        binding_digest: str,\n        session_id: str,",
    1,
)
text = text.replace(
    "            handle = await self._session._resume_with_launch_gate(\n                binding_digest,\n                session_id,",
    "            handle = await self._session._resume_with_launch_gate(\n                agent_digest,\n                binding_digest,\n                session_id,",
    1,
)
write(rel, text)

# Session service persists and compares only agent_digest; binding_digest is passed only to the launched execution.
rel = "linktools-ai/src/linktools/ai/runtime/_session.py"
text = baseline(rel)
text = text.replace("        binding_digest: str | None = None,", "        agent_digest: str | None = None,", 1)
text = text.replace("        binding_digest: str,\n    ) -> tuple[object, ...]: ...", "        agent_digest: str,\n    ) -> tuple[object, ...]: ...", 1)
text = text.replace("    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView:", "    async def create(self, agent_digest: str, request: CreateSessionRequest) -> SessionView:")
text = text.replace('                "binding": binding_digest,', '                "agent_digest": agent_digest,', 1)
text = text.replace("            binding_digest=binding_digest,\n            status=SessionStatus.OPEN,", "            agent_digest=agent_digest,\n            status=SessionStatus.OPEN,", 1)
text = text.replace("                    binding_digest=record.binding_digest,", "                    agent_digest=record.agent_digest,", 1)
text = text.replace("                binding_digest=record.binding_digest,", "                agent_digest=record.agent_digest,", 1)
text = text.replace(
    "    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:\n        return await self._resume(binding_digest, session_id, request, launch_gate=None)",
    "    async def resume(self, agent_digest: str, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:\n        return await self._resume(agent_digest, binding_digest, session_id, request, launch_gate=None)",
)
text = text.replace(
    "    async def _resume_with_launch_gate(\n        self,\n        binding_digest: str,",
    "    async def _resume_with_launch_gate(\n        self,\n        agent_digest: str,\n        binding_digest: str,",
    1,
)
text = text.replace(
    "        return await self._resume(binding_digest, session_id, request, launch_gate=gate)",
    "        return await self._resume(agent_digest, binding_digest, session_id, request, launch_gate=gate)",
    1,
)
text = text.replace(
    "    async def _resume(\n        self,\n        binding_digest: str,",
    "    async def _resume(\n        self,\n        agent_digest: str,\n        binding_digest: str,",
    1,
)
text = text.replace("            if record.binding_digest != binding_digest:", "            if record.agent_digest != agent_digest:", 1)
text = text.replace("    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView:", "    async def fork(self, agent_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView:")
text = text.replace("            if source.binding_digest != binding_digest:", "            if source.agent_digest != agent_digest:", 1)
text = text.replace('                    "binding": source.binding_digest,', '                    "agent_digest": source.agent_digest,', 1)
text = text.replace("                binding_digest=source.binding_digest,", "                agent_digest=source.agent_digest,", 1)
text = text.replace("    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:", "    async def update(self, agent_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:")
text = text.replace("            return await self._update(binding_digest, session_id, request)", "            return await self._update(agent_digest, session_id, request)", 1)
text = text.replace("    async def _update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:", "    async def _update(self, agent_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView:")
text = text.replace("        if current.binding_digest != binding_digest:", "        if current.agent_digest != agent_digest:", 1)
text = text.replace("            record.binding_digest,\n            record.status,", "            record.agent_digest,\n            record.status,", 1)
# These two names are Session-context contract parameters, not execution bindings.
text = text.replace("binding_digest=record.binding_digest", "agent_digest=record.agent_digest")
# Remove baseline imports already unused independently of this refactor only because this file is now changed and linted.
text = text.replace("    ConversationHistoryRecord,\n", "")
text = text.replace("    HistoryQuality,\n", "")
write(rel, text)

# Persistence contracts.
rel = "linktools-ai/src/linktools/ai/runtime/state/_contracts.py"
text = baseline(rel)
future = "from __future__ import annotations\n\n"
if text.startswith(future):
    text = text[len(future):]
    close = text.index('"""', 3) + 3
    text = text[:close] + "\n\nfrom __future__ import annotations" + text[close:]
text = text.replace("class ContextProjection:\n    binding_digest: str", "class ContextProjection:\n    agent_digest: str")
text = text.replace("    owner_principal_id: str\n    binding_digest: str\n    status: SessionStatus", "    owner_principal_id: str\n    agent_digest: str\n    status: SessionStatus", 1)
write(rel, text)

# ContextProjection is Agent-identity scoped everywhere in transcript history.
rel = "linktools-ai/src/linktools/ai/runtime/state/_history.py"
text = baseline(rel).replace("binding_digest", "agent_digest")
write(rel, text)

# Step archive projection checks use Agent identity. Execution binding fields outside projection code stay unchanged.
rel = "linktools-ai/src/linktools/ai/runtime/state/_steps.py"
text = baseline(rel)
text = text.replace("snapshot.projection.binding_digest", "snapshot.projection.agent_digest")
text = text.replace("projection.binding_digest", "projection.agent_digest")
text = text.replace("binding_digest=projection_binding", "agent_digest=projection_binding")
text = text.replace(
    "                binding_digest=binding_digest,\n            )\n        ).model_messages()",
    "                agent_digest=agent_digest,\n            )\n        ).model_messages()",
    1,
)
text = text.replace(
    "        binding_digest: str | None = None,\n    ) -> tuple[object, ...]:\n        require_no_run_history_lock(\"StateStepArchive.load_session_model_context\")",
    "        agent_digest: str | None = None,\n    ) -> tuple[object, ...]:\n        require_no_run_history_lock(\"StateStepArchive.load_session_model_context\")",
    1,
)
text = text.replace(
    "        binding_digest: str | None = None,\n    ) -> bool:\n        require_no_run_history_lock(\n            \"StateStepArchive.verify_snapshot_projection\"",
    "        agent_digest: str | None = None,\n    ) -> bool:\n        require_no_run_history_lock(\n            \"StateStepArchive.verify_snapshot_projection\"",
    1,
)
text = text.replace("        if binding_digest is not None and projection.agent_digest != binding_digest:", "        if agent_digest is not None and projection.agent_digest != agent_digest:", 1)
text = text.replace("            binding_digest=binding_digest,\n        )\n        return (", "            agent_digest=agent_digest,\n        )\n        return (", 1)
write(rel, text)

print("session and context projection identity split applied")
