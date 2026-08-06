#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static Bundle source generation."""

from pathlib import Path


def write_bundle(path: "str | Path", bundle_id: str, agent_name: str, model_name: str) -> Path:
    if not model_name.strip():
        raise ValueError("bundle model name must not be empty")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n"
        "from typing import Any, Final\n"
        "from pydantic_ai import Agent\n"
        "from pydantic_ai.durable_exec.temporal import TemporalDurability\n"
        "from ...agent.deps import AgentDeps\n\n"
        f"BUNDLE_ID: Final[str] = {bundle_id!r}\n"
        f"AGENT_NAME: Final[str] = {agent_name!r}\n"
        f"MODEL_NAME: Final[str] = {model_name!r}\n"
        "TOOLSET_IDS: Final[tuple[str, ...]] = ()\n"
        "agent: \"Agent[AgentDeps, Any]\" = Agent(\n"
        "    MODEL_NAME, name=AGENT_NAME, deps_type=AgentDeps, output_type=Any,\n"
        "    capabilities=[TemporalDurability(name=AGENT_NAME, deps_type=AgentDeps)],\n"
        ")\n",
        encoding="utf-8",
    )
    return target


__all__ = ["write_bundle"]
