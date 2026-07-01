#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compose agent prompts, agent files, and runtime context."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import AgentSpec, SkillSpec, SubagentSpec

from .skill_view import skill_summaries as _skill_summaries


DEFAULT_SYSTEM_PROMPT = (
    "You are a SecOpsEngine agent. Follow the current task prompt and use only the provided runtime context, "
    "tool results, and explicitly readable files. Do not fabricate facts. Record missing information in data_gaps "
    "when the output schema supports it. Output the exact structured format requested by the current task."
)

_IGNORED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_PROMPT_FILES = {"agent.md", "prompt.md", "agent-group.yaml"}


@dataclass(frozen=True, slots=True)
class PromptContext:
    spec: "AgentSpec"
    input_data: Any
    skills: "list[SkillSpec]"
    subagents: "list[SubagentSpec]"
    runtime_dir: Path | None = None


def build_prompt(context: PromptContext) -> str:
    parts: list[str] = []
    instr = context.spec.instructions.strip()
    if instr:
        parts.append(instr)
    parts.extend(f"# {title}\n{body.strip()}" for title, body in context.spec.prompt_sections.items())
    parts.append(runtime_context_note(context.input_data))
    if path_note := runtime_files_note(context.runtime_dir):
        parts.append(path_note)
    if note := _agent_files_note(context.spec):
        parts.append(note)
    if skill_note := candidate_skills_note(context.skills):
        parts.append(skill_note)
    if sub_note := candidate_subagent_note(context.subagents):
        parts.append(sub_note)
    return "\n\n".join(parts)


def runtime_context_note(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return "\n".join([
            "# Runtime Context",
            "```json",
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            "```",
        ])
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif value is None:
        text = ""
    else:
        text = str(value)
    return "\n".join(["# Runtime Context", text])


def runtime_files_note(runtime_dir: Path | None) -> str:
    """Describe runtime path boundaries available to the agent."""
    if runtime_dir is None:
        return ""
    lines = [
        "# Runtime Files",
        f"Working directory (cwd): `{runtime_dir}`",
        "Write output files using relative paths. To access files outside this directory, use absolute paths.",
    ]
    return "\n".join(lines)


def agent_files_note(
    agent_dir: Path,
    files: list[dict[str, Any]],
) -> str:
    """Describe agent-owned files and runtime workspace boundaries."""
    if not files:
        return ""
    by_cat: dict[str, list[str]] = {}
    for item in files:
        rel = str(item.get("path") or "")
        cat = str(item.get("category") or "base")
        if rel:
            by_cat.setdefault(cat, []).append(rel)

    header = "\n".join([
        "# Reference Files",
        f"agent_dir: `{agent_dir}` (read-only; use agent_dir/relative_path as the absolute path)",
        "Files:",
    ])
    all_files: list[str] = []
    for cat in ("scripts", "references", "templates", "assets"):
        all_files.extend(f"- {r}" for r in sorted(by_cat.pop(cat, [])))
    for rels in [sorted(v) for v in sorted(by_cat.values())]:
        all_files.extend(f"- {r}" for r in rels)
    return header + "\n" + "\n".join(all_files)


def _agent_files_note(spec: "AgentSpec") -> str | None:
    if spec.base_dir is None or not spec.base_dir.is_dir():
        return None
    file_specs: list[dict[str, str | int]] = []
    for item in sorted(spec.base_dir.rglob("*"), key=lambda p: p.as_posix()):
        rel = item.relative_to(spec.base_dir)
        if any(part in _IGNORED_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if rel.as_posix() in _PROMPT_FILES or not item.is_file():
            continue
        absolute_path = item.resolve()
        rel_path = rel.as_posix()
        category = rel_path.split("/", 1)[0] if "/" in rel_path else "base"
        file_specs.append({
            "path": rel_path,
            "category": category,
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "absolute_path": str(absolute_path),
        })
    if not file_specs:
        return None
    return agent_files_note(spec.base_dir, file_specs) or None


def candidate_subagent_note(subagents: "list[SubagentSpec]") -> str:
    if not subagents:
        return ""
    items = [
        f"- `{s.name}`" + (f": {s.description}" if s.description else "")
        for s in subagents
    ]
    return "\n".join([
        "# Available Subagents",
        "Backtick value is the subagent_id; call via `call_subagent` tool.",
        *items
    ])


def unavailable_adapters_note(sources: "list[str]") -> str:
    if not sources:
        return ""
    items = [f"- `{s}`" for s in sources]
    return "\n".join([
        "# Unavailable Adapters",
        "The following MCP sources declared in this agent's allowed-tools are not connected in this "
        "environment. Do not call their tools (the call will not succeed and will waste your turn budget). "
        "Proceed with the Runtime Context and any other available tools, and record "
        "`<source>_adapter_unavailable` in `data_gaps`/`degraded_context` for each one listed below:",
        *items,
    ])


def candidate_skills_note(skills: "list[SkillSpec]") -> str:
    skill_summaries = _skill_summaries(skills)
    items: list[str] = []
    for skill in skill_summaries:
        skill_id = str(skill.get("skill_id") or "").strip()
        if not skill_id:
            continue
        description = str(skill.get("description") or "").strip()
        items.append(f"- `{skill_id}`" + (f": {description}" if description else ""))
    if not items:
        return ""
    return "\n".join([
        "# Available Skills",
        "Backtick value is the skill_id; use `skill_view` to inspect before calling. Skills: ",
        *items
    ])
