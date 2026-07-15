#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Skill-private subagent primitives.

A skill's private agents live under ``<skill_root>/agents/*.md`` and are reached
by ``instruction_path`` RELATIVE TO THE ACTIVE SKILL -- never registered globally
. This module holds the pure, side-effect-free core: the active-skill
context, the path resolver (with symlink/escape rejection), and the spec parser.

The provider/resolver that plug into ``call_subagent`` live alongside the
subagent capability (see :mod:`linktools.ai.subagent`); they compose these
primitives. Keeping the security-critical path logic here, isolated and
unit-testable, is the point: ``resolve_skill_agent_path`` is the single choke
point every skill-private resolution passes through."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import contextvars

from ..agent.spec import ToolRef
from ..errors import SkillResourceAccessError, SubagentResolutionError
from ..registry.parser import StrictConfigReader, parse_markdown_text, parse_tool_refs

# Default tools for a skill-private agent with no ``tools:`` key.
_DEFAULT_TOOLS: tuple = (ToolRef(kind="builtin", name="file-read"),)


# ---- Active-skill contextvar (set by read_skill, read by call_subagent) ----
# A ContextVar holds the skill the current task most recently read, so the
# active skill is in scope for a call_subagent(instruction_path=...) issued in
# the same turn. Skill isolation across runs is enforced in the subagent
# executor: the child run's drive resets this var to None, so a subagent starts
# outside any skill and cannot address its parent's active skill. (Structural
# defense-in-depth: skill-private agents also default to max_depth=0, and the
# permission intersection strips any subagent capability they did not earn.)
_active_skill_var: "contextvars.ContextVar[ActiveSkillContext | None]" = (
    contextvars.ContextVar("linktools_active_skill", default=None)
)


def get_active_skill() -> "ActiveSkillContext | None":
    """The skill activated by the most recent ``read_skill`` in this task, or None."""
    return _active_skill_var.get()


def set_active_skill(
    ctx: "ActiveSkillContext | None",
) -> "contextvars.Token[ActiveSkillContext | None]":
    return _active_skill_var.set(ctx)


def reset_active_skill(token) -> None:
    _active_skill_var.reset(token)


def skill_subagent_to_agent_spec(
    spec,
    *,
    model_policy,
    parent_delegated: "set[str] | None" = None,
):
    """Build an executable AgentSpec from a skill-private subagent spec, applying
    the permission intersection: a private agent may only keep tools the
    parent can delegate. ``parent_delegated=None`` means "no constraint" (the
    parent delegated everything); otherwise requested tools are filtered by name.

    A private agent can therefore never ESCALATE beyond the parent's toolset --
    a grader that declares ``terminal`` keeps it only if the parent has it too.
    """
    from ..agent.spec import AgentSpec, PromptSpec

    if parent_delegated is None:
        tools = tuple(spec.requested_tools)
    else:
        tools = tuple(t for t in spec.requested_tools if t.name in parent_delegated)
    return AgentSpec(
        id=identity(spec.skill_id, spec.instruction_path),
        name=spec.name,
        model=model_policy,
        instructions=PromptSpec(instructions=spec.instructions),
        tools=tools,
        output_schema=str,
    )


@dataclass(frozen=True, slots=True)
class ActiveSkillContext:
    """The skill a run is currently inside.

    Recorded in the tool-execution context after a successful ``read_skill`` so
    a later ``call_subagent(instruction_path=...)`` resolves the path relative to
    THIS skill. Carries the skill ``revision`` so a resolution fails if the skill
    changed on disk between the read and the subagent call."""

    skill_id: str
    skill_root: Path
    revision: str


def resolve_skill_agent_path(
    *,
    skill_root: Path,
    instruction_path: str,
) -> Path:
    """Resolve a skill-private agent path, enforcing the invariants.

    The path must be relative, must be under ``agents/``, must be Markdown, must
    exist, and -- after ``resolve(strict=True)`` -- must not escape ``agents/``
    (which also rejects a symlink that points outside). Returns the resolved,
    validated absolute path.

    Rejected (each with SkillResourceAccessError): absolute paths; paths whose
    first segment is not ``agents``; paths that escape ``agents/`` after resolve
    (``../agents/x.md``, symlink-to-outside); non-Markdown; missing files.
    """
    relative = Path(instruction_path)

    if relative.is_absolute():
        raise SkillResourceAccessError("absolute paths are forbidden")
    if not relative.parts or relative.parts[0] != "agents":
        raise SkillResourceAccessError("skill subagent must be under agents/")

    # resolve(strict=True) follows symlinks AND raises if the target does not
    # exist, so a symlink that escapes agents/ resolves outside and is caught
    # by the boundary check below, and a missing file/dangling link fails here.
    try:
        agents_root = (skill_root / "agents").resolve(strict=True)
        candidate = (skill_root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SkillResourceAccessError("skill agent path is not a file") from exc

    if not (candidate == agents_root or agents_root in candidate.parents):
        raise SkillResourceAccessError("skill agent path escapes agents/")
    if candidate.suffix.lower() != ".md":
        raise SkillResourceAccessError("skill agent must be Markdown")
    if not candidate.is_file():
        raise SkillResourceAccessError("skill agent path is not a file")

    return candidate


@dataclass(frozen=True, slots=True)
class SkillSubagentSpec:
    """A resolved skill-private subagent.

    Uniquely identified by ``skill_id`` + ``instruction_path`` (NOT a global id,
    so two skills may both have ``agents/grader.md``). ``fingerprint`` is a stable
    digest of the identity so callers can dedupe/cache without re-reading."""

    skill_id: str
    instruction_path: str
    name: str
    description: "str | None"
    instructions: str
    requested_tools: tuple
    timeout_seconds: int
    max_depth: int
    fingerprint: str


def _fingerprint(skill_id: str, instruction_path: str, instructions: str) -> str:
    payload = f"{skill_id}\n{instruction_path}\n{instructions}".encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def parse_skill_subagent(
    *,
    skill_id: str,
    instruction_path: str,
    path: Path,
    default_timeout_seconds: int,
) -> SkillSubagentSpec:
    """Parse a skill-private agent Markdown file.

    Frontmatter is optional: without it, ``name`` is the file stem, tools default
    to ``(file-read,)``, and timeout/max_depth use the defaults. With frontmatter,
    the strict reader rejects unknown fields and explicit nulls. Reuses the
    project's Markdown/frontmatter loader (no bespoke parser).
    """
    text = path.read_text(encoding="utf-8")
    payload, body = parse_markdown_text(text, source=str(path))
    instructions = body.strip()

    name = path.stem
    description: "str | None" = None
    requested_tools: tuple = _DEFAULT_TOOLS
    timeout_seconds = default_timeout_seconds
    max_depth = 0

    if payload:
        reader = StrictConfigReader(
            payload,
            allowed={"name", "description", "tools", "timeout_seconds", "max_depth"},
            context=f"skill agent {skill_id}/{instruction_path}",
        )
        resolved = reader.optional_str("name")
        if resolved is not None:
            name = resolved.strip() or name
        description = reader.optional_str("description")
        if "tools" in payload:
            # An explicit tools list (even empty) overrides the default.
            requested_tools = parse_tool_refs(payload.get("tools"))
        timeout_seconds = int(
            reader.positive_number(
                "timeout_seconds", default=float(default_timeout_seconds)
            )
        )
        max_depth = reader.non_negative_int("max_depth", default=0)

    return SkillSubagentSpec(
        skill_id=skill_id,
        instruction_path=instruction_path,
        name=name,
        description=description,
        instructions=instructions,
        requested_tools=requested_tools,
        timeout_seconds=timeout_seconds,
        max_depth=max_depth,
        fingerprint=_fingerprint(skill_id, instruction_path, instructions),
    )


def require_active_skill(
    active_skill: "ActiveSkillContext | None",
) -> ActiveSkillContext:
    """Centralize the rule: an ``instruction_path`` call requires an
    active skill. Raises SubagentResolutionError with a clear message otherwise."""
    if active_skill is None:
        raise SubagentResolutionError("instruction_path requires an active skill")
    return active_skill


def identity(skill_id: str, instruction_path: str) -> str:
    """The stable public identity of a skill-private agent: the
    ``skill_id`` and ``instruction_path`` pair, never a bare name."""
    return f"{skill_id}/{instruction_path}"


def validate_call_request(
    *,
    name: "str | None",
    instruction_path: "str | None",
    task: str,
) -> None:
    """Validate a ``call_subagent`` request: exactly one of
    ``name`` / ``instruction_path`` must be set, and ``task`` must not be blank."""
    if (name is None) == (instruction_path is None):
        raise SubagentResolutionError(
            "exactly one of name or instruction_path is required"
        )
    if not task.strip():
        raise SubagentResolutionError("task must not be blank")
