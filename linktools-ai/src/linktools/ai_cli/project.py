#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Project discovery and configuration for the ``lt ai`` CLI/TUI.

All project configuration lives under ``<root>/.linktools/``; run state lives
under ``<data_root>/projects/<project_hash>/`` so two projects never share
state. This module is pure path/config plumbing -- it loads nothing into the
runtime (that is :mod:`linktools.ai_cli.runtime`'s job)."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from linktools.cli import CommandError


class ProjectConfigError(CommandError):
    """Raised when a project's ``.linktools/config.yaml`` exists but is invalid
    (wrong version, wrong type, blank agent). A missing config file is NOT an
    error — the project root defaults to cwd and config values use defaults."""


@dataclass(frozen=True, slots=True)
class CliProject:
    root: Path
    config_root: Path
    agents_root: Path
    skills_root: Path
    mcp_root: Path
    tools_root: Path
    state_root: Path
    default_agent: str
    default_session: str
    allow_mcp_wildcard: bool
    subagent_max_depth: int
    subagent_max_concurrency: int
    subagent_timeout_seconds: int


def find_project_root(start: "Path | None" = None) -> Path:
    """Walk upward from ``start`` (default cwd) to the first directory holding a
    ``.linktools/config.yaml``. If none is found, ``start`` (resolved) is the
    project root — the config file is optional, not a prerequisite."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".linktools" / "config.yaml").is_file():
            return candidate
    return current


def project_hash(root: Path) -> str:
    """A stable 16-hex id for a project root, so its run-state directory is
    isolated from every other project's."""
    return sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]


def load_project(*, data_root: Path, start: "Path | None" = None) -> CliProject:
    """Discover the project root and parse ``.linktools/config.yaml`` (optional).

    ``data_root`` is the ai data directory; the project's run state is placed
    under ``<data_root>/projects/<project_hash>/``. If ``config.yaml`` exists it
    is validated (``version: 1``, non-blank ``default_agent`` / ``default_session``,
    ``mcp`` / ``subagents`` sections); if absent, all values use defaults."""
    root = find_project_root(start)
    config_root = root / ".linktools"
    config_file = config_root / "config.yaml"

    if config_file.is_file():
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ProjectConfigError("config.yaml must be a mapping")
        if raw.get("version") != 1:
            raise ProjectConfigError("unsupported config version")
    else:
        raw = {}

    agent = raw.get("default_agent", "default")
    session = raw.get("default_session", "main")
    if not isinstance(agent, str) or not agent.strip():
        raise ProjectConfigError("default_agent must not be blank")
    if not isinstance(session, str) or not session.strip():
        raise ProjectConfigError("default_session must not be blank")

    mcp = raw.get("mcp") or {}
    subagents = raw.get("subagents") or {}

    try:
        subagent_max_depth = int(subagents.get("max_depth", 3))
        subagent_max_concurrency = int(subagents.get("max_concurrency", 4))
        subagent_timeout_seconds = int(subagents.get("default_timeout_seconds", 120))
    except (TypeError, ValueError) as exc:
        raise ProjectConfigError(f"invalid subagents config: {exc}") from exc

    return CliProject(
        root=root,
        config_root=config_root,
        agents_root=config_root / "agents",
        skills_root=config_root / "skills",
        mcp_root=config_root / "mcp",
        tools_root=config_root / "tools",
        state_root=data_root / "projects" / project_hash(root),
        default_agent=agent.strip(),
        default_session=session.strip(),
        allow_mcp_wildcard=bool(mcp.get("allow_wildcard", False)),
        subagent_max_depth=subagent_max_depth,
        subagent_max_concurrency=subagent_max_concurrency,
        subagent_timeout_seconds=subagent_timeout_seconds,
    )
