#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai doctor`: validate a project's configuration and capabilities.

Checks: config parseable, every agent/skill parses, every skill-private
``agents/*.md`` resolves under ``agents/`` (path safety), MCP env references
resolve (fail-on-missing), the runtime inspects cleanly, and the project state
directory is writable. Exits non-zero if any check fails."""

import asyncio
import tempfile
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.core import environ
from linktools.ai.mcp.env import expand_env_mapping

from .assembly import build_cli_runtime
from .project import load_project

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """validate a project's configuration"""

    def init_arguments(self, parser: "CommandParser") -> None:
        pass

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._doctor_async())

    async def _doctor_async(self) -> "int | None":
        project = load_project(data_root=environ.get_data_path("ai"))
        bundle = build_cli_runtime(project=project, model_router=None)
        failures: "list[str]" = []

        def ok(label: str) -> None:
            self.logger.info(f"[ok] {label}")

        def fail(label: str, detail: str) -> None:
            self.logger.error(f"[fail] {label}: {detail}")
            failures.append(label)

        ok("project config")
        ok(f"default agent: {project.default_agent}")

        # Agents parse.
        agent_ids = await bundle.agents.list_ids()
        if project.default_agent not in agent_ids:
            fail("default agent", f"{project.default_agent!r} not in agents")
        for agent_id in agent_ids:
            try:
                await bundle.agents.get(agent_id)
                ok(f"agent: {agent_id}")
            except Exception as exc:
                fail(f"agent: {agent_id}", str(exc))

        # Skills + skill-private agents (path safety on each agents/*.md).
        for skill_id in await bundle.skill_index.list_ids():
            try:
                info = await bundle.skill_index.get(skill_id)
                ok(f"skill: {skill_id}")
            except Exception as exc:
                fail(f"skill: {skill_id}", str(exc))
                continue
            if info is None:
                continue
            for agent_path in info.list_private_agents():
                rel = agent_path.relative_to(info.root)
                try:
                    from linktools.ai.skill.private import resolve_skill_agent_path

                    resolve_skill_agent_path(
                        skill_root=info.root, instruction_path=str(rel)
                    )
                    ok(f"skill agent: {skill_id}/{rel}")
                except Exception as exc:
                    fail(f"skill agent: {skill_id}/{rel}", str(exc))

        # MCP env expansion (fail-on-missing).
        for mcp_id in await bundle.mcp.list_ids():
            try:
                spec = await bundle.mcp.get(mcp_id)
                expand_env_mapping(getattr(spec, "env", None))
                ok(f"MCP: {mcp_id}")
            except Exception as exc:
                fail(f"MCP: {mcp_id}", str(exc))

        # Runtime inspects cleanly for the default agent.
        try:
            default_spec = await bundle.agents.get(project.default_agent)
            await bundle.runtime.inspect(default_spec)
            ok("runtime inspect")
        except Exception as exc:
            fail("runtime inspect", str(exc))

        # Storage writable.
        try:
            project.state_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=project.state_root):
                pass
            ok("storage writable")
        except Exception as exc:
            fail("storage writable", str(exc))

        if failures:
            self.logger.error(f"{len(failures)} check(s) failed")
            return 1
        return 0


command = Command()
