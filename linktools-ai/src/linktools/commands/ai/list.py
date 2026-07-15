#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai list`: enumerate a project's capabilities (agents/skills/mcp) and
run state (sessions/approvals).

Reads directly from the project registries + the directory skill index -- no
Runtime or model configuration is needed, so it works in a freshly-initialized
project. ``list skills --verbose`` also shows each skill's private
``agents/*.md`` (their skill-scoped path, never a global id)."""

import asyncio
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.core import environ
from linktools.ai.registry.agent import AgentRegistry
from linktools.ai.registry.mcp import MCPRegistry
from linktools.ai.registry.parser import SpecLoader

from .skill_index import DirectorySkillIndex
from .project import load_project
from .assembly import project_storage

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser

_KINDS = ("agents", "skills", "mcp", "sessions", "approvals")


class Command(BaseCommand):
    """list project capabilities or run state"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("kind", choices=_KINDS, help="what to list")
        parser.add_argument(
            "--verbose", action="store_true", help="show skill-private agents"
        )

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._list_async(args))

    async def _list_async(self, args: "Namespace") -> "int | None":
        project = load_project(data_root=environ.get_data_path("ai"))
        kind = args.kind
        if kind == "agents":
            await self._list_agents(project)
        elif kind == "skills":
            await self._list_skills(project, verbose=args.verbose)
        elif kind == "mcp":
            await self._list_mcp(project)
        elif kind == "sessions":
            await self._list_sessions()
        elif kind == "approvals":
            await self._list_approvals()
        return 0

    async def _list_agents(self, project) -> None:
        registry = AgentRegistry(SpecLoader.from_filesystem(project.agents_root))
        ids = await registry.list_ids()
        if not ids:
            self.logger.info("no agents")
            return
        for agent_id in ids:
            self.logger.info(agent_id)

    async def _list_skills(self, project, *, verbose: bool) -> None:
        index = DirectorySkillIndex(project.skills_root)
        ids = await index.list_ids()
        if not ids:
            self.logger.info("no skills")
            return
        for skill_id in ids:
            self.logger.info(skill_id)
            if verbose:
                info = await index.get(skill_id)
                if info is None:
                    continue
                for agent_path in info.list_private_agents():
                    rel = agent_path.relative_to(info.root)
                    self.logger.info(f"  {rel}")

    async def _list_mcp(self, project) -> None:
        registry = MCPRegistry(SpecLoader.from_filesystem(project.mcp_root))
        ids = await registry.list_ids()
        if not ids:
            self.logger.info("no mcp servers")
            return
        for mcp_id in ids:
            self.logger.info(mcp_id)

    async def _list_sessions(self) -> None:
        from .sessions import _list_sessions as _list

        storage = project_storage()
        for rec in await _list(storage):
            self.logger.info(
                f"{rec.id}\t{rec.status.value}\t{rec.updated_at.isoformat()}"
            )

    async def _list_approvals(self) -> None:
        from .approvals import _list_pending_approvals as _list

        storage = project_storage()
        requests = await _list(storage)
        if not requests:
            self.logger.info("no pending approvals")
            return
        for req in requests:
            self.logger.info(f"{req.id}\trun={req.run_id}\ttool={req.tool_name}")


command = Command()
