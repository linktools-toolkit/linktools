#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai inspect`: show what a project agent resolves to (declared tools,
resolved tool descriptors, prompt sections, skills, skill-private agents, MCP,
subagents, warnings).

Delegates capability resolution to ``runtime.inspect`` (the single assembly-
inspection entry point) -- it does not re-implement capability assembly. Works
without a model client: capability resolution does not compile a pydantic-ai
Agent, so the runtime is built with ``model_router=None``."""

import asyncio
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.core import environ

from .assembly import build_cli_runtime
from .project import load_project

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """inspect a project agent's resolved capabilities"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument(
            "agent", nargs="?", default=None, help="agent id (default: project default)"
        )
        parser.add_argument("--json", action="store_true", help="emit JSON")

    def run(self, args: "Namespace") -> "int | None":
        return asyncio.run(self._inspect_async(args))

    async def _inspect_async(self, args: "Namespace") -> "int | None":
        project = load_project(data_root=environ.get_data_path("ai"))
        bundle = build_cli_runtime(project=project, model_router=None)
        agent_id = args.agent or project.default_agent
        try:
            spec = await bundle.agents.get(agent_id)
        except Exception as exc:  # RegistryNotFoundError et al.
            raise CommandError(f"agent not found: {agent_id}") from exc

        inspection = await bundle.runtime.inspect(spec)

        if args.json:
            import json

            print(json.dumps(_to_jsonable(inspection), indent=2, default=str))
            return 0

        self.logger.info(f"agent: {agent_id}")
        descriptors = getattr(inspection, "tool_descriptors", ()) or ()
        if descriptors:
            self.logger.info("tools:")
            for descriptor in descriptors:
                name = getattr(descriptor, "name", descriptor)
                self.logger.info(f"  - {name}")
        sections = getattr(inspection, "prompt_sections", {}) or {}
        if sections:
            self.logger.info("prompt sections:")
            for key in sections:
                self.logger.info(f"  - {key}")
        warnings = getattr(inspection, "warnings", ()) or ()
        for warning in warnings:
            self.logger.warning(f"warning: {warning}")

        # Skills + skill-private agents (from the directory index, not a global
        # registry -- private agents are never globally addressable).
        for skill_id in await bundle.skill_index.list_ids():
            self.logger.info(f"skill: {skill_id}")
            info = await bundle.skill_index.get(skill_id)
            if info is None:
                continue
            for agent_path in info.list_private_agents():
                self.logger.info(
                    f"  skill agent: {skill_id}/{agent_path.relative_to(info.root)}"
                )
        return 0


def _to_jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


command = Command()
