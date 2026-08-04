#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`linktools ai acp`: start the ACP v1 stdio Agent."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.core import environ
from linktools.ai.acp.errors import AcpDependencyError, require_sdk

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser


class Command(BaseCommand):
    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None)
        parser.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default="warning")
        parser.add_argument("--log-file", type=Path, default=None)

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [AcpDependencyError]

    def run(self, args: "Namespace") -> int:
        try:
            return asyncio.run(_run(args))
        except asyncio.CancelledError:
            return 130


async def _run(args: "Namespace") -> int:
    require_sdk()
    from linktools.ai.acp.agent import LinktoolsAcpAgent
    from linktools.ai.acp.capabilities import AcpMode, CapabilityInput
    from linktools.ai.acp.client_services import AcpClientServices
    from linktools.ai.acp.persistence import AcpSessionRepository, ProjectProcessLock
    from linktools.ai.acp.server import serve_stdio
    from linktools.ai.acp.sessions import AcpSessionService
    from linktools.ai.cli.project import load_project
    from linktools.ai.cli.runtime import build_cli_runtime, load_agent_spec
    from linktools.ai.cli.client import trusted_local_principal
    from linktools.ai.execution.live_events import ExecutionEventHub

    logging_options = {
        "level": getattr(logging, args.log_level.upper()),
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
    }
    if args.log_file:
        logging_options["filename"] = str(args.log_file)
    else:
        logging_options["stream"] = sys.stderr
    logging.basicConfig(**logging_options)
    environ.debug = args.log_level == "debug"
    project = load_project(data_root=environ.get_data_path("ai"), start=args.project)
    lock = ProjectProcessLock(project.state_root / "acp" / "agent.lock")
    try:
        lock.acquire(project_root=project.root)
    except RuntimeError as exc:
        raise AcpDependencyError(str(exc)) from exc
    hub = ExecutionEventHub()
    bundle = build_cli_runtime(
        project=project,
        model_resolver=None,
        live_events=hub,
        require_tool_approval=True,
    )
    mode_ids = await bundle.agents.list_ids()
    if not mode_ids:
        mode_ids = (project.default_agent,)

    async def resolve(mode_id: str):
        return await load_agent_spec(bundle, mode_id)

    modes = tuple(AcpMode(mode_id, mode_id) for mode_id in mode_ids)
    client_services = AcpClientServices(project_root=project.root)
    session_service = AcpSessionService(
        runtime=bundle.runtime,
        repository=AcpSessionRepository(project.state_root),
        project_root=project.root,
        principal=trusted_local_principal(),
        default_mode_id=modes[0].id,
        mode_ids=tuple(mode.id for mode in modes),
        client_services=client_services,
    )
    agent = LinktoolsAcpAgent(
        runtime=bundle.runtime,
        event_hub=hub,
        session_service=session_service,
        project_root=str(project.root),
        spec_resolver=resolve,
        modes=modes,
        capability_input=CapabilityInput(modes=modes),
    )
    try:
        await serve_stdio(agent)
    finally:
        await bundle.runtime.aclose()
        lock.release()
    return 0


command = Command()
