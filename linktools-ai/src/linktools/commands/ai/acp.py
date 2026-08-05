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
from linktools.ai.acp.protocol import AcpDependencyError, require_sdk

logger = environ.get_logger("commands.ai.acp")

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
    from linktools.ai.acp.client import AcpClient
    from linktools.ai.acp.codec import AcpCodec
    from linktools.ai.acp.protocol import AcpMode, AcpProtocol, CapabilityInput
    from linktools.ai.acp.server import run_acp_server
    from linktools.ai.cli.project import load_project
    from linktools.ai.cli.runtime import ProjectProcessLock, build_cli_runtime, load_agent_spec
    from linktools.ai.governance.identity import trusted_local_principal
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
    lock = ProjectProcessLock(project.state_root / "runtime.lock")
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
    principal = trusted_local_principal()
    protocol = AcpProtocol(
        principal=principal,
        spec_resolver=resolve,
        modes=modes,
        capability_input=CapabilityInput(modes=modes),
    )
    agent = LinktoolsAcpAgent(
        runtime=bundle.runtime,
        codec=AcpCodec(),
        client=AcpClient(project_root=project.root),
        protocol=protocol,
    )
    transport_error: "BaseException | None" = None
    preclose_results = ()
    client_failures = ()
    client_cleanup_error = False
    runtime_result = None
    runtime_error: "BaseException | None" = None
    try:
        await run_acp_server(agent)
    except BaseException as error:
        transport_error = error
    finally:
        try:
            preclose_results = await bundle.runtime.shutdown_sessions()
        except BaseException as error:
            logger.error(
                "event=ai.acp.preclose_failed session_count=0 error_id=%s",
                type(error).__name__,
                exc_info=environ.debug,
            )
        try:
            client_failures = await agent.client.close()
        except BaseException as error:
            client_failures = ()
            client_cleanup_error = True
            logger.error(
                "event=ai.acp.client_close_failed client_failure_count=1 error_id=%s",
                type(error).__name__,
                exc_info=environ.debug,
            )
        try:
            runtime_result = await bundle.runtime.aclose()
        except BaseException as error:
            runtime_error = error
            logger.error(
                "event=ai.acp.runtime_close_failed runtime_closed=False error_id=%s",
                type(error).__name__,
                exc_info=environ.debug,
            )
        finally:
            lock.release()
    runtime_closed = runtime_result is not None and runtime_result.closed and runtime_error is None
    logger.info(
        "event=ai.acp.shutdown_complete client_failure_count=%s runtime_closed=%s transport_error_id=%s preclose_failure_count=%s",
        len(client_failures),
        runtime_closed,
        type(transport_error).__name__ if transport_error is not None else None,
        sum(len(result.failures) for result in preclose_results),
    )
    if transport_error is not None:
        raise transport_error
    if not runtime_closed or client_cleanup_error or client_failures:
        return 4
    return 0


command = Command()
