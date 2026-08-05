#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`linktools ai acp`: start the ACP v1 stdio Agent."""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.core import environ
from linktools.errors import ConfigError
from linktools.ai.acp.protocol import AcpDependencyError, AcpTransportError, require_sdk

logger = environ.get_logger("commands.ai.acp")

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser
    from linktools.ai.runtime.session import ResourceFailure


@dataclass(slots=True)
class _AcpLifecycleResult:
    initialization_error: "BaseException | None" = None
    transport_error: "BaseException | None" = None
    preclose_failures: "tuple[object, ...]" = ()
    client_failures: "tuple[ResourceFailure, ...]" = ()
    runtime_closed: bool = False
    cleanup_error: "BaseException | None" = None
    interrupted: bool = False


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
    from linktools.ai.runtime.session import ResourceFailure

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
    lock: "ProjectProcessLock | None" = None
    lock_acquired = False
    bundle = None
    client = None
    agent = None
    result = _AcpLifecycleResult()
    try:
        logger.info("event=ai.acp.initialization_stage stage=lock_acquire")
        lock = ProjectProcessLock(project.state_root / "runtime.lock")
        try:
            lock.acquire(project_root=project.root)
        except RuntimeError as error:
            raise AcpDependencyError(str(error)) from error
        lock_acquired = True
        logger.info("event=ai.acp.initialization_stage stage=runtime_build")
        hub = ExecutionEventHub()
        bundle = build_cli_runtime(
            project=project,
            model_resolver=None,
            live_events=hub,
            require_tool_approval=True,
        )
        logger.info("event=ai.acp.initialization_stage stage=mode_resolution")
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
        logger.info("event=ai.acp.initialization_stage stage=client_construct")
        client = AcpClient(project_root=project.root)
        logger.info("event=ai.acp.initialization_stage stage=agent_construct")
        agent = LinktoolsAcpAgent(
            runtime=bundle.runtime,
            codec=AcpCodec(),
            client=client,
            protocol=protocol,
        )
        logger.info("event=ai.acp.initialization_stage stage=transport")
        await run_acp_server(agent)
    except asyncio.CancelledError:
        result.interrupted = True
        logger.warning("event=ai.acp.interrupted initialization_stage=transport")
    except AcpTransportError as error:
        result.transport_error = error
        logger.error(
            "event=ai.acp.transport_failed transport_error_id=%s",
            type(error).__name__,
            exc_info=environ.debug,
        )
    except BaseException as error:
        result.initialization_error = error
        logger.error(
            "event=ai.acp.initialization_failed initialization_stage=unknown error_id=%s",
            type(error).__name__,
            exc_info=environ.debug,
        )
    finally:
        if bundle is not None:
            try:
                preclose_results = await bundle.runtime.shutdown_sessions()
                result.preclose_failures = tuple(
                    failure
                    for close_result in preclose_results
                    for failure in close_result.failures
                )
            except BaseException as error:
                result.preclose_failures = (error,)
                logger.error(
                    "event=ai.acp.preclose_failed initialization_stage=preclose cleanup_failure_count=1 error_id=%s",
                    type(error).__name__,
                    exc_info=environ.debug,
                )
        if client is not None:
            try:
                result.client_failures = await client.close()
            except BaseException as error:
                result.client_failures = (
                    ResourceFailure("acp.client", None, type(error).__name__),
                )
                logger.error(
                    "event=ai.acp.client_close_failed cleanup_failure_count=1 error_id=%s",
                    type(error).__name__,
                    exc_info=environ.debug,
                )
        if bundle is not None:
            try:
                runtime_result = await bundle.runtime.aclose()
                result.runtime_closed = runtime_result.closed
                if not runtime_result.closed:
                    result.cleanup_error = RuntimeError("runtime_cleanup_failed")
            except BaseException as error:
                result.cleanup_error = error
                logger.error(
                    "event=ai.acp.runtime_close_failed runtime_closed=False error_id=%s",
                    type(error).__name__,
                    exc_info=environ.debug,
                )
        if lock_acquired and lock is not None:
            try:
                lock.release()
            except BaseException as error:
                result.cleanup_error = result.cleanup_error or error
                logger.error(
                    "event=ai.acp.lock_release_failed cleanup_failure_count=1 error_id=%s",
                    type(error).__name__,
                    exc_info=environ.debug,
                )
    cleanup_failure_count = (
        len(result.preclose_failures)
        + len(result.client_failures)
        + (1 if result.cleanup_error is not None else 0)
    )
    if result.interrupted:
        exit_code = 130
    elif cleanup_failure_count:
        exit_code = 4
    elif result.transport_error is not None:
        exit_code = 3
    elif isinstance(result.initialization_error, (AcpDependencyError, ConfigError)):
        exit_code = 3
    elif result.initialization_error is not None:
        exit_code = 10
    else:
        exit_code = 0
    logger.info(
        "event=ai.acp.shutdown_complete client_failure_count=%s runtime_closed=%s transport_error_id=%s cleanup_failure_count=%s exit_code=%s error_id=%s",
        len(result.client_failures),
        result.runtime_closed,
        type(result.transport_error).__name__ if result.transport_error is not None else None,
        cleanup_failure_count,
        exit_code,
        type(result.initialization_error).__name__ if result.initialization_error is not None else None,
    )
    return exit_code


command = Command()
