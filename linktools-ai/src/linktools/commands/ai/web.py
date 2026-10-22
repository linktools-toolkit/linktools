#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai web`: serve a local Runtime console."""

import asyncio
from argparse import Namespace
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.core import environ
from ._common import (
    _add_local_runtime_arguments, _load_workspace, _local_metrics,
    _local_runtime_assets, _local_models, _local_model_options,
    _local_runtime_storage, _run_async,
)

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_logger = environ.get_logger("commands.ai.web")


class Command(BaseCommand):
    """Open one local Web console for conversations, executions, history and metrics."""

    def init_arguments(self, parser: "CommandParser") -> None:
        _add_local_runtime_arguments(parser)
        parser.add_argument("--port", type=int, default=8765, help="loopback HTTP port (default: 8765)")
        parser.add_argument("--open", action="store_true", help="open the console in your browser")

    def run(self, args: Namespace) -> int:
        try:
            import uvicorn
            from linktools.ai.web import create_app
        except ModuleNotFoundError as error:
            raise CommandError("ai web requires linktools-ai[web]") from error
        if not 1 <= args.port <= 65535:
            raise CommandError("port must be between 1 and 65535")
        workspace = _load_workspace(args.project)

        async def serve() -> int:
            from linktools.ai.capability import AssetCapability, WorkspaceCapability
            from linktools.ai.runtime import RuntimeFeatures
            from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
            from linktools.ai.runtime import Runtime

            model_options = _local_model_options(workspace, args)
            models = _local_models(workspace, **model_options)
            model_binding = models.capture().resolve("default")
            model_contract = model_binding.contract
            async with AsyncExitStack() as stack:
                assets = _local_runtime_assets(workspace)
                await assets.initialize()
                stack.push_async_callback(assets.close)
                features = RuntimeFeatures("workspace")
                features.capability(AssetCapability(assets))
                features.capability(WorkspaceCapability(workspace))
                features.capability(ExecutionCapability(retain_history=True))
                features.capability(CheckpointCapability())
                features.capability(TraceCapability())
                metrics = await _local_metrics(workspace)
                capture = await features.capture()
                declarations = [{"kind": item.kind, "id": item.id, "revision": item.revision} for item in capture.contributions]
                if not any(item["kind"] == "agent" and item["id"] == "default" for item in declarations):
                    declarations.insert(0, {"kind": "agent", "id": "default", "revision": 1})
                runtime = await stack.enter_async_context(Runtime.open(
                    "default", storage=_local_runtime_storage(workspace),
                    models=models, features=capture, metrics=metrics,
                ))
                status = {
                    "workspace": str(workspace.root),
                    "asset_root": str(workspace.storage_root),
                    "model": model_binding.model_identity.removeprefix(model_binding.provider + ":"),
                    "model_api": model_options["api"],
                    "input_mime_types": model_contract.get("input_mime_types"),
                    "base_url_configured": bool(model_options["base_url"]),
                    "api_key_configured": bool(model_options["api_key"]),
                }
                app = create_app(
                    runtime=runtime,
                    metrics=metrics,
                    status=status,
                    capabilities=declarations,
                    memory_scope=args.memory or "default",
                )
                url = f"http://127.0.0.1:{args.port}"
                _logger.info("AI Web console: %s", url)
                server = uvicorn.Server(uvicorn.Config(
                    app, host="127.0.0.1", port=args.port, log_config=None,
                    access_log=False, proxy_headers=False, timeout_graceful_shutdown=5,
                ))

                async def open_browser() -> None:
                    import webbrowser
                    while not server.started:
                        await asyncio.sleep(0.05)
                    await asyncio.to_thread(webbrowser.open, url)

                browser_task = asyncio.create_task(open_browser()) if args.open else None
                try:
                    await server.serve()
                finally:
                    if browser_task is not None:
                        browser_task.cancel()
                        await asyncio.gather(browser_task, return_exceptions=True)
            return 0

        return _run_async(serve())


command = Command()
