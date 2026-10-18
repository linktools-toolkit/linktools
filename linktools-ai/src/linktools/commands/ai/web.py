#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai web`: serve a local Runtime console."""

import asyncio
import os
from pathlib import Path
from argparse import Namespace
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.core import environ
from linktools.ai.observe import Metrics
from linktools.ai.workspace import Workspace
from ._common import (
    _add_local_runtime_arguments, _load_workspace, _local_input_config, _local_metrics,
    _local_runtime_assets, _local_runtime_models, _local_runtime_root,
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
        parser.add_argument("--read-only", action="store_true", help="browse persisted history without opening an execution Runtime")
        parser.add_argument("--proxy", action="store_true", help="accept external Host/Origin headers from a loopback reverse proxy; the proxy must authenticate users")
        parser.add_argument("--open", action="store_true", help="open the console in your browser")

    def run(self, args: Namespace) -> int:
        try:
            import uvicorn
            from linktools.ai.web import create_app
        except ModuleNotFoundError as error:
            raise CommandError("ai web requires linktools-ai[web]") from error
        if not 1 <= args.port <= 65535:
            raise CommandError("port must be between 1 and 65535")
        workspace = Workspace.discover(Path.cwd(), root=args.project) if args.read_only else _load_workspace(args.project)
        root = _local_runtime_root(workspace)
        model = args.model or workspace.config.get("model") or os.getenv("OPENAI_MODEL", "").strip()
        read_only = args.read_only or not model

        async def serve() -> int:
            from linktools.ai.runtime import RuntimeHistory

            _local_input_config(workspace)
            models = _local_runtime_models(workspace, args) if model else None
            # Show resolved MIME permissions, not the authoring category labels.
            input_mime_types = models.capture().resolve("default").contract.get("input_mime_types") if models is not None else None
            async with AsyncExitStack() as stack:
                runtime = None
                history = None
                metrics = Metrics.sqlite(root / "metrics.db", namespace="default") if (root / "metrics.db").exists() else None
                declarations = []
                if read_only:
                    if (root / "runtime.db").exists():
                        history = await stack.enter_async_context(RuntimeHistory.open("default", storage=_local_runtime_storage(workspace)))
                else:
                    from linktools.ai.capability import AssetCapability, RuntimeFeatures, WorkspaceCapability
                    from linktools.ai.capability import CheckpointCapability, ConversationCapability, TraceCapability
                    from linktools.ai.runtime import Runtime

                    assets = _local_runtime_assets(workspace)
                    await assets.initialize()
                    stack.push_async_callback(assets.close)
                    features = RuntimeFeatures("workspace")
                    features.capability(AssetCapability(assets))
                    features.capability(WorkspaceCapability(workspace))
                    features.capability(ConversationCapability())
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
                    "namespace": "default", "workspace": str(workspace.root), "asset_root": str(workspace.storage_root),
                    "runtime_db": {"path": str(root / "runtime.db"), "exists": (root / "runtime.db").exists()},
                    "object_store": {"path": str(root / "objects"), "exists": (root / "objects").exists()},
                    "metrics_db": {"path": str(root / "metrics.db"), "exists": (root / "metrics.db").exists()},
                    "model": model or None, "input_mime_types": input_mime_types,
                    "base_url_configured": bool(args.base_url or os.getenv("OPENAI_BASE_URL", "").strip()),
                    "api_key_configured": bool(args.api_key or os.getenv("OPENAI_API_KEY", "").strip()),
                }
                app = create_app(runtime=runtime, history=history, metrics=metrics, status=status,
                                 capabilities=declarations, memory_scope=args.memory or "default", port=args.port, proxy=args.proxy)
                url = f"http://127.0.0.1:{args.port}"
                _logger.info("AI Web console: %s%s", url, " (read-only)" if read_only else "")
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
