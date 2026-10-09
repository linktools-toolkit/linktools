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
from linktools.ai.capability import CapabilityGroup
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeHistory
from linktools.ai.workspace import Workspace
from ._common import (
    _add_local_runtime_arguments, _load_workspace, _local_metrics,
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
            async with AsyncExitStack() as stack:
                runtime = None
                history = None
                metrics = Metrics.sqlite(root / "metrics.db", namespace="default") if (root / "metrics.db").exists() else None
                declarations = []
                if read_only:
                    if (root / "runtime.db").exists():
                        history = await stack.enter_async_context(RuntimeHistory.open("default", storage=_local_runtime_storage(workspace)))
                else:
                    assets = _local_runtime_assets(workspace)
                    await assets.initialize()
                    stack.push_async_callback(assets.close)
                    capture = await CapabilityGroup("workspace", workspace=workspace, assets=assets).capture()
                    declarations = [{"kind": item.kind, "id": item.id, "revision": item.revision} for item in capture.contributions]
                    if not any(item["kind"] == "agent" and item["id"] == "default" for item in declarations):
                        declarations.insert(0, {"kind": "agent", "id": "default", "revision": 1})
                    metrics = await _local_metrics(workspace)
                    runtime = await stack.enter_async_context(Runtime.open(
                        "default", storage=_local_runtime_storage(workspace), metrics=metrics,
                        models=_local_runtime_models(workspace, args), capabilities=(capture,),
                    ))
                status = {
                    "namespace": "default", "workspace": str(workspace.root), "asset_root": str(workspace.storage_root),
                    "runtime_db": {"path": str(root / "runtime.db"), "exists": (root / "runtime.db").exists()},
                    "object_store": {"path": str(root / "objects"), "exists": (root / "objects").exists()},
                    "metrics_db": {"path": str(root / "metrics.db"), "exists": (root / "metrics.db").exists()},
                    "model": model or None,
                    "vision": args.vision if args.vision is not None else os.getenv("OPENAI_VISION", "").strip() or "default",
                    "base_url_configured": bool(args.base_url or os.getenv("OPENAI_BASE_URL", "").strip()),
                    "api_key_configured": bool(args.api_key or os.getenv("OPENAI_API_KEY", "").strip()),
                }
                app = create_app(runtime=runtime, history=history, metrics=metrics, status=status,
                                 capabilities=declarations, memory_scope=args.memory or "default", port=args.port)
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
