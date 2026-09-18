#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai status`: show local AI Runtime composition status."""

import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from rich import get_console
from rich.panel import Panel
from rich.table import Table

from linktools.cli import BaseCommand

from ._common import _load_workspace, _local_runtime_root

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """Show local Workspace, persistence, model, and metrics configuration."""

    def init_arguments(self, parser: "CommandParser") -> None:
        del parser

    def run(self, args: Namespace) -> int:
        del args
        workspace = _load_workspace()
        runtime_root = _local_runtime_root(workspace)
        model = workspace.config.get("model")
        model_name = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else os.getenv("OPENAI_MODEL", "").strip() or "-"
        )

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Workspace ID", workspace.workspace_id)
        table.add_row("Workspace Root", str(workspace.root))
        table.add_row("Asset Root", str(workspace.storage_root))
        table.add_row("Runtime DB", _path_state(runtime_root / "runtime.db"))
        table.add_row("Object Store", _path_state(runtime_root / "objects"))
        table.add_row("Metrics DB", _path_state(runtime_root / "metrics.db"))
        table.add_row("Model", model_name)
        table.add_row("Base URL", _configured("OPENAI_BASE_URL"))
        table.add_row("API Key", _configured("OPENAI_API_KEY"))
        table.add_row(
            "Vision",
            os.getenv("OPENAI_VISION", "").strip() or "default",
        )
        get_console().print(Panel(table, title="AI Runtime Status", expand=False))
        return 0


def _path_state(path: Path) -> str:
    state = "exists" if path.exists() else "missing"
    return f"{path} [{state}]"


def _configured(name: str) -> str:
    return "configured" if os.getenv(name, "").strip() else "-"


command = Command()
