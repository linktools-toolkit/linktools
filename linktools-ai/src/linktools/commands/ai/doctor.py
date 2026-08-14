#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai doctor`: inspect a local project using the local boundary."""

import json
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from ...ai.workspace import Workspace

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """validate local project and Skill configuration"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=Path.cwd(), help="project root")
        parser.add_argument("--json", action="store_true", help="emit JSON")

    def run(self, args: Namespace) -> int:
        workspace = Workspace.load(args.project)
        report = {
            "root": str(workspace.root),
            "workspace_id": workspace.workspace_id,
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            self.logger.info("workspace root=%s", report["root"])
            self.logger.info("workspace id=%s", report["workspace_id"])
        return 0


command = Command()
