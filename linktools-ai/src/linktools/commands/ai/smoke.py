#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai smoke`: construct the local ACP application boundary."""

import json
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from ...ai.entry.acp import ACPApplication
from ...ai.local import LocalProject

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """construct and validate the local ACP application"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument("--storage", type=Path, default=None, help="runtime storage directory")
        parser.add_argument("--json", action="store_true", help="emit JSON")

    def run(self, args: Namespace) -> int:
        project = LocalProject.discover(Path.cwd(), root=args.project, storage_root=args.storage)
        application = ACPApplication.for_project(project)
        report = {
            "ok": application.agent().__class__.__name__ == "LocalACPAgent",
            "boundary": ACPApplication.__name__,
            "project_root": project.root.as_posix(),
        }
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            self.logger.info("ACP application ready: project=%s", project.root)
        return 0


command = Command()
