#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai doctor`: inspect a local project using the v8 local boundary."""

import json
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from ...ai.local import LocalProject, SkillIndex

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """validate local project and Skill configuration"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=Path.cwd(), help="project root")
        parser.add_argument("--json", action="store_true", help="emit JSON")

    def run(self, args: Namespace) -> int:
        project = LocalProject.load(args.project)
        index = SkillIndex(project.root)
        revision = index.refresh()
        report = {
            "root": str(project.root),
            "project_id": project.project_id,
            "config": str(project.config_path) if project.config_path else None,
            "skill_revision": revision,
            "skills": [skill.skill_id for skill in index.list()],
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            self.logger.info("project root=%s", report["root"])
            self.logger.info("project id=%s", report["project_id"])
            self.logger.info("skills=%s", ", ".join(report["skills"]) or "none")
        return 0


command = Command()
