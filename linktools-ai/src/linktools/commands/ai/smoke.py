#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai smoke`: verify the explicit ACP boundary can be imported."""

import json
from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from ...ai.inbound.acp import ACPServer

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """run a boundary-only ACP smoke check"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--json", action="store_true", help="emit JSON")

    def run(self, args: Namespace) -> int:
        report = {"ok": True, "boundary": ACPServer.__name__, "application_required": True}
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            self.logger.info("ACP boundary import succeeded; application wiring is required")
        return 0


command = Command()
