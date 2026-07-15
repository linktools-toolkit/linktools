#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai approve`: approve a pending approval request."""

import asyncio
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from .assembly import project_storage
from .support import resolve_approval

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """approve a pending request"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("approval_id", help="approval request id")

    def run(self, args: "Namespace") -> "int | None":
        storage = project_storage()
        return asyncio.run(
            resolve_approval(storage, args.approval_id, approved=True, reason=None)
        )


command = Command()
