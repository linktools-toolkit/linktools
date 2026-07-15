#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai reject`: reject a pending approval request."""

import asyncio
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from .assembly import project_storage
from .support import resolve_approval

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """reject a pending request"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("approval_id", help="approval request id")
        parser.add_argument("--reason", default=None, help="rejection reason")

    def run(self, args: "Namespace") -> "int | None":
        storage = project_storage()
        return asyncio.run(
            resolve_approval(
                storage,
                args.approval_id,
                approved=False,
                reason=args.reason,
            )
        )


command = Command()
