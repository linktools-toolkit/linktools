#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai session`: show a single session's messages."""

import asyncio
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError

from .assembly import project_storage
from .support import validate_session_id

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """show a session's messages"""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("session_id", help="session id")

    def run(self, args: "Namespace") -> "int | None":
        storage = project_storage()
        return asyncio.run(self._show_async(storage, args.session_id))

    async def _show_async(self, storage, session_id: str) -> "int | None":
        session_id = validate_session_id(session_id)
        record = await storage.sessions.get(session_id)
        if record is None:
            raise CommandError(f'session "{session_id}" not found')
        messages = await storage.sessions.list_messages(session_id)
        self.logger.info(
            f"session: {record.id} ({record.status.value}, {len(messages)} messages)"
        )
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else repr(msg.content)
            self.logger.info(f"[{msg.role.value}] {content}")
        return 0


command = Command()
