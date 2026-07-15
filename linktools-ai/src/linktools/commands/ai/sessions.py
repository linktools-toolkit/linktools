#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai sessions`: list all sessions."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand

from .assembly import project_storage

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """list the project's sessions"""

    def init_arguments(self, parser: "CommandParser") -> None:
        pass

    def run(self, args: "Namespace") -> "int | None":
        storage = project_storage()
        return asyncio.run(self._list_async(storage))

    async def _list_async(self, storage) -> "int | None":
        records = await _list_sessions(storage)
        if not records:
            self.logger.info("no sessions")
            return 0
        for rec in records:
            self.logger.info(
                f"{rec.id}\t{rec.status.value}\t{rec.updated_at.isoformat()}"
            )
        return 0


async def _list_sessions(storage) -> list:
    """Enumerate every session record under the project's storage by scanning
    the sessions directory. ``SessionStore`` exposes no ``list()``, so we glob
    the on-disk layout (``<root>/sessions/{id}/record.json``) and rehydrate each
    record through the store's own ``get()`` to keep deserialization encapsulated.
    """
    root = Path(storage.root) / "sessions"
    records = []
    for record_path in sorted(root.glob("*/record.json")):
        session_id = record_path.parent.name
        record = await storage.sessions.get(session_id)
        if record is not None:
            records.append(record)
    return records


command = Command()
