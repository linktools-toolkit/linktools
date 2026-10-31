#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed local SQLite profiles and resource-owned filesystem write permission."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ._lock import FilesystemWriterLock

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

_LOCAL_PROFILE = 0x4C544C53
_CONNECTION_PROFILE = 'linktools.local_sqlite.profile'


@dataclass(frozen=True, slots=True)
class _SqliteFile:
    profile: int
    path: Path | None
    identity: tuple[int, int] | None


async def _connection_file(connection: 'AsyncConnection') -> _SqliteFile:
    cached = connection.info.get(_CONNECTION_PROFILE)
    if cached is not None:
        return cached
    rows = (await connection.exec_driver_sql('PRAGMA database_list')).all()
    main = next((str(row[2]) for row in rows if row[1] == 'main'), '')
    profile = int((await connection.exec_driver_sql('PRAGMA main.application_id')).scalar_one())
    path = Path(main).resolve() if main else None
    identity = None
    if profile == _LOCAL_PROFILE and path is not None:
        stat = path.stat()
        if stat.st_nlink != 1:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING, 'Dedicated SQLite files cannot use hard links')
        identity = stat.st_dev, stat.st_ino
    result = _SqliteFile(profile, path, identity)
    connection.info[_CONNECTION_PROFILE] = result
    return result


async def _finish_owned_io(task: 'asyncio.Task[None]') -> None:
    cancelled = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as error:
            cancelled = error
    try:
        task.result()
    except BaseException as error:
        if cancelled is not None:
            raise cancelled from error
        raise
    if cancelled is not None:
        raise cancelled


class _SqliteFileOwner:
    """Keep one dedicated local database writable until its resource scope closes.

    The profile is fixed before publishing a new database, never changed while
    a source is live. Replacing a live database or using hard links is unsupported.
    A borrowed engine does not inherit this permission.
    """

    def __init__(self, path: 'str | Path') -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = FilesystemWriterLock(self.path.with_name(self.path.name + '.writer.lock'))
        self._identity: tuple[int, int] | None = None
        self._closing = False

    @property
    def active(self) -> bool:
        return self._lock.acquired and self._identity is not None and not self._closing

    async def acquire(self) -> None:
        if self._closing:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        await _finish_owned_io(asyncio.create_task(self._lock.acquire()))

    def require_open(self) -> None:
        if self._closing or not self._lock.acquired:
            raise AIError(ErrorCode.STORAGE_CLOSED)

    def _require(self, source: _SqliteFile) -> None:
        self.require_open()
        if (source.profile != _LOCAL_PROFILE or source.path != self.path or source.identity is None
                or self._identity is not None and self._identity != source.identity):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._identity = source.identity

    def stop_writes(self) -> None:
        self._closing = True

    async def close(self) -> None:
        self.stop_writes()
        await _finish_owned_io(asyncio.create_task(self._lock.release()))
        self._identity = None


async def require_sqlite_write(connection: 'AsyncConnection', owner: _SqliteFileOwner | None = None) -> None:
    if connection.dialect.name != 'sqlite':
        if owner is not None:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        return
    source = await _connection_file(connection)
    if owner is not None:
        owner._require(source)
    elif source.profile == _LOCAL_PROFILE:
        raise AIError(ErrorCode.STORAGE_CONFLICT, 'Dedicated local SQLite requires its resource write owner')


__all__ = []
