#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-effort notices within one user and PID namespace; not ownership state."""

import asyncio
import contextlib
import hmac
import itertools
import json
import os
import re
import secrets
import stat
import tempfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

_HOST = "127.0.0.1"
_DIRECTORY = "session-receivers"
_TIMEOUT = 0.25
_DRAIN_TIMEOUT = 2 * _TIMEOUT
_MAX_FRAME = 2048
_MAX_PENDING = 16
_MAX_PEERS = 16
_MAX_SCAN = 128
_NONCE = 16
_SECRET = 32


class SessionNoticeOffer(Protocol):
    def __call__(self, notice: dict[str, object], discover: bool = True) -> None: ...


def _directory(root: Path, *, create: bool = False) -> Path | None:
    # The descriptor contains a credential. Do not assume POSIX chmod implements Windows ACLs.
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        return None
    path = root / _DIRECTORY
    try:
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077:
            return path
    except OSError:
        pass
    return None


def _register(directory: Path, port: int, secret: bytes) -> tuple[Path, os.stat_result]:
    path = directory / (secrets.token_hex(16) + ".json")
    fd, temporary = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump([os.getpid(), port, secret.hex()], handle)
            info = os.fstat(handle.fileno())
        os.replace(temporary, path)
        return path, info
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _private_file(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077


def _unlink_descriptor(path: Path, original: os.stat_result) -> bool:
    try:
        info = path.lstat()
        if _private_file(info) and (info.st_dev, info.st_ino) == (original.st_dev, original.st_ino):
            path.unlink()
            return True
    except OSError:
        pass
    return False


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass
    # A reused PID may keep an orphan, but never treat a failed connection as proof of death.
    return False


def _discover(root: Path) -> list[tuple[int, bytes]]:
    directory = _directory(root)
    if directory is None:
        return []
    peers: dict[str, tuple[int, int, bytes]] = {}
    try:
        for _ in range(2):
            removed = False
            with os.scandir(directory) as entries:
                for entry in itertools.islice(entries, _MAX_SCAN):
                    if not re.fullmatch(r"[0-9a-f]{32}\.json", entry.name):
                        continue
                    try:
                        fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with os.fdopen(fd, "rb") as handle:
                            info = os.fstat(handle.fileno())
                            if not _private_file(info) or info.st_size > 128:
                                continue
                            pid, port, key = json.loads(handle.read(129))
                            secret = bytes.fromhex(key)
                            if not (type(pid) is int and 0 < pid < 2**31
                                    and type(port) is int and 0 < port < 65536 and len(secret) == _SECRET):
                                continue
                            if _process_gone(pid):
                                removed |= _unlink_descriptor(Path(entry.path), info)
                            else:
                                peers[entry.name] = (info.st_mtime_ns, port, secret)
                    except (OSError, TypeError, ValueError, RecursionError):
                        continue
            if not removed or len(peers) >= _MAX_PEERS:
                break
    except OSError:
        pass
    return [(port, secret) for _, port, secret in sorted(peers.values(), reverse=True)[:_MAX_PEERS]]


def _invalid_constant(value: str) -> object:
    raise ValueError("invalid JSON number")


def _proof(secret: bytes, role: bytes, client: bytes, server: bytes) -> bytes:
    return hmac.digest(secret, b"linktools-local-notices\0" + role + client + server, "sha256")


@contextlib.asynccontextmanager
async def session_notice_receiver(
    runtime_root: Path, on_notice: "Callable[[dict[str, object]], None]",
) -> "AsyncIterator[bool]":
    """Receive local notices with a synchronous callback that only schedules work."""
    directory = _directory(runtime_root, create=True)
    if directory is None:
        yield False
        return
    secret = secrets.token_bytes(_SECRET)
    active: set[asyncio.Task[None]] = set()

    async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            async with asyncio.timeout(_TIMEOUT):
                client = await reader.readexactly(_NONCE)
                server = secrets.token_bytes(_NONCE)
                writer.write(server + _proof(secret, b"receive", client, server))
                await writer.drain()
                if not hmac.compare_digest(await reader.readexactly(_SECRET), _proof(secret, b"send", client, server)):
                    return
                writer.write(b"\x01")
                await writer.drain()
                length = int.from_bytes(await reader.readexactly(2), "big")
                if not 0 < length <= _MAX_FRAME:
                    return
                notice = json.loads(await reader.readexactly(length), parse_constant=_invalid_constant)
                if isinstance(notice, dict):
                    with contextlib.suppress(Exception):
                        on_notice(notice)
        except (OSError, asyncio.IncompleteReadError, TimeoutError, ValueError, RecursionError):
            pass
        finally:
            writer.close()

    def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(active) >= _MAX_PEERS:
            writer.close()
            return
        task = asyncio.create_task(receive(reader, writer))
        active.add(task)
        task.add_done_callback(active.discard)

    listener = None
    descriptor = None
    try:
        try:
            listener = await asyncio.start_server(connected, _HOST, 0, limit=_MAX_FRAME, backlog=_MAX_PEERS)
            descriptor = _register(directory, listener.sockets[0].getsockname()[1], secret)
        except OSError:
            if listener is not None:
                listener.close()
                await listener.wait_closed()
                listener = None
        yield descriptor is not None
    finally:
        if descriptor is not None:
            _unlink_descriptor(*descriptor)
        if listener is not None:
            listener.close()
            await listener.wait_closed()
        for task in tuple(active):
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def _send(peer: tuple[int, bytes], frame: bytes) -> bool:
    port, secret = peer
    writer = None
    try:
        async with asyncio.timeout(_TIMEOUT):
            reader, writer = await asyncio.open_connection(_HOST, port, limit=_MAX_FRAME)
            client = secrets.token_bytes(_NONCE)
            writer.write(client)
            await writer.drain()
            handshake = await reader.readexactly(_NONCE + _SECRET)
            server, proof = handshake[:_NONCE], handshake[_NONCE:]
            if not hmac.compare_digest(proof, _proof(secret, b"receive", client, server)):
                return False
            writer.write(_proof(secret, b"send", client, server))
            await writer.drain()
            if await reader.readexactly(1) != b"\x01":
                return False
            writer.write(frame)
            await writer.drain()
            return True
    except (OSError, asyncio.IncompleteReadError, TimeoutError):
        return False
    finally:
        if writer is not None:
            writer.close()


@contextlib.asynccontextmanager
async def session_notice_publisher(
    runtime_root: Path,
) -> "AsyncIterator[SessionNoticeOffer]":
    """Yield a nonblocking offer callback; only lifecycle notices discover peers."""
    pending: deque[tuple[bytes, bool]] = deque()
    ready = asyncio.Event()
    closing = False
    inflight: asyncio.Task[None] | None = None
    inflight_lifecycle = False
    peers: list[tuple[int, bytes]] = []

    def publish(notice: dict[str, object], discover: bool = True) -> None:
        if closing or os.name != "posix" or not isinstance(notice, dict):
            return
        try:
            payload = json.dumps(notice, allow_nan=False, separators=(",", ":")).encode()
        except (TypeError, ValueError, RecursionError):
            return
        if len(payload) > _MAX_FRAME:
            return
        if len(pending) >= _MAX_PENDING:
            if not discover:
                return
            for index, (_, lifecycle) in enumerate(pending):
                if not lifecycle:
                    del pending[index]
                    break
            else:
                pending.popleft()
        pending.append((len(payload).to_bytes(2, "big") + payload, discover))
        ready.set()

    async def send(frame: bytes, discover: bool) -> None:
        nonlocal peers
        if discover:
            peers = await asyncio.to_thread(_discover, runtime_root)
        delivered = await asyncio.gather(*(_send(peer, frame) for peer in peers))
        peers = [peer for peer, sent in zip(peers, delivered) if sent]

    async def send_pending() -> None:
        nonlocal inflight, inflight_lifecycle
        while not closing or pending:
            if not pending:
                ready.clear()
                await ready.wait()
                continue
            frame, inflight_lifecycle = pending.popleft()
            inflight = asyncio.create_task(send(frame, inflight_lifecycle))
            await asyncio.gather(inflight, return_exceptions=True)
            inflight = None

    task = asyncio.create_task(send_pending())
    try:
        yield publish
    finally:
        closing = True
        pending = deque((frame, lifecycle) for frame, lifecycle in pending if lifecycle)
        if inflight is not None and not inflight_lifecycle:
            inflight.cancel()
        ready.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(task, _DRAIN_TIMEOUT)


__all__ = ["SessionNoticeOffer", "session_notice_publisher", "session_notice_receiver"]
