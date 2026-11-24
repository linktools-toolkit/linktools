#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local notices are useful hints, including when a Web peer disappears."""

import asyncio
import contextlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from linktools.commands.ai._session_notices import session_notice_publisher, session_notice_receiver

pytestmark = pytest.mark.skipif(os.name != "posix", reason="transport requires private POSIX descriptors")

_CHILD = '''
import asyncio, json, sys
from pathlib import Path
from linktools.commands.ai._session_notices import session_notice_receiver

async def main():
    if sys.argv[1] == "receive":
        def print_notice(notice):
            print(json.dumps(notice), flush=True)
        async with session_notice_receiver(Path(sys.argv[2]), print_notice) as ready:
            assert ready
            print("ready", flush=True)
            await asyncio.to_thread(sys.stdin.read)
        return

    async def receive(reader, writer):
        challenge = await reader.readexactly(16)
        if sys.argv[1] == "reused":
            writer.write(bytes(48))
            await writer.drain()
        tail = await reader.read()
        print(json.dumps([len(challenge), len(tail)]), flush=True)
        writer.close()

    listener = await asyncio.start_server(receive, "127.0.0.1", int(sys.argv[2]), reuse_address=True)
    async with listener:
        print("ready", flush=True)
        if sys.argv[2] == "0":
            print(listener.sockets[0].getsockname()[1], flush=True)
        await asyncio.to_thread(sys.stdin.read)

asyncio.run(main())
'''


async def _line(process: asyncio.subprocess.Process) -> bytes:
    assert process.stdout is not None
    result = await asyncio.wait_for(process.stdout.readline(), 5)
    assert result, (await asyncio.wait_for(process.stderr.read(), 2)).decode()
    return result.strip()


@contextlib.asynccontextmanager
async def _child(mode: str, argument: str):
    root = Path(__file__).resolve().parents[2]
    paths = [str(root / package / "src") for package in ("linktools", "linktools-ai")]
    if inherited := os.environ.get("PYTHONPATH"):
        paths.append(inherited)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", _CHILD, mode, argument,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await _line(process) == b"ready"
        yield process
    finally:
        if process.returncode is None:
            process.stdin.close()
            try:
                assert await asyncio.wait_for(process.wait(), 5) == 0
            except TimeoutError:
                process.kill()
                await process.wait()
                raise


@pytest.mark.asyncio
async def test_same_workspace_fans_out_to_two_independent_receivers_after_one_dies(tmp_path):
    accepted, tool, terminal = ({"kind": key} for key in ("accepted", "tool", "terminal"))
    async with (
        _child("receive", str(tmp_path)) as first,
        _child("receive", str(tmp_path)) as second,
        session_notice_publisher(tmp_path) as publish,
    ):
        assert first.pid != second.pid
        publish(accepted)
        publish(tool, False)
        assert [json.loads(await _line(first)) for _ in range(2)] == [accepted, tool]
        assert [json.loads(await _line(second)) for _ in range(2)] == [accepted, tool]
        first.kill()
        await first.wait()

        directory = tmp_path / "session-receivers"
        assert first.pid in {json.loads(path.read_bytes())[0] for path in directory.glob("*.json")}
        for index in range(140):
            orphan = directory / f"{index:032x}.json"
            orphan.write_text(json.dumps([first.pid, 7, "aa" * 32]))
            orphan.chmod(0o600)

        async with _child("receive", str(tmp_path)) as restarted:
            publish(terminal)
            assert json.loads(await _line(second)) == terminal
            assert json.loads(await _line(restarted)) == terminal
            assert len(tuple(directory.glob("*.json"))) == 2
        assert len(tuple(directory.glob("*.json"))) == 1
    assert not tuple(directory.glob("*.json"))


@pytest.mark.asyncio
async def test_reused_port_cannot_receive_secret_and_stalled_peer_cannot_erase_final_notice(tmp_path):
    notice = {"kind": "terminal", "key": "must-not-reach-reused-port"}
    directory = tmp_path / "session-receivers"
    async with _child("receive", str(tmp_path)) as receiver:
        descriptor, = directory.glob("*.json")
        _, port, key = json.loads(descriptor.read_bytes())
        receiver.kill()
        await receiver.wait()

    async with _child("reused", str(port)) as unauthorized:
        # Simulate PID reuse: no process-death proof, even though this socket is foreign.
        descriptor.write_text(json.dumps([unauthorized.pid, port, key]))
        async with session_notice_publisher(tmp_path) as publish:
            publish(notice)
            assert json.loads(await _line(unauthorized)) == [16, 0]
        assert descriptor.is_file()

    async with (
        _child("receive", str(tmp_path)) as healthy,
        _child("stalled", "0") as stalled,
    ):
        port = int(await _line(stalled))
        async with session_notice_publisher(tmp_path) as publish:
            publish({"kind": "accepted"})
            assert json.loads(await _line(healthy)) == {"kind": "accepted"}
            descriptor.write_text(json.dumps([stalled.pid, port, key]))
            descriptor.chmod(0o600)
            for index in range(24):
                publish({"kind": "tool", "sequence": index}, False)
            publish(notice)
        assert json.loads(await _line(healthy)) == notice
        assert json.loads(await _line(stalled)) == [16, 0]
        assert descriptor.is_file()


@pytest.mark.asyncio
async def test_registration_ignores_nonprivate_files_and_symbolic_links(tmp_path):
    notice = {"kind": "accepted"}
    directory = tmp_path / "session-receivers"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    received = []
    arrived = asyncio.Event()

    def observe(value):
        received.append(value)
        arrived.set()

    async with session_notice_receiver(tmp_path, observe) as ready:
        assert not ready
    directory.chmod(0o700)

    async with session_notice_receiver(tmp_path, observe) as ready:
        assert ready
        descriptor, = directory.glob("*.json")
        assert stat.S_IMODE(descriptor.stat().st_mode) == 0o600
        descriptor.chmod(0o644)
        async with session_notice_publisher(tmp_path) as publish:
            publish(notice)
        assert not received

        descriptor.chmod(0o600)
        outside = tmp_path / "otherwise-valid.json"
        descriptor.rename(outside)
        descriptor.symlink_to(outside)
        async with session_notice_publisher(tmp_path) as publish:
            publish(notice)
        assert not received
        assert outside.is_file()

        descriptor.unlink()
        outside.rename(descriptor)
        async with session_notice_publisher(tmp_path) as publish:
            publish({"invalid": float("nan")})
            publish({"too_large": "x" * 10_000})
            publish(notice)
        await asyncio.wait_for(arrived.wait(), 2)
        assert received == [notice]


@pytest.mark.asyncio
async def test_publisher_skips_its_own_descriptor_but_sends_to_same_process_peer(tmp_path, monkeypatch):
    from linktools.commands.ai import _session_notices

    attempted = []
    sent = _session_notices._send
    own = []
    peer = asyncio.Queue()

    async def record(target, frame):
        attempted.append(target[0])
        return await sent(target, frame)

    monkeypatch.setattr(_session_notices, "_send", record)
    async with (
        session_notice_receiver(tmp_path, own.append) as own_descriptor,
        session_notice_receiver(tmp_path, peer.put_nowait) as peer_descriptor,
        session_notice_publisher(tmp_path, exclude=own_descriptor) as publish,
    ):
        assert own_descriptor and peer_descriptor and own_descriptor != peer_descriptor
        peer_port = json.loads((tmp_path / "session-receivers" / peer_descriptor).read_bytes())[1]
        for kind in ("admitted", "tool"):
            message = {"kind": kind}
            publish(message, discover=kind != "tool")
            assert await asyncio.wait_for(peer.get(), 2) == message
    assert attempted == [peer_port, peer_port]
    assert not own


def test_local_session_feed_decodes_existing_tool_shape_without_forwarding_invalid_rows():
    from types import SimpleNamespace

    from linktools.ai.session import ToolProcessItem
    from linktools.commands.ai._common import _LocalSessionFeed

    invalidations = []
    feed = _LocalSessionFeed(SimpleNamespace(sessions=SimpleNamespace(invalidate=invalidations.append)))
    received = []
    unsubscribe = feed.subscribe("session", lambda execution, item: received.append((execution, item)))
    notice = {
        "kind": "tool", "session_id": "session", "execution_id": "execution",
        "process": {
            "kind": "tool", "key": "tool:abc", "tool_name": "search", "status": "SUCCEEDED", "position": [2, 0],
            "elapsed_ms": 7,
        },
    }
    for change in (
        {"key": "tool:"}, {"status": "UNCONFIRMED"}, {"kind": "other"},
        {"position": [True, 0]}, {"position": [2, -1]}, {"position": [2]}, {"position": "2:0"},
    ):
        feed.receive({**notice, "process": {**notice["process"], **change}})
    feed.receive(notice)
    feed.receive({**notice, "process": {**notice["process"], "position": None}})
    unsubscribe()
    feed.receive(notice)
    feed.receive({**notice, "kind": "terminal"})
    assert received == [
        ("execution", ToolProcessItem("tool:abc", "search", "SUCCEEDED", (2, 0), elapsed_ms=7)),
        ("execution", ToolProcessItem("tool:abc", "search", "SUCCEEDED", elapsed_ms=7)),
    ]
    assert invalidations == ["session"]
