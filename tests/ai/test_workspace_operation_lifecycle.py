#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace shutdown owns operations, independent of caller task lifetime."""

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from linktools.ai.errors import AIError
from linktools.ai.workspace import LocalSandbox, StdioSandboxSession
from linktools.ai.workspace import _bubblewrap
from linktools.ai.workspace._sandbox_protocol import PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_close_does_not_wait_for_caller_after_read_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value"
    target.write_bytes(b"content")
    entered = threading.Event()
    release = threading.Event()
    read_finished = asyncio.Event()
    caller_release = asyncio.Event()
    close_waiting = asyncio.Event()
    read_bytes = Path.read_bytes
    session = await LocalSandbox().open(root=tmp_path)
    wait_operations = session._wait_operations

    def delayed_read(path: Path) -> bytes:
        if path == target:
            entered.set()
            assert release.wait(5)
        return read_bytes(path)

    async def wait_for_operations() -> None:
        close_waiting.set()
        await wait_operations()

    async def caller() -> None:
        assert await session.read_bytes("value") == b"content"
        read_finished.set()
        await caller_release.wait()

    monkeypatch.setattr(Path, "read_bytes", delayed_read)
    monkeypatch.setattr(session, "_wait_operations", wait_for_operations)
    reading = asyncio.create_task(caller())
    closing: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(session.close())
        await asyncio.wait_for(close_waiting.wait(), 5)
        release.set()
        await asyncio.wait_for(read_finished.wait(), 5)
        done, _ = await asyncio.wait((closing,), timeout=2)
        assert closing in done
        closing.result()
        assert not reading.done()
    finally:
        release.set()
        caller_release.set()
        await reading
        await session.close()
        if closing is not None:
            await closing


@pytest.mark.asyncio
@pytest.mark.parametrize("stdio", (False, True))
async def test_close_waits_for_process_start_cleanup_not_caller_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio: bool,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    call_finished = asyncio.Event()
    caller_release = asyncio.Event()
    close_waiting = asyncio.Event()
    create_process = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    session = await LocalSandbox().open(root=tmp_path)
    assert isinstance(session, StdioSandboxSession)
    wait_starting = session._wait_operations

    async def delayed_start(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await create_process(*args, **kwargs)
        processes.append(process)
        started.set()
        await release.wait()
        return process

    async def wait_for_starting() -> None:
        close_waiting.set()
        await wait_starting()

    async def caller() -> None:
        with pytest.raises(AIError):
            if stdio:
                await session.open_stdio_process(
                    sys.executable, ("-c", "import time; time.sleep(60)")
                )
            else:
                await session.start_command("echo ready")
        call_finished.set()
        await caller_release.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_start)
    monkeypatch.setattr(session, "_wait_operations", wait_for_starting)
    starting = asyncio.create_task(caller())
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 5)
        closing = asyncio.create_task(session.close())
        await asyncio.wait_for(close_waiting.wait(), 5)
        release.set()
        await asyncio.wait_for(call_finished.wait(), 5)
        done, _ = await asyncio.wait((closing,), timeout=2)
        assert closing in done
        closing.result()
        assert not starting.done()
        assert all(process.returncode is not None for process in processes)
    finally:
        release.set()
        caller_release.set()
        await starting
        await session.close()
        if closing is not None:
            await closing


@pytest.mark.asyncio
@pytest.mark.parametrize("unread_output", (False, True))
async def test_stdio_close_terminates_child_with_full_unread_input(
    tmp_path: Path,
    unread_output: bool,
) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    assert isinstance(session, StdioSandboxSession)
    try:
        process = await session.open_stdio_process(
            sys.executable,
            ("-c", "import sys,time; print('r', end='', flush=True); "
             + ("sys.stdout.write('x' * (20 * 1024 * 1024)); sys.stdout.flush(); "
                if unread_output else "")
             + "time.sleep(60)"),
        )
        assert await asyncio.wait_for(process.read_stdout(1), 5) == b"r"
        if unread_output:
            async def wait_for_backpressure() -> None:
                while not process._process.stdout._paused:
                    await asyncio.sleep(0)
            await asyncio.wait_for(wait_for_backpressure(), 5)
        writing = asyncio.create_task(process.write_stdin(b"x" * (2 * 1024 * 1024)))
        await asyncio.sleep(0)
        closing = asyncio.create_task(process.close())
        try:
            done, _ = await asyncio.wait((closing,), timeout=5)
            assert closing in done
            closing.result()
        finally:
            if not closing.done():
                process._process.kill()
            await asyncio.gather(writing, closing, return_exceptions=True)
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stdio", (False, True))
async def test_repeated_cancel_during_local_start_waits_for_child_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    create_process = asyncio.create_subprocess_exec

    async def delayed_start(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        entered.set()
        await release.wait()
        process = await create_process(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_start)
    session = await LocalSandbox().open(root=tmp_path)
    assert isinstance(session, StdioSandboxSession)
    arguments = [sys.executable, "-c", "import time; time.sleep(60)"]
    if stdio:
        opening = asyncio.create_task(session.open_stdio_process(arguments[0], arguments[1:]))
    else:
        command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
        opening = asyncio.create_task(session.start_command(command))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        opening.cancel()
        await asyncio.sleep(0)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        closing = asyncio.create_task(session.close())
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        await asyncio.wait_for(closing, 10)
        assert processes and all(process.returncode is not None for process in processes)
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await session.close()
        if closing is not None:
            await closing


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Bubblewrap requires POSIX process support")
async def test_repeated_cancel_during_guardian_start_waits_for_child_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    create_process = asyncio.create_subprocess_exec

    async def delayed_start(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        entered.set()
        await release.wait()
        process = await create_process(
            sys.executable, "-c", "import time; time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(process)
        return process

    async def stop_child(process: asyncio.subprocess.Process) -> None:
        process.kill()
        await process.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_start)
    monkeypatch.setattr(_bubblewrap, "_abort_guardian", stop_child)
    opening = asyncio.create_task(_bubblewrap._spawn_guardian({"mode": "stdio"}, -1))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        opening.cancel()
        await asyncio.sleep(0)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert processes and all(process.returncode is not None for process in processes)
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
@pytest.mark.parametrize(
    ("ignore_term", "unread_output"),
    ((False, False), (True, False), (False, True)),
)
async def test_guardian_close_bypasses_buffered_stdin_and_proves_child_exit(
    ignore_term: bool,
    unread_output: bool,
) -> None:
    child_source = (
        "import os,signal,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print(os.getpid(), flush=True); "
        + ("os.write(1, b'x' * (20 * 1024 * 1024)); " if unread_output else "")
        + "time.sleep(60)"
    )
    guardian_source = (
        "import os,subprocess,sys\n"
        "from linktools.ai.workspace import sandbox_guardian as g\n"
        "def start(arguments):\n"
        "    reader, writer = os.pipe()\n"
        "    child = subprocess.Popen(\n"
        f"        [sys.executable, '-c', {child_source!r}],\n"
        "        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,\n"
        "        start_new_session=True)\n"
        "    return child, reader\n"
        "g._start_bwrap = start\n"
        "g._wait_bwrap_started = lambda *arguments: None\n"
        # This direct child has no descendants; namespace reaping is separately tested.
        "g._child_process_ids = lambda: ()\n"
        "raise SystemExit(g.main())\n"
    )
    runtime_pidfd = os.pidfd_open(os.getpid())
    config_read, config_write = os.pipe()
    control_read, control_write = os.pipe()
    guardian = await asyncio.create_subprocess_exec(
        sys.executable, "-c", guardian_source,
        "--config-fd", str(config_read),
        "--runtime-pidfd", str(runtime_pidfd),
        "--stdio-control-fd", str(control_write),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=(config_read, runtime_pidfd, control_write),
        start_new_session=True,
    )
    for fd in (config_read, control_write, runtime_pidfd):
        os.close(fd)
    os.write(config_write, json.dumps({
        "version": PROTOCOL_VERSION,
        "mode": "stdio",
        "bwrap_args": ["unused", "--", "unused"],
    }).encode())
    os.close(config_write)
    os.set_blocking(control_read, False)
    process = None
    writing: asyncio.Task[None] | None = None
    reading: asyncio.Task[bytes] | None = None
    child_pid: int | None = None
    try:
        await _bubblewrap._wait_stdio_ready(guardian, control_read)
        process = _bubblewrap._BubblewrapStdioProcess(
            guardian, control_fd=control_read, on_close=lambda value: None
        )
        assert guardian.stdout is not None
        child_pid = int(await asyncio.wait_for(guardian.stdout.readline(), 5))
        if unread_output:
            async def wait_for_backpressure() -> None:
                while not guardian.stdout._paused:
                    await asyncio.sleep(0)
            await asyncio.wait_for(wait_for_backpressure(), 5)
        else:
            reading = asyncio.create_task(process.read_stdout())
        writing = asyncio.create_task(process.write_stdin(b"x" * (20 * 1024 * 1024)))
        await asyncio.sleep(0)
        assert not writing.done()
        await process.close()
        assert guardian.returncode == 0
        if reading is not None:
            assert await reading == b""
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if guardian.returncode is None:
            guardian.kill()
        if process is not None:
            await process._drain_stdout()
        await guardian.wait()
        if writing is not None:
            await asyncio.gather(writing, return_exceptions=True)
        if process is not None:
            process._stderr_task.cancel()
            await asyncio.gather(process._stderr_task, return_exceptions=True)
        if reading is not None:
            await asyncio.gather(reading, return_exceptions=True)
        try:
            os.close(control_read)
        except OSError:
            pass
