#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guardian stdio relay stream guarantees."""

import io
import os
import selectors
import select
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from linktools.ai.workspace import sandbox_guardian


def test_guardian_status_option_does_not_reinterpret_argument_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = ["bwrap", "--setenv", "VALUE", "--", "--", "program", ""]
    observed: list[str] = []
    child = object()

    def capture_process(command: list[str], **kwargs: object) -> object:
        observed.extend(command)
        return child

    monkeypatch.setattr(sandbox_guardian.subprocess, "Popen", capture_process)
    result, status_fd = sandbox_guardian._start_bwrap(arguments)
    try:
        assert result is child
        assert observed[:2] == ["bwrap", "--json-status-fd"]
        assert observed[3:] == arguments[1:]
    finally:
        os.close(status_fd)


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
@pytest.mark.parametrize("mode", ("worker", "stdio"))
@pytest.mark.parametrize("tracked", (False, True))
def test_guardian_startup_failure_terminates_started_child(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    tracked: bool,
) -> None:
    config_read, config_write = os.pipe()
    control_read, control_write = os.pipe()
    runtime_pidfd = os.pidfd_open(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(sandbox_guardian, "_set_child_subreaper", lambda: None)
    monkeypatch.setattr(sandbox_guardian, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(sandbox_guardian, "_child_process_ids", lambda: ())
    monkeypatch.setattr(sandbox_guardian, "_read_config", lambda *args: {
        "version": sandbox_guardian.PROTOCOL_VERSION,
        "mode": mode,
        "bwrap_args": ["unused", "--", "unused"],
    })
    monkeypatch.setattr(sandbox_guardian, "_start_bwrap", lambda args: (child, -1))

    def fail_start(*args: object) -> None:
        raise OSError("startup failed")

    monkeypatch.setattr(sandbox_guardian, "_wait_bwrap_started", fail_start)
    if not tracked:
        monkeypatch.setattr(sandbox_guardian, "_open_child_pidfd", fail_start)
    arguments = ["--config-fd", str(config_read), "--runtime-pidfd", str(runtime_pidfd)]
    if mode == "stdio":
        arguments.extend(("--stdio-control-fd", str(control_write)))
    try:
        assert sandbox_guardian.main(arguments) == sandbox_guardian.GUARDIAN_EXIT_SESSION_FAILED
        assert child.poll() is not None
    finally:
        for fd in (config_read, config_write, control_read, control_write, runtime_pidfd):
            try:
                os.close(fd)
            except OSError:
                pass
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd is required")
def test_stdio_relay_backpressures_without_changing_stream_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_input, parent_writer = os.pipe()
    parent_reader, parent_output = os.pipe()
    status_read, status_write = os.pipe()
    runtime_pidfd = os.pidfd_open(os.getpid())
    child = subprocess.Popen(
        ["/bin/cat"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for name, mode in (("stdin", "wb"), ("stdout", "rb"), ("stderr", "rb")):
        stream = getattr(child, name)
        assert stream is not None
        fd = os.dup(stream.fileno())
        stream.close()
        setattr(child, name, io.FileIO(fd, mode, closefd=False))
    child_pidfd = os.pidfd_open(child.pid)
    selector = selectors.DefaultSelector()
    payload = bytes(range(256)) * (66 * 1024)
    received = bytearray()
    writer_errors: list[Exception] = []
    reader_errors: list[Exception] = []

    monkeypatch.setattr(
        sandbox_guardian,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(
                buffer=io.FileIO(parent_input, "rb", closefd=False)
            ),
            stdout=SimpleNamespace(
                buffer=io.FileIO(parent_output, "wb", closefd=False)
            ),
        ),
    )

    def write_input() -> None:
        view = memoryview(payload)
        try:
            while view:
                count = os.write(parent_writer, view)
                view = view[count:]
        except Exception as error:
            writer_errors.append(error)
        finally:
            os.close(parent_writer)

    def read_output() -> None:
        time.sleep(0.25)
        try:
            while True:
                readable, _, _ = select.select([parent_reader], [], [], 10)
                if not readable:
                    raise TimeoutError("guardian output stopped flowing")
                data = os.read(parent_reader, 64 * 1024)
                if not data:
                    return
                received.extend(data)
        except Exception as error:
            reader_errors.append(error)

    writer_thread = threading.Thread(target=write_input, daemon=True)
    reader_thread = threading.Thread(target=read_output, daemon=True)
    writer_thread.start()
    reader_thread.start()
    try:
        result = sandbox_guardian._relay_stdio(
            selector,
            runtime_pidfd,
            child,
            child_pidfd,
            status_read,
        )
        writer_thread.join(timeout=10)
        reader_thread.join(timeout=10)
        assert result == sandbox_guardian.GUARDIAN_EXIT_OK
        assert not writer_thread.is_alive()
        assert not reader_thread.is_alive()
        assert not writer_errors
        assert not reader_errors
        assert received == payload
    finally:
        selector.close()
        for fd in (
            parent_input,
            parent_writer,
            parent_reader,
            parent_output,
            status_read,
            status_write,
            runtime_pidfd,
            child_pidfd,
        ):
            try:
                os.close(fd)
            except OSError:
                pass
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        child.wait()
