#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guardian stdio relay stream guarantees."""

import io
import os
import selectors
import select
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from linktools.ai.workspace import sandbox_guardian


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
