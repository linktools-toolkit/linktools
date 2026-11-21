#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise the CLI execution imports and real loopback server lifecycle."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest


def test_web_cli_exits_cleanly_with_an_idle_observer(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    pytest.importorskip("uvicorn")
    source_root = Path(__file__).parents[2]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((
            str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"),
            environment.get("PYTHONPATH", ""),
        )),
        LINKTOOLS_PATH=str(tmp_path / "home"),
        DEBUG="false",
        PYDANTIC_AI_NO_BANNER="1",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    arguments = [
        sys.executable, "-m", "linktools", "ai", "web",
        "--project", str(tmp_path / "workspace"), "--port", str(port),
    ]
    # Opening Runtime must not require a provider request or live credentials.
    arguments.extend(("--model", "startup-test", "--base-url", "http://127.0.0.1:1",
                      "--api-key", "offline-test-key"))
    headers = {"Host": "console.example", "Origin": "https://console.example", "Sec-Fetch-Site": "same-origin"}
    opener = build_opener(ProxyHandler({}))
    with (tmp_path / "server.log").open("w+") as output:
        process = subprocess.Popen(arguments, cwd=tmp_path, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with opener.open(Request(f"http://127.0.0.1:{port}/api/config", headers=headers), timeout=1) as response:
                        config = json.load(response)
                    break
                except URLError:
                    time.sleep(0.05)
            else:
                output.seek(0)
                pytest.fail("Web CLI did not become ready:\n" + output.read())
            assert config["read_only"] is False
            with opener.open(Request(f"http://127.0.0.1:{port}/", headers=headers), timeout=2) as response:
                assert response.status == 200
                assert b"<html" in response.read().lower()
            with opener.open(Request(
                f"http://127.0.0.1:{port}/api/sessions",
                data=json.dumps({"session_id": "idle", "request_id": "create"}).encode(),
                headers={**headers, "Content-Type": "application/json", "X-Linktools-Console": "1"},
            ), timeout=2) as response:
                assert response.status == 201
            with opener.open(Request(f"http://127.0.0.1:{port}/api/session/events?session_id=idle",
                                     headers=headers), timeout=7) as stream:
                assert stream.headers.get_content_type() == "text/event-stream"
                assert stream.readline().startswith(b"data: ")
                assert stream.readline() == b"\n"
                process.send_signal(signal.SIGINT)
                assert stream.read() == b""
                process.wait(timeout=15)
            output.seek(0)
            log = output.read()
            assert process.returncode == 130, log
            assert "timeout graceful shutdown exceeded" not in log
            assert "Exception in ASGI application" not in log
            assert "Application shutdown complete." in log
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
