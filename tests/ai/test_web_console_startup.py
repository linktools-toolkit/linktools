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
from argparse import Namespace
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from packaging.requirements import Requirement
import yaml


def test_sdk_dependency_requires_synthesized_tool_return_metadata() -> None:
    source_root = Path(__file__).parents[2]
    config = yaml.safe_load((source_root / "linktools-ai/linktools.yml").read_text())
    requirements = {item.name: item for item in map(Requirement, config["dependencies"])}
    sdk = requirements["pydantic-ai-slim"]
    assert "2.52.0" not in sdk.specifier
    assert "2.53.0" in sdk.specifier
    assert "3.0.0" not in sdk.specifier


def test_development_install_selects_declared_web_extra() -> None:
    from manage import _install_requirements, get_modules

    modules = get_modules()
    development = dict(_install_requirements(Namespace(module=["linktools-ai"], editable=True), modules))
    ordinary = dict(_install_requirements(Namespace(module=["linktools-ai"], editable=False), modules))
    assert development["linktools-ai"] == modules["linktools-ai"]["path"] + "[web]"
    assert ordinary["linktools-ai"] == modules["linktools-ai"]["path"]


@pytest.mark.parametrize(("read_only", "proxy"), [(True, False), (False, True)])
def test_web_cli_serves_http_and_exits_cleanly(tmp_path: Path, read_only: bool, proxy: bool) -> None:
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
    if read_only:
        arguments.append("--read-only")
    else:
        # Opening Runtime must not require a provider request or live credentials.
        arguments.extend(("--model", "startup-test", "--base-url", "http://127.0.0.1:1",
                          "--api-key", "offline-test-key"))
    if proxy:
        arguments.append("--proxy")
    headers = {"Host": "console.example", "Origin": "https://console.example", "Sec-Fetch-Site": "same-origin"} if proxy else {}
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
            assert config["read_only"] is read_only
            with opener.open(Request(f"http://127.0.0.1:{port}/", headers=headers), timeout=2) as response:
                assert response.status == 200
                assert b"<html" in response.read().lower()
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
            output.seek(0)
            log = output.read()
            assert process.returncode == 130, log
            assert "Application shutdown complete." in log
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
