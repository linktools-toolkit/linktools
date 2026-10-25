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
import tomllib
from pathlib import Path
from argparse import Namespace
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from packaging.requirements import Requirement
import yaml


def test_sdk_dependencies_require_typed_errors_and_matching_optional_packages() -> None:
    source_root = Path(__file__).parents[2]
    config = yaml.safe_load((source_root / "linktools-ai/linktools.yml").read_text())
    requirements = {item.name: item for item in map(Requirement, config["dependencies"])}
    sdk = requirements["pydantic-ai-slim"]
    assert "2.58.0" not in sdk.specifier
    assert "2.59.0" in sdk.specifier
    assert "3.0.0" not in sdk.specifier
    harness = requirements["pydantic-ai-harness"]
    assert "0.58.0" not in harness.specifier
    assert "0.59.0" in harness.specifier
    for extra, sdk_extra in (("openai", "openai"), ("anthropic", "anthropic"), ("evaluation", "evals")):
        optional = {item.name: item for item in map(Requirement, config["optional-dependencies"][extra])}
        assert optional["pydantic-ai-slim"].specifier == sdk.specifier
        assert optional["pydantic-ai-slim"].extras == {sdk_extra}
        assert "pydantic-evals" not in optional


def test_ai_python_floor_leaves_other_package_platforms_unchanged() -> None:
    from manage import get_modules

    for name, module in get_modules().items():
        project = tomllib.loads((Path(module["path"]) / "pyproject.toml").read_text())["project"]
        assert project["requires-python"] == (">=3.11" if name == "linktools-ai" else ">=3.6")


def test_development_install_selects_declared_optional_extras() -> None:
    from manage import _install_requirements, get_modules

    modules = get_modules()
    development = dict(_install_requirements(Namespace(module=["linktools-ai"], editable=True), modules))
    ordinary = dict(_install_requirements(Namespace(module=["linktools-ai"], editable=False), modules))
    assert development["linktools-ai"] == modules["linktools-ai"]["path"] + "[anthropic,magic,openai,web]"
    assert ordinary["linktools-ai"] == modules["linktools-ai"]["path"]


def test_web_cli_serves_http_and_exits_cleanly(tmp_path: Path) -> None:
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
