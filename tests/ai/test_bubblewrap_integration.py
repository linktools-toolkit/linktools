#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Linux Bubblewrap acceptance coverage."""

import asyncio
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath

import pytest
from fastmcp import Client
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._mcp import (
    _MCPDiscoveryToolset,
    _MCPModelToolset,
    _model_tool_name,
)
from linktools.ai.runtime._mcp_transport import (
    _SandboxMCPTransport,
    _close_process,
)
from linktools.ai.workspace import (
    BubblewrapSandbox,
    ReadOnlySandboxPolicy,
    SandboxResource,
    SandboxResourcePath,
)
from linktools.ai.workspace import _bubblewrap

pytestmark = pytest.mark.asyncio


def _sandbox_configuration() -> tuple[Path, Path]:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")
    return Path(runtime_root_value), Path(executable_value)


def _create_stdio_server(root: Path) -> SandboxResource:
    root.mkdir()
    (root / "resource.txt").write_text("resource", encoding="utf-8")
    (root / "server.py").write_text(
        textwrap.dedent(
            '''\
            import json
            import os
            import pathlib
            import socket
            import subprocess
            import sys
            import time

            def attempt(callback):
                try:
                    callback()
                    return "allowed"
                except OSError:
                    return "blocked"

            def probe(root, external, parent, port, resource):
                root = pathlib.Path(root)

                def append(path):
                    with pathlib.Path(path).open("a") as stream:
                        stream.write("x")

                def connect():
                    with socket.create_connection(
                        ("127.0.0.1", int(port)), timeout=1
                    ):
                        pass

                return {
                    "read": attempt(lambda: (root / "read.txt").read_text()),
                    "create": attempt(
                        lambda: (root / "created.txt").write_text("created")
                    ),
                    "modify": attempt(lambda: append(root / "modify.txt")),
                    "delete": attempt(lambda: (root / "delete.txt").unlink()),
                    "rename": attempt(
                        lambda: os.replace(root / "rename.txt", root / "renamed.txt")
                    ),
                    "chmod": attempt(lambda: os.chmod(root / "mode.txt", 0o600)),
                    "absolute": attempt(
                        lambda: pathlib.Path(external).read_text()
                    ),
                    "parent": attempt(
                        lambda: pathlib.Path(parent).read_text()
                    ),
                    "symlink": attempt(lambda: (root / "escape").read_text()),
                    "resource": attempt(lambda: append(resource)),
                    "network": attempt(connect),
                }

            def send(message):
                data = json.dumps(
                    message, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8") + b"\\n"
                for offset in range(0, len(data), 7):
                    os.write(1, data[offset : offset + 7])

            if len(sys.argv) > 1 and sys.argv[1] == "--probe":
                result = probe(
                    sys.argv[2],
                    sys.argv[3],
                    sys.argv[4],
                    sys.argv[5],
                    sys.argv[6],
                )
                print(json.dumps(result))
                raise SystemExit(0)

            external = sys.argv[1]
            port = sys.argv[2]
            parent = sys.argv[3]
            mode = sys.argv[4] if len(sys.argv) > 4 else ""
            resource = pathlib.Path(__file__).with_name("resource.txt")

            def spawn_descendant(delay, filename):
                child = (
                    "import pathlib,time; "
                    f"time.sleep({delay}); "
                    f"pathlib.Path('/workspace/{filename}')"
                    ".write_text('alive')"
                )
                subprocess.Popen(
                    [sys.executable, "-c", child],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )

            if mode == "spawn-on-start":
                pathlib.Path("/workspace/spawn-ready.txt").write_text("ready")
                spawn_descendant(2, "spawn-alive.txt")

            for line in sys.stdin.buffer:
                request = json.loads(line)
                method = request.get("method")
                if "id" not in request:
                    continue
                if method == "initialize":
                    if mode == "hang-init":
                        time.sleep(60)
                    result = {
                        "protocolVersion": request["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "acceptance", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {
                        "tools": [{
                            "name": "probe",
                            "description": "Exercise stdio isolation",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "notify": {"type": "boolean"},
                                    "spawn": {"type": "boolean"},
                                    "hang": {"type": "boolean"},
                                },
                            },
                        }]
                    }
                elif method == "tools/call":
                    arguments = request["params"].get("arguments", {})
                    if arguments.get("hang"):
                        time.sleep(3)
                        pathlib.Path("/workspace/call-alive.txt").write_text(
                            "alive"
                        )
                    if arguments.get("notify"):
                        for value in ("通知一", "通知🙂"):
                            send({
                                "jsonrpc": "2.0",
                                "method": "notifications/message",
                                "params": {
                                    "level": "info",
                                    "logger": "acceptance",
                                    "data": value,
                                },
                            })
                    if arguments.get("spawn"):
                        delay = 6 if mode == "spawn-close-child" else 2
                        spawn_descendant(
                            delay,
                            "detached.txt",
                        )
                    if arguments.get("flood"):
                        os.write(2, b"x" * 1048576)
                    observations = probe(
                        "/workspace",
                        external,
                        parent,
                        port,
                        resource,
                    )
                    content = [
                        {"type": "text", "text": json.dumps(observations)},
                        {"type": "text", "text": arguments.get("value", "")},
                    ]
                    result = {"content": content, "isError": False}
                else:
                    continue
                send({"jsonrpc": "2.0", "id": request["id"], "result": result})
            if mode in {"hang-close", "spawn-close-child"}:
                time.sleep(60)
            '''
        ),
        encoding="utf-8",
    )
    return SandboxResource("mcp", root)


def _prepare_probe_tree(root: Path, external: Path) -> None:
    root.mkdir(parents=True)
    for name in (
        "read.txt",
        "modify.txt",
        "delete.txt",
        "rename.txt",
        "mode.txt",
    ):
        (root / name).write_text(name, encoding="utf-8")
    (root / "escape").symlink_to(external)


def _runtime_mcp_client(
    transport: _SandboxMCPTransport,
    *,
    init_timeout: float | None = None,
) -> tuple[Client, _MCPModelToolset, RunContext[None], str]:
    server_id = "security/audit"
    client = Client(transport, init_timeout=init_timeout)
    discovery = _MCPDiscoveryToolset(
        client,
        id=f"mcp:{server_id}",
        cache_tools=True,
    )
    toolset = _MCPModelToolset(
        discovery,
        server_id,
        frozenset({"probe"}),
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="stdio-acceptance",
        tool_call_id="stdio-acceptance",
    )
    return client, toolset, context, _model_tool_name(server_id, "probe")


class _LocalStdioProcess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    async def write_stdin(self, data: bytes) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        stdin.write(data)
        await stdin.drain()

    async def read_stdout(self, max_bytes: int = 65536) -> bytes:
        stdout = self._process.stdout
        assert stdout is not None
        return await stdout.read(max_bytes)

    async def close_stdin(self) -> None:
        stdin = self._process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()
            await stdin.wait_closed()

    async def close(self) -> None:
        await self.close_stdin()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1)
            return
        except asyncio.TimeoutError:
            self._process.terminate()
        await self._process.wait()


class _LocalStdioSession:
    def __init__(self) -> None:
        self.processes: list[_LocalStdioProcess] = []

    async def open_stdio_process(
        self,
        command: str,
        args: tuple[str | SandboxResourcePath, ...],
        *,
        resources: tuple[SandboxResource, ...],
    ) -> _LocalStdioProcess:
        assert command == "/usr/bin/python3"
        roots = {resource.id: resource.source for resource in resources}
        command_args = []
        for argument in args:
            if isinstance(argument, str):
                command_args.append(argument)
                continue
            resource_root = roots[argument.resource_id]
            command_args.append(
                str(resource_root.joinpath(*PurePosixPath(argument.path).parts))
            )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *command_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        value = _LocalStdioProcess(process)
        self.processes.append(value)
        return value


async def test_mcp_transport_stops_message_pumps_before_process_close() -> None:
    events: list[str] = []

    class _Scope:
        def __init__(self, name: str) -> None:
            self._name = name

        def cancel(self) -> None:
            events.append(self._name)

    class _Event:
        def __init__(self, name: str) -> None:
            self._name = name

        async def wait(self) -> None:
            events.append(self._name)

    class _Stream:
        def __init__(self, name: str) -> None:
            self._name = name

        async def aclose(self) -> None:
            events.append(self._name)

    class _Process:
        async def close_stdin(self) -> None:
            events.append("stdin_closed")

        async def read_stdout(self, max_bytes: int = 65536) -> bytes:
            events.append("stdout_drained")
            return b""

        async def close(self) -> None:
            events.append("process_closed")

    await _close_process(
        _Process(),
        _Scope("reader_cancelled"),
        _Scope("writer_cancelled"),
        _Event("reader_stopped"),
        _Event("writer_stopped"),
        _Stream("read_sender_closed"),
        _Stream("read_receiver_closed"),
        _Stream("write_sender_closed"),
        _Stream("write_receiver_closed"),
    )

    assert events.index("reader_stopped") < events.index("process_closed")
    assert events.index("writer_stopped") < events.index("process_closed")
    assert events.index("stdin_closed") < events.index("process_closed")


async def test_mcp_transport_discovers_and_routes_through_fastmcp(
    tmp_path: Path,
) -> None:
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    transport = _SandboxMCPTransport(
        _LocalStdioSession(),
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            "missing",
            "9",
            "missing",
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    value = "审计🙂" * 1_024

    async with client:
        tools = await toolset.get_tools(context)
        assert tuple(tools) == (model_name,)
        assert tools[model_name].tool_def.name == model_name
        result = await toolset.call_tool(
            model_name,
            {"value": value, "notify": True},
            context,
            tools[model_name],
        )

    assert result[1] == value


async def test_mcp_transport_cancellation_during_handshake_closes_process(
    tmp_path: Path,
) -> None:
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    local_session = _LocalStdioSession()
    transport = _SandboxMCPTransport(
        local_session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            "missing",
            "9",
            "missing",
            "hang-init",
        ),
        (resource,),
    )
    client, _, _, _ = _runtime_mcp_client(transport, init_timeout=30)
    enter = asyncio.create_task(client.__aenter__())
    await asyncio.sleep(0.2)
    enter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await enter

    assert local_session.processes
    assert local_session.processes[0]._process.returncode is not None


async def test_bubblewrap_session_is_real_and_shares_workspace(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    hidden = workspace_root / ".linktools"
    hidden.mkdir()
    (hidden / "host-control.txt").write_text("host-only", encoding="utf-8")
    skill_root = tmp_path / "skill"
    skill_root.mkdir()
    (skill_root / "resource.txt").write_text("read-only", encoding="utf-8")

    sandbox = BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    )
    session = await sandbox.open(
        root=workspace_root,
        resources=(SandboxResource("skill-resource", skill_root),),
    )
    try:
        command_result = await session.run_command(
            "printf 'created-by-worker' > worker.txt"
        )
        assert "status: exited" in command_result
        assert (workspace_root / "worker.txt").read_text(encoding="utf-8") == (
            "created-by-worker"
        )

        resource_path = session.resource_path("skill-resource")
        resource_result = await session.run_command(
            f"test -r '{resource_path}/resource.txt' && "
            f"! printf blocked >> '{resource_path}/resource.txt'"
        )
        assert "exit_code: 0" in resource_result

        hidden_result = await session.run_command(
            "test ! -e /workspace/.linktools/host-control.txt"
        )
        assert "exit_code: 0" in hidden_result
    finally:
        await session.close()


async def test_bubblewrap_blocks_external_network_but_keeps_loopback(
    tmp_path: Path,
) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    try:
        blocked_code = (
            "import socket,sys; "
            "s=socket.socket(); s.settimeout(1); "
            "\ntry: s.connect(('1.1.1.1', 53))\n"
            "except OSError: sys.exit(0)\n"
            "sys.exit(1)"
        )
        blocked = await session.run_command(
            f"/usr/bin/python3 -c {shlex.quote(blocked_code)}"
        )
        assert "exit_code: 0" in blocked

        loopback_code = (
            "import socket; "
            "server=socket.socket(); server.bind(('127.0.0.1', 0)); "
            "server.listen(1); port=server.getsockname()[1]; "
            "client=socket.socket(); client.connect(('127.0.0.1', port)); "
            "conn,_=server.accept(); client.sendall(b'ok'); "
            "assert conn.recv(2) == b'ok'"
        )
        loopback = await session.run_command(
            f"/usr/bin/python3 -c {shlex.quote(loopback_code)}"
        )
        assert "exit_code: 0" in loopback
    finally:
        await session.close()


async def test_bubblewrap_close_reaps_background_command(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    marker = workspace_root / "background-marker.txt"
    code = (
        "import pathlib,time; time.sleep(2); "
        "pathlib.Path('background-marker.txt').write_text('alive', encoding='utf-8')"
    )
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    started = await session.start_command(
        f"/usr/bin/python3 -c {shlex.quote(code)}"
    )
    assert re.search(r"command_id: [A-Za-z0-9._-]+", started)
    await session.close()
    await asyncio.sleep(2.2)
    assert not marker.exists()


async def test_bubblewrap_guardian_loss_is_observable(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    marker = workspace_root / "guardian-loss-marker.txt"
    code = (
        "import pathlib,time; time.sleep(2); "
        "pathlib.Path('guardian-loss-marker.txt').write_text('alive', encoding='utf-8')"
    )
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    started = await session.start_command(
        f"/usr/bin/python3 -c {shlex.quote(code)}"
    )
    assert re.search(r"command_id: [A-Za-z0-9._-]+", started)

    process = session._process
    process.kill()
    await process.wait()
    await asyncio.sleep(0)
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("true")
        assert raised.value.code is ErrorCode.SANDBOX_SESSION_LOST
    finally:
        await session.close()

    await asyncio.sleep(2.2)
    assert not marker.exists()


async def test_bubblewrap_mcp_spawn_cancellation_reaps_process_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
    ).open(root=workspace_root, resources=(resource,))
    ready_wait = asyncio.Event()

    async def wait_for_ready(process, control_fd) -> None:
        del process, control_fd
        ready_wait.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_bubblewrap, "_wait_stdio_ready", wait_for_ready)
    spawn = asyncio.create_task(
        session.open_stdio_process(
            "/usr/bin/python3",
            (
                SandboxResourcePath("mcp", "server.py"),
                str(external),
                "1",
                str(external),
                "spawn-on-start",
            ),
            resources=(resource,),
        )
    )
    try:
        await asyncio.wait_for(ready_wait.wait(), timeout=15)
        deadline = asyncio.get_running_loop().time() + 5
        while (
            not (workspace_root / "spawn-ready.txt").exists()
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        assert (workspace_root / "spawn-ready.txt").exists()
        spawn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await spawn
    finally:
        if not spawn.done():
            spawn.cancel()
            await asyncio.gather(spawn, return_exceptions=True)
        await session.close()

    await asyncio.sleep(2.2)
    assert not (workspace_root / "spawn-alive.txt").exists()


async def test_workspace_mcp_stdio_protocol_and_readonly_boundary(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    try:
        tmp_path.resolve().relative_to(runtime_root.resolve())
    except ValueError:
        pass
    else:
        pytest.fail("the acceptance workspace must be outside the runtime root")

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")

    control_root = tmp_path / "control-workspace"
    control_external = tmp_path / "control-secret.txt"
    control_external.write_text("host-only", encoding="utf-8")
    _prepare_probe_tree(control_root, control_external)
    control_resource = _create_stdio_server(tmp_path / "control-resource")
    control = subprocess.run(
        [
            sys.executable,
            str(control_resource.source / "server.py"),
            "--probe",
            str(control_root),
            str(control_external),
            str(control_external),
            str(port),
            str(control_resource.source / "resource.txt"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(json.loads(control.stdout).values()) == {"allowed"}

    workspace_root = tmp_path / "workspace"
    _prepare_probe_tree(workspace_root, external)
    (workspace_root / ".linktools").mkdir()
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    sandbox = BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
        read_policy=ReadOnlySandboxPolicy(readable_paths=("**",)),
    )
    session = await sandbox.open(root=workspace_root)
    with pytest.raises(AIError) as hidden_resource:
        session.resource_path("mcp")
    assert hidden_resource.value.code is ErrorCode.REQUEST_FIELD_INVALID

    transport = _SandboxMCPTransport(
        session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            str(external),
            str(port),
            "/workspace/.." + str(external),
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    large_value = "审计🙂" * 20_000
    try:
        async with client:
            tools = await toolset.get_tools(context)
            assert tuple(tools) == (model_name,)
            assert len(model_name) == 55
            assert '"server_id":"security/audit"' in (
                tools[model_name].tool_def.description or ""
            )
            result = await toolset.call_tool(
                model_name,
                {
                    "value": large_value,
                    "notify": True,
                    "flood": True,
                },
                context,
                tools[model_name],
            )
            observations = json.loads(result[0])
            assert observations["read"] == "allowed"
            assert set(observations.values()) - {"allowed"} == {"blocked"}
            assert result[1] == large_value
    finally:
        await session.close()
        listener.close()


async def test_workspace_mcp_stdio_empty_and_narrow_read_policies(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    workspace_root = tmp_path / "workspace"
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    _prepare_probe_tree(workspace_root, external)
    (workspace_root / ".linktools").mkdir()
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]

    empty_session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
        read_policy=ReadOnlySandboxPolicy(readable_paths=()),
    ).open(root=workspace_root, resources=(resource,))
    transport = _SandboxMCPTransport(
        empty_session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            str(external),
            str(port),
            "/workspace/.." + str(external),
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    try:
        async with client:
            tools = await toolset.get_tools(context)
            result = await toolset.call_tool(
                model_name,
                {},
                context,
                tools[model_name],
            )
            observations = json.loads(result[0])
            assert observations["read"] == "blocked"
            assert observations["create"] == "blocked"
            assert observations["resource"] == "blocked"
    finally:
        await empty_session.close()

    narrow_session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
        read_policy=ReadOnlySandboxPolicy(
            readable_paths=("allowed/*.txt",),
        ),
    ).open(root=workspace_root)
    try:
        with pytest.raises(AIError) as raised:
            await narrow_session.open_stdio_process("/usr/bin/python3")
        assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
        assert raised.value.safe_details == {
            "reason": "stdio_read_policy_unsupported"
        }
    finally:
        await narrow_session.close()
        listener.close()


async def test_workspace_mcp_stdio_close_reaps_setsid_descendant(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
    ).open(root=workspace_root, resources=(resource,))
    transport = _SandboxMCPTransport(
        session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            str(external),
            "1",
            "/workspace/.." + str(external),
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    try:
        async with client:
            tools = await toolset.get_tools(context)
            await toolset.call_tool(
                model_name,
                {"spawn": True},
                context,
                tools[model_name],
            )
    finally:
        await session.close()

    await asyncio.sleep(2.2)
    assert not (workspace_root / "detached.txt").exists()


async def test_workspace_mcp_stdio_cancelled_call_closes_process(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
    ).open(root=workspace_root, resources=(resource,))
    transport = _SandboxMCPTransport(
        session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            str(external),
            "1",
            "/workspace/.." + str(external),
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    try:
        async with client:
            tools = await toolset.get_tools(context)
            call = asyncio.create_task(
                toolset.call_tool(
                    model_name,
                    {"hang": True},
                    context,
                    tools[model_name],
                )
            )
            await asyncio.sleep(0.2)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
    finally:
        await session.close()

    await asyncio.sleep(3.2)
    assert not (workspace_root / "call-alive.txt").exists()


async def test_workspace_mcp_stdio_cancelled_close_reaps_process_tree(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    session = await BubblewrapSandbox(
        runtime_root=runtime_root,
        bwrap_executable=bwrap,
    ).open(root=workspace_root, resources=(resource,))
    transport = _SandboxMCPTransport(
        session,
        "/usr/bin/python3",
        (
            SandboxResourcePath("mcp", "server.py"),
            str(external),
            "1",
            str(external),
            "spawn-close-child",
        ),
        (resource,),
    )
    client, toolset, context, model_name = _runtime_mcp_client(transport)
    closing: asyncio.Task[object] | None = None
    try:
        await client.__aenter__()
        tools = await toolset.get_tools(context)
        await toolset.call_tool(
            model_name,
            {"spawn": True},
            context,
            tools[model_name],
        )
        closing = asyncio.create_task(client.__aexit__(None, None, None))
        await asyncio.sleep(0.2)
        assert not closing.done()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
    finally:
        if closing is not None and not closing.done():
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)
        await session.close()

    await asyncio.sleep(6.2)
    assert not (workspace_root / "detached.txt").exists()


async def test_workspace_mcp_stdio_runtime_death_reaps_process_tree(
    tmp_path: Path,
) -> None:
    runtime_root, bwrap = _sandbox_configuration()
    try:
        tmp_path.resolve().relative_to(runtime_root.resolve())
    except ValueError:
        pass
    else:
        pytest.fail("the acceptance workspace must be outside the runtime root")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    external = tmp_path / "host-secret.txt"
    external.write_text("host-only", encoding="utf-8")
    resource = _create_stdio_server(tmp_path / "mcp-resource")
    runtime_script = tmp_path / "runtime.py"
    runtime_script.write_text(
        textwrap.dedent(
            """\
            import asyncio
            import sys
            from pathlib import Path

            from fastmcp import Client
            from linktools.ai.runtime._mcp_transport import _SandboxMCPTransport
            from linktools.ai.workspace import (
                BubblewrapSandbox,
                SandboxResource,
                SandboxResourcePath,
            )

            async def main():
                runtime_root = Path(sys.argv[1])
                bwrap = Path(sys.argv[2])
                workspace = Path(sys.argv[3])
                resource_root = Path(sys.argv[4])
                external = Path(sys.argv[5])
                resource = SandboxResource("mcp", resource_root)
                session = await BubblewrapSandbox(
                    runtime_root=runtime_root,
                    bwrap_executable=bwrap,
                ).open(root=workspace, resources=(resource,))
                transport = _SandboxMCPTransport(
                    session,
                    "/usr/bin/python3",
                    (
                        SandboxResourcePath("mcp", "server.py"),
                        str(external),
                        "1",
                        "/workspace/.." + str(external),
                    ),
                    (resource,),
                )
                async with Client(transport) as client:
                    await client.call_tool("probe", {"spawn": True})
                    (workspace / "runtime-ready").write_text("ready")
                    await asyncio.Event().wait()

            asyncio.run(main())
            """
        ),
        encoding="utf-8",
    )
    runtime = await asyncio.create_subprocess_exec(
        sys.executable,
        str(runtime_script),
        str(runtime_root),
        str(bwrap),
        str(workspace_root),
        str(resource.source),
        str(external),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        ready = workspace_root / "runtime-ready"
        deadline = asyncio.get_running_loop().time() + 30
        while (
            not ready.exists()
            and runtime.returncode is None
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)
        if runtime.returncode is None:
            runtime.kill()
        stdout, stderr = await runtime.communicate()
        if not ready.exists():
            pytest.fail(
                "the child Runtime did not reach the MCP call: "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
    finally:
        if runtime.returncode is None:
            runtime.kill()
            await runtime.wait()

    await asyncio.sleep(2.2)
    assert not (workspace_root / "detached.txt").exists()
