#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Pinned non-file Asset packages execute as isolated, run-owned stdio trees."""
from ._runtime_test_helpers import model_features

from linktools.ai.capability import TraceCapability

import asyncio
from functools import partial
import json
import os
import sys
import tempfile
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetMaterializer,
    AssetStore,
    AssetStoreReader,
    AssetVersionRef,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    MaterializedAssets,
    SqlAssetBackend,
)
from linktools.ai.capability import AssetCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.spec import mcp_tool_selector
from linktools.ai.storage import FilesystemObjectStore, PayloadPolicy, StorageOverlay
from linktools.ai.workspace import BubblewrapWorkspaceBackend, LocalWorkspaceBackend
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo

from ._runtime_test_helpers import RuntimeUsageModels, _UsageFunctionModel
from .test_bubblewrap_integration import _sandbox_configuration

_SERVER = textwrap.dedent(
    '''\
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    import json
    import os
    import signal
    import sys
    import time
    from pathlib import Path
    from sibling import IMPORTED

    def observe():
        return {
            "cwd": str(Path.cwd()),
            "script": str(Path(__file__).resolve()),
            "pid": os.getpid(),
            "imported": IMPORTED,
            "data": Path("data/value.txt").read_text(),
        }

    def log(phase):
        if os.environ.get("TEST_LOG"):
            result = {"phase": phase, "cwd": str(Path.cwd()), "pid": os.getpid()}
            result["resource_exists"] = Path("data/value.txt").is_file()
            if result["resource_exists"]:
                result["data"] = Path("data/value.txt").read_text()
            target = Path(os.environ["TEST_LOG"], str(os.getpid()) + ".json")
            pending = target.with_suffix(".pending")
            pending.write_text(json.dumps(result))
            pending.replace(target)

    def terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        log("started")
        for line in sys.stdin:
            request = json.loads(line)
            if "id" not in request:
                continue
            method = request.get("method")
            if method == "initialize":
                if os.environ.get("TEST_MODE") == "fail-init":
                    raise SystemExit(3)
                if os.environ.get("TEST_MODE") == "hang-init":
                    time.sleep(60)
                result = {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "asset-probe", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": "probe",
                    "description": "Read the pinned package and report its execution root",
                    "inputSchema": {"type": "object", "properties": {}},
                }]}
            elif method == "tools/call":
                value = observe()
                result = {
                    "content": [{"type": "text", "text": json.dumps(value)}],
                    "structuredContent": value,
                    "isError": False,
                }
            elif method == "ping":
                result = {}
            else:
                print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {
                    "code": -32601, "message": "Method not found"
                }}), flush=True)
                continue
            print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        log("closed")
    '''
).encode()


class _ProbeBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:asset-stdio"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "asset-stdio"}

    def __init__(
        self,
        on_result: Callable[[dict[str, JsonValue]], Awaitable[None]] | None = None,
    ) -> None:
        self.on_result = on_result

    def materialize(self) -> _UsageFunctionModel:
        async def request(
            messages: list[ModelMessage], info: AgentInfo,
        ) -> ModelResponse:
            returned = [
                part
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            if not returned:
                tool = next(
                    tool for tool in info.function_tools
                    if '"tool_name":"probe"' in (tool.description or "")
                )
                return ModelResponse(parts=[ToolCallPart(tool.name, {}, "probe")])
            value = json.loads(returned[-1].model_response_str())
            assert isinstance(value, dict)
            if self.on_result is not None:
                await self.on_result(value)
            return ModelResponse(parts=[TextPart(json.dumps(value))])

        return _UsageFunctionModel(request)


class _ProbeModels:
    def __init__(
        self,
        on_result: Callable[[dict[str, JsonValue]], Awaitable[None]] | None = None,
    ) -> None:
        self.binding = _ProbeBinding(on_result)

    def __call__(self) -> _ProbeBinding:
        return self.binding


@pytest.fixture
def temporary_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "temporary"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    materialize = AssetMaterializer.materialize
    close = AssetMaterializer.close
    owned: dict[AssetMaterializer, set[Path]] = {}

    async def record_materialization(
        owner: AssetMaterializer,
        reader: AssetStoreReader,
        files: Mapping[str, AssetVersionRef],
        *,
        executable_bits: Mapping[str, int] | None = None,
    ) -> MaterializedAssets:
        value = await materialize(owner, reader, files, executable_bits=executable_bits)
        owned.setdefault(owner, set()).add(value.root)
        return value

    async def check_close(owner: AssetMaterializer) -> None:
        # Process exit alone cannot reveal whether its files were deleted early.
        # Inspect this public ownership boundary before deletion actually starts.
        for path in (tmp_path / "processes").glob("*.json"):
            value = json.loads(path.read_text())
            if Path(value["cwd"]) in owned.get(owner, set()):
                with pytest.raises(ProcessLookupError):
                    os.kill(value["pid"], 0)
        await close(owner)

    monkeypatch.setattr(AssetMaterializer, "materialize", record_materialization)
    monkeypatch.setattr(AssetMaterializer, "close", check_close)
    return root


@asynccontextmanager
async def _store(kind: str, root: Path) -> AsyncIterator[AssetStore]:
    engine = None
    if kind == "memory":
        backend = InMemoryAssetBackend()
    elif kind in {"sqlite", "sqlite-filesystem"}:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'assets.sqlite'}")
        await provision_asset_database(engine)
        backend = SqlAssetBackend(
            engine, namespace="stdio",
            object_store=FilesystemObjectStore(root / "objects") if kind == "sqlite-filesystem" else None,
            payload_policy=PayloadPolicy(0) if kind == "sqlite-filesystem" else None,
        )
    else:
        raise AssertionError(kind)
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


def _package(
    *,
    command: str = "python",
    argument: str = "server.py",
    log: Path | None = None,
    mode: str = "",
) -> dict[str, bytes]:
    # JSON values are also valid YAML scalars and lists.
    declaration = (
        f"command: {json.dumps(command)}\n"
        f"args: {json.dumps([argument])}\n"
        f"env: {json.dumps({'TEST_LOG': str(log) if log else '', 'TEST_MODE': mode})}\n"
    ).encode()
    return {
        "mcp.yaml": declaration,
        "server.py": _SERVER,
        "sibling.py": b'IMPORTED = "captured-module"\n',
        "data/value.txt": b"captured-data",
    }


async def _put_package(store: AssetStore, files: Mapping[str, bytes]) -> None:
    for path, content in files.items():
        await store.put(AssetKey("mcp", f"probe/{path}"), content)


def _workspace_factory(kind: str):
    return LocalWorkspaceBackend


def _group(store: AssetStore, workspace_factory) -> RuntimeFeatures[object]:
    group = RuntimeFeatures[object]("assets", workspace_factory=workspace_factory)
    group_assets = AssetCapability(store)
    group.capability(group_assets, scope="runtime")
    group.agent("default", allow_tools=(mcp_tool_selector("probe", "probe"),))
    return group


def _assert_observation(value: object) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    observation = json.loads(value["text"])
    assert observation["imported"] == "captured-module"
    assert observation["data"] == "captured-data"
    assert Path(observation["script"]).parent == Path(observation["cwd"])
    return observation


def _assert_closed(log: Path, root: Path, *, retained: bool = False) -> None:
    values = [json.loads(path.read_text()) for path in log.glob("*.json")]
    assert values
    for value in values:
        assert value["resource_exists"] is True
        assert value["data"] == "captured-data"
        assert Path(value["cwd"]).exists() is retained
        with pytest.raises(ProcessLookupError):
            os.kill(value["pid"], 0)
    if retained:
        assert set(root.iterdir()) == {Path(value["cwd"]) for value in values}
    else:
        assert not tuple(root.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(("backend_kind", "sandbox_kind"), (
    ("memory", "default"),
    ("memory", "local"),
    ("sqlite-filesystem", "local"),
    ("sqlite", "local"),
))
async def test_plain_filename_uses_pinned_asset_package_and_preserves_history(
    backend_kind: str,
    sandbox_kind: str,
    tmp_path: Path,
    temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "processes"
    log.mkdir()
    sandbox = _workspace_factory(sandbox_kind)
    async with _store(backend_kind, tmp_path) as store:
        await _put_package(store, _package(log=log))
        features = _group(store, sandbox)
        features.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('asset-stdio', storage=PersistenceStorage.in_memory()) as persistence_334, Runtime.attach(persistence=persistence_334, features=model_features(_ProbeModels(), features)) as runtime:
            await store.put(AssetKey("mcp", "probe/data/value.txt"), b"new-head-data")
            await store.put(AssetKey("mcp", "probe/sibling.py"), b'IMPORTED = "new-head-module"\n')
            result = (await runtime.agents.get("default").run("probe", timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            observation = _assert_observation(result.output)
            assert Path(observation["cwd"]).is_relative_to(temporary_root)
            assert not Path(observation["cwd"]).exists()
            execution = await runtime.executions.get(result.execution_id)
            history = await execution.history(include_content=True)
            assert "captured-data" in str([item.content for item in history.items])
            assert "new-head-data" not in str([item.content for item in history.items])
        _assert_closed(log, temporary_root)


@pytest.mark.asyncio
@pytest.mark.parametrize(("sandbox_kind", "argument"), (
    ("local", "resource:server.py"),
    ("injected-no-session", "resource:server.py"),
    ("bubblewrap", "resource:server.py"),
    ("bubblewrap", "server.py"),
))
async def test_resource_arguments_use_the_package_working_directory(
    sandbox_kind: str,
    argument: str,
    tmp_path: Path,
    temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sandbox_kind == "bubblewrap":
        runtime_root, executable = _sandbox_configuration()
        sandbox = partial(BubblewrapWorkspaceBackend, runtime_root=runtime_root, bwrap_executable=executable)
        command = "/usr/bin/python3"
    else:
        sandbox = _workspace_factory(sandbox_kind)
        command = sys.executable
    async with _store("memory", tmp_path) as store:
        await _put_package(store, _package(command=command, argument=argument))
        async with Persistence.open('asset-resource-argument', storage=PersistenceStorage.in_memory()) as persistence_375, Runtime.attach(persistence=persistence_375, features=model_features(_ProbeModels(), _group(store, sandbox))) as runtime:
            result = (await runtime.agents.get("default").run("probe", timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            _assert_observation(result.output)
    assert not tuple(temporary_root.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_kind", ("local",))
async def test_concurrent_runs_own_independent_trees_until_their_processes_close(
    sandbox_kind: str,
    tmp_path: Path,
    temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: asyncio.Queue[dict[str, JsonValue]] = asyncio.Queue()
    releases: dict[int, asyncio.Event] = {}

    async def hold(value: dict[str, JsonValue]) -> None:
        pid = value["pid"]
        assert isinstance(pid, int)
        releases[pid] = asyncio.Event()
        await observations.put(value)
        await releases[pid].wait()

    log = tmp_path / "processes"
    log.mkdir()
    sandbox = _workspace_factory(sandbox_kind)
    async with _store("memory", tmp_path) as store:
        await _put_package(store, _package(command=sys.executable, log=log))
        async with Persistence.open('asset-concurrency', storage=PersistenceStorage.in_memory()) as persistence_408, Runtime.attach(persistence=persistence_408, features=model_features(_ProbeModels(hold), _group(store, sandbox))) as runtime:
            first = asyncio.create_task(runtime.agents.get("default").run("first", timeout_seconds=15))
            second = None
            try:
                one = await asyncio.wait_for(observations.get(), timeout=10)
                second = asyncio.create_task(runtime.agents.get("default").run("second", timeout_seconds=15))
                two = await asyncio.wait_for(observations.get(), timeout=10)
                assert one["cwd"] != two["cwd"]
                assert one["pid"] != two["pid"]
                assert Path(one["cwd"]).is_dir()
                assert Path(two["cwd"]).is_dir()
                releases[one["pid"]].set()
                assert (await first).result.status is ExecutionStatus.SUCCEEDED
                assert not Path(one["cwd"]).exists()
                assert Path(two["cwd"]).is_dir()
                os.kill(two["pid"], 0)
                releases[two["pid"]].set()
                assert (await second).result.status is ExecutionStatus.SUCCEEDED
            finally:
                for event in releases.values():
                    event.set()
                await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)
        _assert_closed(log, temporary_root)


async def _wait_started(log: Path) -> dict[str, JsonValue]:
    async def started() -> dict[str, JsonValue]:
        while True:
            for path in log.glob("*.json"):
                try:
                    value = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                if value["phase"] == "started":
                    return value
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(started(), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_kind", ("local",))
@pytest.mark.parametrize("mode", ("fail-init", "hang-init"))
async def test_startup_outcomes_keep_process_and_resource_lifetimes_ordered(
    sandbox_kind: str,
    mode: str,
    tmp_path: Path,
    temporary_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "processes"
    log.mkdir()
    sandbox = _workspace_factory(sandbox_kind)
    async with _store("memory", tmp_path) as store:
        await _put_package(store, _package(command=sys.executable, log=log, mode=mode))
        async with Persistence.open('asset-init-failure', storage=PersistenceStorage.in_memory()) as persistence_466, Runtime.attach(persistence=persistence_466, features=model_features(_ProbeModels(), _group(store, sandbox))) as runtime:
            execution = await runtime.agents.get("default").start("probe")
            if mode == "hang-init":
                value = await _wait_started(log)
                assert Path(value["cwd"]).is_dir()
                await asyncio.wait_for(execution.cancel(), timeout=15)
            result = (await execution.wait(timeout_seconds=15)).result
            if mode == "fail-init":
                assert result.status is ExecutionStatus.FAILED
                assert result.error_code == ErrorCode.SANDBOX_SESSION_LOST.value
            else:
                assert result.status is ExecutionStatus.CANCELLED
        _assert_closed(log, temporary_root)



@pytest.mark.asyncio
async def test_native_directory_package_keeps_original_files(
    tmp_path: Path,
    temporary_root: Path,
) -> None:
    root = tmp_path / "native"
    package = root / "mcp" / "probe"
    for name, content in _package(command=sys.executable).items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    store = AssetStore(StorageOverlay(DirectoryAssetBackend(str(root), kinds=("mcp",))))
    await store.initialize()
    sandbox = LocalWorkspaceBackend
    try:
        async with Persistence.open('native-asset-stdio', storage=PersistenceStorage.in_memory()) as persistence_508, Runtime.attach(persistence=persistence_508, features=model_features(_ProbeModels(), _group(store, sandbox))) as runtime:
            result = (await runtime.agents.get("default").run("probe", timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            observation = _assert_observation(result.output)
            assert Path(observation["cwd"]) == package
        assert (package / "server.py").read_bytes() == _SERVER
        assert (package / "data/value.txt").read_bytes() == b"captured-data"
        assert not tuple(temporary_root.iterdir())
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_disabled_workspace_never_falls_back_to_host_stdio(tmp_path: Path) -> None:
    log = tmp_path / "processes"
    log.mkdir()
    async with _store("memory", tmp_path) as store:
        await _put_package(store, _package(command=sys.executable, log=log))
        async with Persistence.open("disabled-stdio", storage=PersistenceStorage.in_memory()) as persistence:
            async with Runtime.attach(
                persistence=persistence, features=model_features(_ProbeModels(), _group(store, None)),
            ) as runtime:
                with pytest.raises(AIError) as denied:
                    await runtime.agents.get("default").start("probe")
                assert denied.value.code is ErrorCode.SANDBOX_UNAVAILABLE
        assert not tuple(log.iterdir())
