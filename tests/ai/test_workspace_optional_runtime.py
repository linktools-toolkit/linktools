#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-optional Runtime behavior and file-source boundaries."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

from pathlib import Path

import pytest

from linktools.ai.asset import (
    AssetStore,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import AssetCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ExecutionCapability
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime import _factory as runtime_factory
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import (
    DisabledSandbox,
    LocalSandbox,
    SandboxResource,
    SandboxSession,
    Workspace,
)

from ._runtime_test_helpers import RuntimeUsageModels


class _RecordingLocalSandbox:
    def __init__(self) -> None:
        self.opened: list[tuple[Path, tuple[SandboxResource, ...]]] = []

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        self.opened.append((root, resources))
        return await LocalSandbox().open(root=root, resources=resources)


@pytest.mark.asyncio
async def test_workspace_less_runtime_runs_session_and_history() -> None:
    features = RuntimeFeatures("conversation")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Runtime.open(
        "web-chat",
        storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        assert runtime.namespace == "web-chat"
        session = await runtime.agents.get("default").create_session("chat")
        result = (await session.run("hello", timeout_seconds=10)).result

        assert result.status is ExecutionStatus.SUCCEEDED
        history = await session.history()
        assert history.items


@pytest.mark.asyncio
async def test_workspace_less_runtime_rejects_files_and_explicit_cwd() -> None:
    features = RuntimeFeatures("conversation")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Runtime.open(
        "web-chat",
        storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        with pytest.raises(AIError) as files_error:
            await runtime.agents.get("default").run(
                "inspect",
                files=("evidence.txt",),
                timeout_seconds=10,
            )
        assert files_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert files_error.value.safe_details == {
            "field": "files",
            "reason": "workspace_required",
        }

        with pytest.raises(AIError) as cwd_error:
            await runtime.agents.get("default").create_session(
                "cwd-session",
                cwd=".",
            )
        assert cwd_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert cwd_error.value.safe_details == {
            "field": "cwd",
            "reason": "workspace_required",
        }


@pytest.mark.asyncio
async def test_workspace_group_sandbox_controls_input_reads(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    group = workspace_features(
        "workspace",
        workspace=workspace,
        sandbox=DisabledSandbox(),
    )

    async with Runtime.open(
        "workspace-sandbox",
        storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels(), group),
    ) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.agents.get("default").run(
                "inspect",
                files=("evidence.txt",),
            )

    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE


@pytest.mark.asyncio
async def test_sandbox_group_can_be_composed_without_workspace() -> None:
    sandbox = DisabledSandbox()
    group = RuntimeFeatures("sandbox", sandbox=sandbox)
    snapshot = await group.capture()
    assert snapshot.workspace is None
    assert snapshot.sandbox is sandbox

    async with Runtime.open(
        "sandbox-only",
        storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels(), group),
    ) as runtime:
        result = (await runtime.agents.get("default").run("hello", timeout_seconds=10)).result
    assert result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_sandbox_without_workspace_exposes_local_skill_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "assets" / "skills" / "review"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review files\n---\n\nRun the script.\n",
        encoding="utf-8",
    )
    script = package / "run.sh"
    script.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    script.chmod(0o755)
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(tmp_path / "assets"),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    sandbox = _RecordingLocalSandbox()
    monkeypatch.chdir(tmp_path)
    try:
        asset_features = RuntimeFeatures("assets", sandbox=sandbox)
        asset_features.capability(AssetCapability(store), scope="runtime")
        async with Runtime.open(
            "sandbox-skill",
            storage=RuntimeStorage.in_memory(),
            features=model_features(RuntimeUsageModels(), asset_features),
        ) as runtime:
            result = (await runtime.agents.get("default").run(
                "review",
                timeout_seconds=10,
            )).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert len(sandbox.opened) == 1
        root, resources = sandbox.opened[0]
        assert root == tmp_path.resolve()
        assert len(resources) == 1
        assert resources[0].files is not None
        assert resources[0].files["run.sh"] == script.resolve()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_explicit_sandbox_controls_workspace_input_reads(
    tmp_path: Path,
) -> None:
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    async with Runtime.open(
        "separate-sandbox",
        storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels(), workspace_features("workspace", workspace=Workspace.load(tmp_path), sandbox=DisabledSandbox())),
    ) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.agents.get("default").run(
                "inspect",
                files=("evidence.txt",),
            )
    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE


@pytest.mark.asyncio
async def test_runtime_features_rejects_multiple_sources() -> None:
    # A single RuntimeFeatures owns sandbox selection; arrays are no longer an API.
    with pytest.raises(TypeError, match="features"):
        async with Runtime.open(
            "sandbox-conflict", storage=RuntimeStorage.in_memory(),
            features=model_features(RuntimeUsageModels(), (RuntimeFeatures("first", sandbox=DisabledSandbox()),
                              RuntimeFeatures("second", sandbox=DisabledSandbox()))),
        ):
            pytest.fail("multiple feature sources were accepted")


@pytest.mark.asyncio
async def test_existing_workspace_cwd_requires_workspace_for_new_turn(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace.load(project)
    storage_root = tmp_path / "runtime"

    features = workspace_features("workspace", workspace=workspace)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Runtime.open(
        "workspace",
        storage=RuntimeStorage.from_root(storage_root),
        features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        await runtime.agents.get("default").create_session(
            "cwd-session",
            cwd=".",
        )

    features = RuntimeFeatures("conversation")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Runtime.open(
        "workspace",
        storage=RuntimeStorage.from_root(storage_root),
        features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.agents.get("default").run(
                "continue",
                session_id="cwd-session",
                timeout_seconds=10,
            )

        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert error.value.safe_details == {
            "field": "cwd",
            "reason": "workspace_required",
        }


@pytest.mark.asyncio
async def test_workspace_less_runtime_does_not_require_host_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_factory, "_capture_host_cwd", lambda: None)
    components = await runtime_factory.compose_runtime_components(
        "web-chat",
        features=model_features(RuntimeUsageModels()),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
    )
    try:
        backend = components.execution.runtime_backend()
        assert backend._execution_cwd is None  # type: ignore[attr-defined]
    finally:
        await components.close_callback()
