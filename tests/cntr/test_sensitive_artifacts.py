#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitive generated files stay private even when their text is unchanged."""
import os
import stat
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr._container.compose import write_docker_compose_file, write_docker_file
from linktools.cntr._operations import ComposeOperations
from linktools.cntr.artifacts import AppliedServiceModels, ArtifactIndex, atomic_write_text_if_changed


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("mode", [None, 0o600])
def test_explicit_permissions_do_not_change_generic_writer_contract(tmp_path: Path, changed: bool,
                                                                   mode: "int | None") -> None:
    path = tmp_path / "artifact"
    path.write_text("before")
    path.chmod(0o644)
    mtime = path.stat().st_mtime_ns
    assert atomic_write_text_if_changed(path, "after" if changed else "before", mode=mode) is changed
    assert stat.S_IMODE(path.stat().st_mode) == (0o644 if mode is None else mode)
    if not changed:
        assert path.stat().st_mtime_ns == mtime


@pytest.fixture
def owner(tmp_path: Path) -> SimpleNamespace:
    manager = SimpleNamespace(
        data_path=tmp_path, project_name="test", docker_compose_names=("compose.yml",),
        environ=SimpleNamespace(locks=SimpleNamespace(process_lock=lambda key: nullcontext())),
    )
    manager.artifact_index = ArtifactIndex(manager)
    return SimpleNamespace(
        name="app", manager=manager, repo_context=None,
        get_source_path=lambda name: tmp_path / "sources" / name,
        docker_compose={"services": {"app": {"image": "image:one", "environment": {"PASSWORD": "test-only"}}}},
        docker_file="FROM scratch\nENV DNS_TOKEN=test-only\n",
    )


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("kind", ["compose", "dockerfile"])
def test_sensitive_output_restricts_existing_permissions(owner: SimpleNamespace, kind: str, changed: bool) -> None:
    writer = write_docker_compose_file if kind == "compose" else write_docker_file
    path = writer(owner)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.chmod(0o644)
    if changed:
        owner.docker_compose["services"]["app"]["environment"]["PASSWORD"] = "updated-test-only"
        owner.docker_file += "ENV EXTRA_TOKEN=updated-test-only\n"
    writer(owner)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("changed", [False, True])
def test_applied_compose_snapshot_remains_private_on_record_and_restore(owner: SimpleNamespace, changed: bool) -> None:
    manager = owner.manager
    path = manager.data_path / "compose" / "app.yml"
    previous = "services:\n  app:\n    image: old\n"
    content = previous if not changed else "services:\n  app:\n    image: new\n"
    context = SimpleNamespace(
        service_models=AppliedServiceModels(manager, owner.docker_compose),
        compose_files={str(path): content}, compose_owners={str(path): owner.name},
        saved_compose={str(path): previous}, original_applied_compose={str(path): previous},
        applied_compose={},
    )
    destination = manager.data_path / "compose" / "applied" / "app.yml"
    destination.parent.mkdir(parents=True)
    destination.write_text(previous)
    destination.chmod(0o644)
    operations = ComposeOperations(manager)
    operations._record_applied_compose(owner, context, ("app",))
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    destination.chmod(0o644)
    operations._restore_applied_compose(owner, context, {str(path): previous})
    assert destination.read_text() == previous
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_sensitive_replacement_never_restores_a_permissive_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "secret"
    path.write_text("before")
    path.chmod(0o644)
    modes = []
    chmod = os.chmod

    def record_mode(target, mode):
        modes.append(mode)
        chmod(target, mode)

    monkeypatch.setattr(os, "chmod", record_mode)
    atomic_write_text_if_changed(path, "after", mode=0o600)
    assert modes == [0o600]
