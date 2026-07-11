#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manifest docker-engine/docker-compose runtime requirements are enforced
before up/restart/compose actually run -- and never for down/status/
plan down, which must stay usable no matter what a repository's manifest
declares."""
import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry
from linktools.cntr.repo.context import ContainerRepositoryContext
from linktools.cntr.runtime.inspect import DockerEngineVersion
from linktools.core import LinktoolsManifest, ManifestComponent


def _attach_manifest(container, requires, url="https://example.invalid/repo.git"):
    """``requires`` uses runtime:docker-engine/runtime:docker-compose keys,
    attached as the cntr component's own requirements."""
    component = ManifestComponent(schema_version=1, requires=requires, config={}, metadata={}, extensions={})
    manifest = LinktoolsManifest(
        schema_version=1, kind="linktools-project", name=None, version=None, description=None,
        requires={}, components={"cntr": component}, metadata={}, extensions={},
    )
    container._repository = ContainerRepositoryContext(
        url=url, root_path="/tmp/fake-repo", manifest=manifest, builtin=False,
    )


class _InertProcess:
    def check_call(self):
        return 0


def _neutralize_real_dispatch(monkeypatch, manager):
    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", lambda *a, **k: _InertProcess())
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)


def test_satisfied_requirement_does_not_block(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.0"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.30.0")

    selection = fresh_manager.compose_operations.select()
    fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")  # must not raise


def test_unsatisfied_engine_requirement_blocks(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-engine": ">=99.0"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_engine_version",
                        lambda *a, **k: DockerEngineVersion(client="20.0", server="20.0", api="1.40"))

    selection = fresh_manager.compose_operations.select()
    with pytest.raises(ContainerError, match="docker-engine"):
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")


def test_unsatisfied_compose_requirement_blocks(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    selection = fresh_manager.compose_operations.select()
    with pytest.raises(ContainerError, match="docker-compose"):
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")


def test_unreachable_docker_blocks_fail_closed(fresh_manager, monkeypatch):
    """An unqueryable runtime must fail closed for a real action, not
    silently proceed as if the requirement were satisfied."""
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.0"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: None)

    selection = fresh_manager.compose_operations.select()
    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")


def test_multiple_repos_aggregate_into_one_error(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    lldap = fresh_manager.containers["lldap"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/repo-a.git")
    _attach_manifest(lldap, {"runtime:docker-engine": ">=99.0"}, url="https://example.invalid/repo-b.git")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_engine_version",
                        lambda *a, **k: DockerEngineVersion(client="20.0", server="20.0", api="1.40"))

    selection = fresh_manager.compose_operations.select()
    with pytest.raises(ContainerError) as exc_info:
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")
    message = str(exc_info.value)
    assert "repo-a.git" in message
    assert "repo-b.git" in message


def test_same_repo_across_containers_is_deduplicated(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    lldap = fresh_manager.containers["lldap"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/shared.git")
    _attach_manifest(lldap, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/shared.git")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    selection = fresh_manager.compose_operations.select()
    with pytest.raises(ContainerError) as exc_info:
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")
    assert str(exc_info.value).count("shared.git") == 1


def test_up_is_blocked_before_any_real_dispatch(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    def fail(*a, **k):
        raise AssertionError("must not dispatch a real compose call when requirements are unmet")

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)
    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.up()


def test_restart_is_blocked_before_any_real_dispatch(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    def fail(*a, **k):
        raise AssertionError("must not dispatch a real compose call when requirements are unmet")

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)
    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.restart()


def test_compose_render_is_blocked(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.render()


def test_compose_check_is_blocked(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    with pytest.raises(ContainerError):
        fresh_manager.compose_operations.render(check=True)


def test_down_is_never_blocked(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")
    _neutralize_real_dispatch(monkeypatch, fresh_manager)

    fresh_manager.compose_operations.down()  # must not raise


def test_status_is_never_blocked(fresh_manager, monkeypatch):
    from linktools.cntr.runtime.inspect import ProjectRuntimeState

    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")
    monkeypatch.setattr(
        fresh_manager.docker_inspector, "get_project_state",
        lambda *a, **k: ProjectRuntimeState(project="aio", services=(), backend="docker"),
    )

    fresh_manager.compose_operations.status()  # must not raise


def test_plan_up_is_blocked(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    with pytest.raises(ContainerError):
        fresh_manager.planner.plan("up")


def test_plan_restart_is_blocked(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    with pytest.raises(ContainerError):
        fresh_manager.planner.plan("restart")


def test_plan_down_only_warns(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    plan = fresh_manager.planner.plan("down")  # must not raise
    assert any("docker-compose" in w for w in plan.warnings)


def test_partial_up_checks_requirements_from_all_compose_files(fresh_manager, monkeypatch):
    """A partial `up`/`restart`/`compose` still generates a `--file` set from
    the whole installed project, so an unselected repository's requirement
    must still gate the action -- not just the user's explicit target."""
    nginx = fresh_manager.containers["nginx"]
    lldap = fresh_manager.containers["lldap"]
    _attach_manifest(nginx, {}, url="https://example.invalid/repo-a.git")
    _attach_manifest(lldap, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/repo-b.git")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    # Only nginx is targeted; lldap's unmet requirement must still block.
    selection = fresh_manager.compose_operations.select(["nginx"])
    with pytest.raises(ContainerError, match="repo-b.git"):
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")


def test_partial_compose_checks_requirements_from_all_project_repositories(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    lldap = fresh_manager.containers["lldap"]
    _attach_manifest(nginx, {}, url="https://example.invalid/repo-a.git")
    _attach_manifest(lldap, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/repo-b.git")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    with pytest.raises(ContainerError, match="repo-b.git"):
        fresh_manager.compose_operations.render(names=["nginx"])


def test_duplicate_repository_requirements_are_checked_once(fresh_manager, monkeypatch):
    """Two containers backed by the same repository (same url) must not
    duplicate that repository's problems in the aggregated error."""
    nginx = fresh_manager.containers["nginx"]
    lldap = fresh_manager.containers["lldap"]
    _attach_manifest(nginx, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/shared.git")
    _attach_manifest(lldap, {"runtime:docker-compose": ">=2.30"}, url="https://example.invalid/shared.git")
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.10.0")

    # Even when only one of the two is targeted, the full project is scanned
    # once per unique repository, not once per container.
    selection = fresh_manager.compose_operations.select(["nginx"])
    with pytest.raises(ContainerError) as exc_info:
        fresh_manager.compose_operations.ensure_runtime_requirements(selection, "up")
    assert str(exc_info.value).count("shared.git") == 1
