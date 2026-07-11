#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment Lock: canonical JSON, no secrets, Git/local
repo representation, lock --check/diff read-only, legacy no-lock
compatibility."""
import json
import os

import pytest

from linktools.cntr.lock.diff import compute_diff
from linktools.cntr.lock.model import DeploymentLock, LockedArtifact, LockedContainer, LockedRepository
from linktools.cntr.lock.store import LockInvalid, LockNotFound, LockSchemaUnsupported, LockStore
from linktools.cntr.runtime.structured import CommandResult


@pytest.fixture(autouse=True)
def _inert_preflight(monkeypatch, fresh_manager):
    monkeypatch.setattr(fresh_manager.runtime, "create_docker_process", lambda *a, **k: object())
    monkeypatch.setattr(
        fresh_manager.structured_runner, "execute_text",
        lambda *a, **k: CommandResult(args=(), returncode=0, stdout="", stderr="", duration=0.0),
    )


def test_build_produces_containers_and_artifacts(fresh_manager):
    lock = fresh_manager.lock_store.build()
    assert lock.project == fresh_manager.project_name
    assert lock.linktools_cntr
    names = {c.name for c in lock.containers}
    assert "nginx" in names
    assert any(path.startswith("dockerfile/") for path in lock.artifacts)


def test_container_services_are_recorded(fresh_manager):
    lock = fresh_manager.lock_store.build()
    nginx = next(c for c in lock.containers if c.name == "nginx")
    assert "nginx" in nginx.services


def test_write_produces_canonical_json_with_trailing_newline(fresh_manager):
    lock = fresh_manager.lock_store.build()
    fresh_manager.lock_store.write(lock)

    with open(fresh_manager.lock_store.path, "r", encoding="utf-8") as f:
        content = f.read()

    assert content.endswith("\n")
    data = json.loads(content)
    assert json.dumps(data, sort_keys=True, indent=2) + "\n" == content
    assert data["schema_version"] == 1


def test_lock_does_not_contain_secrets_or_full_config(fresh_manager):
    fresh_manager.env_config.persist("DOCKER_USER", "super-secret-password-value")
    lock = fresh_manager.lock_store.build()
    fresh_manager.lock_store.write(lock)
    with open(fresh_manager.lock_store.path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "super-secret-password-value" not in content
    assert "RUNNING_CONTAINERS" not in content


def test_round_trip_to_dict_from_dict(fresh_manager):
    lock = fresh_manager.lock_store.build()
    data = fresh_manager.lock_store.to_dict(lock)
    restored = fresh_manager.lock_store.from_dict(data)
    assert restored == lock


def test_load_returns_none_when_missing(fresh_manager):
    assert fresh_manager.lock_store.load() is None


def test_load_required_raises_not_found_when_missing(fresh_manager):
    with pytest.raises(LockNotFound):
        fresh_manager.lock_store.load_required()


def _write_raw(fresh_manager, text):
    path = fresh_manager.lock_store.path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_load_raises_invalid_on_corrupt_json(fresh_manager):
    _write_raw(fresh_manager, "not json")
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_raises_invalid_on_empty_file(fresh_manager):
    _write_raw(fresh_manager, "")
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_raises_invalid_on_non_object_root(fresh_manager):
    _write_raw(fresh_manager, json.dumps([1, 2, 3]))
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_raises_invalid_when_schema_version_missing(fresh_manager):
    _write_raw(fresh_manager, json.dumps({"project": "aio"}))
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_raises_schema_unsupported_for_unknown_version(fresh_manager):
    _write_raw(fresh_manager, json.dumps({"schema_version": 999}))
    with pytest.raises(LockSchemaUnsupported):
        fresh_manager.lock_store.load()
    # LockSchemaUnsupported is itself a kind of LockInvalid.
    _write_raw(fresh_manager, json.dumps({"schema_version": 999}))
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_raises_invalid_when_repositories_is_not_a_list(fresh_manager):
    _write_raw(fresh_manager, json.dumps({"schema_version": 1, "repositories": "nope"}))
    with pytest.raises(LockInvalid):
        fresh_manager.lock_store.load()


def test_load_succeeds_for_a_valid_minimal_lock(fresh_manager):
    _write_raw(fresh_manager, json.dumps({"schema_version": 1}))
    lock = fresh_manager.lock_store.load()
    assert lock.schema_version == 1


def test_write_then_load_round_trips(fresh_manager):
    lock = fresh_manager.lock_store.build()
    fresh_manager.lock_store.write(lock)
    loaded = fresh_manager.lock_store.load()
    assert loaded == lock


def test_local_repo_records_definitions_sha256_not_revision(fresh_manager, tmp_path):
    repo_dir = tmp_path / "local_repo"
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\nclass Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    fresh_manager.repo_store.add(str(repo_dir), force=True)

    lock = fresh_manager.lock_store.build()
    local_repo = next(r for r in lock.repositories if r.type == "local")
    assert local_repo.revision is None
    assert local_repo.branch is None
    assert local_repo.definitions_sha256 is not None
    assert local_repo.reproducible is True


def test_local_repo_definitions_sha256_is_stable_across_rebuilds(fresh_manager, tmp_path):
    repo_dir = tmp_path / "local_repo"
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\nclass Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    fresh_manager.repo_store.add(str(repo_dir), force=True)

    lock1 = fresh_manager.lock_store.build()
    lock2 = fresh_manager.lock_store.build()
    repo1 = next(r for r in lock1.repositories if r.type == "local")
    repo2 = next(r for r in lock2.repositories if r.type == "local")
    assert repo1.definitions_sha256 == repo2.definitions_sha256


def test_local_repo_definitions_sha256_changes_with_content(fresh_manager, tmp_path):
    repo_dir = tmp_path / "local_repo"
    repo_dir.mkdir()
    container_py = repo_dir / "container.py"
    container_py.write_text(
        "from linktools.cntr.container import BaseContainer\n\n\nclass Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    lock_before = fresh_manager.lock_store.build()

    container_py.write_text(
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n    extra = 1\n",
        encoding="utf-8",
    )
    fresh_manager.__dict__.pop("containers", None)
    lock_after = fresh_manager.lock_store.build()

    before = next(r for r in lock_before.repositories if r.type == "local").definitions_sha256
    after = next(r for r in lock_after.repositories if r.type == "local").definitions_sha256
    assert before != after


def test_local_repo_definitions_sha256_covers_templates_directory(fresh_manager, tmp_path):
    # Every built-in container that ships a "templates" dir (nginx, authelia,
    # lldap) renders arbitrarily-named files from it via `render_template`
    # -- those are the "same-directory template files" the definitions hash covers.
    repo_dir = tmp_path / "local_repo"
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\nclass Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    templates_dir = repo_dir / "templates"
    templates_dir.mkdir()
    config_conf = templates_dir / "config.conf"
    config_conf.write_text("listen 80;\n", encoding="utf-8")
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    lock_before = fresh_manager.lock_store.build()

    config_conf.write_text("listen 443;\n", encoding="utf-8")
    fresh_manager.__dict__.pop("containers", None)
    lock_after = fresh_manager.lock_store.build()

    before = next(r for r in lock_before.repositories if r.type == "local").definitions_sha256
    after = next(r for r in lock_after.repositories if r.type == "local").definitions_sha256
    assert before != after


# -- Diff ----------------------------------------------------------------

def test_diff_against_none_reports_everything_added():
    new = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0",
        repositories=(LockedRepository(url="u", type="git", branch="main", revision="abc",
                                       manifest_sha256=None, manifest_version=None),),
        containers=(LockedContainer(name="nginx", repository_url=None, services=("nginx",)),),
        artifacts={"compose/nginx.yml": LockedArtifact(kind="compose", sha256="abc")},
    )
    diff = compute_diff(None, new)
    assert diff.containers_added == ("nginx",)
    assert diff.repositories_added == ("u",)
    assert diff.artifact_drifts[0].change == "added"
    assert not diff.is_empty


def test_diff_detects_no_drift_when_identical():
    lock = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0",
        repositories=(), containers=(), artifacts={},
    )
    diff = compute_diff(lock, lock)
    assert diff.is_empty


def test_diff_detects_cntr_version_change():
    old = DeploymentLock(schema_version=1, project="aio", linktools_cntr="1.0", repositories=(), containers=(),
                         artifacts={})
    new = DeploymentLock(schema_version=1, project="aio", linktools_cntr="2.0", repositories=(), containers=(),
                         artifacts={})
    diff = compute_diff(old, new)
    assert diff.cntr_version_changed is True
    assert not diff.is_empty


def test_diff_detects_repo_revision_drift():
    old = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0",
        repositories=(LockedRepository(url="u", type="git", branch="main", revision="abc",
                                       manifest_sha256=None, manifest_version=None),),
        containers=(), artifacts={},
    )
    new = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0",
        repositories=(LockedRepository(url="u", type="git", branch="main", revision="def",
                                       manifest_sha256=None, manifest_version=None),),
        containers=(), artifacts={},
    )
    diff = compute_diff(old, new)
    assert len(diff.repository_drifts) == 1
    assert diff.repository_drifts[0].field == "revision"
    assert diff.repository_drifts[0].old == "abc"
    assert diff.repository_drifts[0].new == "def"


def test_diff_detects_container_added_and_removed():
    old = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0", repositories=(),
        containers=(LockedContainer(name="a", repository_url=None, services=()),), artifacts={},
    )
    new = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0", repositories=(),
        containers=(LockedContainer(name="b", repository_url=None, services=()),), artifacts={},
    )
    diff = compute_diff(old, new)
    assert diff.containers_added == ("b",)
    assert diff.containers_removed == ("a",)


def test_diff_detects_artifact_added_changed_removed():
    old = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0", repositories=(), containers=(),
        artifacts={
            "a": LockedArtifact(kind="compose", sha256="1"),
            "b": LockedArtifact(kind="compose", sha256="2"),
        },
    )
    new = DeploymentLock(
        schema_version=1, project="aio", linktools_cntr="1.0", repositories=(), containers=(),
        artifacts={
            "b": LockedArtifact(kind="compose", sha256="22"),
            "c": LockedArtifact(kind="compose", sha256="3"),
        },
    )
    diff = compute_diff(old, new)
    by_path = {d.path: d.change for d in diff.artifact_drifts}
    assert by_path == {"a": "removed", "b": "changed", "c": "added"}


# -- CLI: lock is opt-in, --check/diff are read-only --------------------------

def test_lock_check_without_persisted_lock_reports_missing(fresh_manager):
    from linktools.cntr.commands.lock import LockCommand
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.container import ContainerError
    orig = cntr_shared.manager
    cntr_shared.manager = fresh_manager
    try:
        command = LockCommand()

        class _Args:
            check = True
            as_json = False

        with pytest.raises(ContainerError):
            command.run(_Args())
        # --check must never write the lock file itself.
        assert not os.path.exists(fresh_manager.lock_store.path)
    finally:
        cntr_shared.manager = orig


def test_diff_command_is_read_only_and_does_not_write_lock(fresh_manager):
    from linktools.cntr.commands.lock import DiffCommand
    import linktools.cntr.commands._shared as cntr_shared
    orig = cntr_shared.manager
    cntr_shared.manager = fresh_manager
    try:
        command = DiffCommand()

        class _Args:
            as_json = False

        command.run(_Args())
        assert not os.path.exists(fresh_manager.lock_store.path)
    finally:
        cntr_shared.manager = orig


def test_lock_check_on_corrupt_lock_exits_non_zero_distinctly_from_missing(fresh_manager):
    """A corrupt lock file must not be silently treated as "no lock" --
    lock --check must fail loudly (LockInvalid), not report "missing"."""
    from linktools.cntr.commands.lock import LockCommand
    import linktools.cntr.commands._shared as cntr_shared
    orig = cntr_shared.manager
    cntr_shared.manager = fresh_manager
    try:
        _write_raw(fresh_manager, "not json")
        command = LockCommand()

        class _Args:
            check = True
            as_json = False

        with pytest.raises(LockInvalid):
            command.run(_Args())
    finally:
        cntr_shared.manager = orig


def test_diff_on_corrupt_lock_exits_non_zero(fresh_manager):
    from linktools.cntr.commands.lock import DiffCommand
    import linktools.cntr.commands._shared as cntr_shared
    orig = cntr_shared.manager
    cntr_shared.manager = fresh_manager
    try:
        _write_raw(fresh_manager, "not json")
        command = DiffCommand()

        class _Args:
            as_json = False

        with pytest.raises(LockInvalid):
            command.run(_Args())
    finally:
        cntr_shared.manager = orig


def test_doctor_reports_lock_invalid_as_error_finding(fresh_manager):
    from linktools.cntr.doctor import ERROR, LOCK_INVALID, Doctor
    _write_raw(fresh_manager, "not json")
    findings = Doctor(fresh_manager).check_lock()
    assert len(findings) == 1
    assert findings[0].severity == ERROR
    assert findings[0].code == LOCK_INVALID


def test_doctor_run_does_not_crash_on_corrupt_lock(fresh_manager):
    from linktools.cntr.doctor import ERROR, Doctor
    _write_raw(fresh_manager, "not json")
    findings = Doctor(fresh_manager).run()  # must not raise
    assert any(f.severity == ERROR for f in findings)


def test_lock_write_is_atomic_and_skips_unchanged_content(fresh_manager):
    lock = fresh_manager.lock_store.build()
    fresh_manager.lock_store.write(lock)
    path = fresh_manager.lock_store.path
    before_mtime = os.stat(path).st_mtime_ns

    lock_again = fresh_manager.lock_store.build()
    changed = fresh_manager.lock_store.write(lock_again)

    assert changed is False
    assert os.stat(path).st_mtime_ns == before_mtime


def test_up_command_does_not_require_or_touch_lock(fresh_manager, monkeypatch):
    """Lock is fully opt-in; up must work identically
    whether or not a lock file exists, and must never write/modify one."""
    import linktools.cntr.__main__ as cntr_main
    import linktools.cntr.commands._shared as cntr_shared
    from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
    from linktools.cntr.lifecycle.hooks import HookRegistry

    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    def fake_create(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                return 0
        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fake_create)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)

    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False)

    assert not os.path.exists(fresh_manager.lock_store.path)
