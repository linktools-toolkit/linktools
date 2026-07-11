#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContainerManifestPolicy: cntr's own policy on top of the generic
linktools.core project manifest -- requiring a `components.cntr` block,
its schema_version, host compatibility, repo add cleanup-on-failure, and
pre-import gating in ContainerLoader."""
import json
import os

import pytest

from linktools.cntr.repo.manifest import (
    ContainerIncompatible, ContainerManifestInvalid, ContainerManifestPolicy, ContainerManifestSchemaUnsupported,
)

_VALID = {
    "schema_version": 1,
    "kind": "linktools-project",
    "name": "example",
    "version": "1.2.0",
    "components": {
        "cntr": {
            "schema_version": 1,
            "requires": {},
            "config": {},
            "metadata": {},
            "extensions": {},
        },
    },
}


def _cntr_manifest(requires=None, top_requires=None, component_schema_version=1):
    data = json.loads(json.dumps(_VALID))  # deep copy
    if top_requires is not None:
        data["requires"] = top_requires
    data["components"]["cntr"]["schema_version"] = component_schema_version
    if requires is not None:
        data["components"]["cntr"]["requires"] = requires
    return data


def _write(tmp_path, data):
    if isinstance(data, str):
        (tmp_path / ".linktools.json").write_text(data, encoding="utf-8")
    else:
        (tmp_path / ".linktools.json").write_text(json.dumps(data), encoding="utf-8")


# -- load() / static validation (delegated to linktools.core, spot-checked here) --

def test_missing_manifest_is_legacy_and_returns_none(fresh_manager, tmp_path):
    assert fresh_manager.manifest_policy.load(tmp_path) is None


def test_empty_file_is_invalid(fresh_manager, tmp_path):
    _write(tmp_path, "")
    with pytest.raises(ContainerManifestInvalid):
        fresh_manager.manifest_policy.load(tmp_path)


def test_invalid_json_is_invalid(fresh_manager, tmp_path):
    _write(tmp_path, "{not json")
    with pytest.raises(ContainerManifestInvalid):
        fresh_manager.manifest_policy.load(tmp_path)


def test_unsupported_schema_version_rejected(fresh_manager, tmp_path):
    _write(tmp_path, dict(_VALID, schema_version=2))
    with pytest.raises(ContainerManifestSchemaUnsupported):
        fresh_manager.manifest_policy.load(tmp_path)


def test_old_cntr_kind_is_rejected(fresh_manager, tmp_path):
    """The old cntr-only kind (linktools-cntr-repository) is not accepted
    at all -- there is no dual-format parser or kind alias."""
    _write(tmp_path, dict(_VALID, kind="linktools-cntr-repository"))
    with pytest.raises(ContainerManifestInvalid):
        fresh_manager.manifest_policy.load(tmp_path)


def test_schema_version_1_is_supported(fresh_manager, tmp_path):
    _write(tmp_path, _VALID)
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    assert manifest.schema_version == 1


def test_dollar_schema_field_is_never_fetched(fresh_manager, tmp_path, monkeypatch):
    def fail_on_network(*a, **k):
        raise AssertionError("must never make a network request")

    monkeypatch.setattr("urllib.request.urlopen", fail_on_network, raising=False)
    _write(tmp_path, dict(_VALID, **{"$schema": "https://example.invalid/schema.json"}))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    assert manifest is not None


# -- cntr component gate -----------------------------------------------------

def test_manifest_with_no_cntr_component_is_rejected(fresh_manager, tmp_path):
    data = {"schema_version": 1, "kind": "linktools-project", "components": {
        "ai": {"schema_version": 1, "requires": {}, "config": {}, "metadata": {}, "extensions": {}},
    }}
    _write(tmp_path, data)
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    with pytest.raises(ContainerManifestInvalid):
        fresh_manager.manifest_policy.get_component(manifest)
    with pytest.raises(ContainerManifestInvalid):
        fresh_manager.manifest_policy.ensure_loadable(manifest)


def test_manifest_with_multiple_components_only_uses_cntr(fresh_manager, tmp_path):
    data = _cntr_manifest()
    data["components"]["ai"] = {
        "schema_version": 1, "requires": {"package:linktools-ai": ">=999.0"}, "config": {}, "metadata": {},
        "extensions": {},
    }
    _write(tmp_path, data)
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    fresh_manager.manifest_policy.ensure_loadable(manifest)  # ai's unmet requirement is ignored by cntr


def test_cntr_component_unsupported_schema_version_is_rejected(fresh_manager, tmp_path):
    _write(tmp_path, _cntr_manifest(component_schema_version=2))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    with pytest.raises(ContainerManifestSchemaUnsupported):
        fresh_manager.manifest_policy.get_component(manifest)


# -- host requirement compatibility ------------------------------------------

def test_compatible_cntr_and_python_requirement(fresh_manager, tmp_path):
    import platform
    data = _cntr_manifest(
        requires={"package:linktools-cntr": ">=0.0.1"},
        top_requires={"python": f">={platform.python_version()}"},
    )
    _write(tmp_path, data)
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    assert fresh_manager.manifest_policy.check_host_requirements(manifest) == []
    fresh_manager.manifest_policy.ensure_loadable(manifest)  # must not raise


def test_incompatible_cntr_requirement_blocks_loading(fresh_manager, tmp_path):
    _write(tmp_path, _cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"}))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    issues = fresh_manager.manifest_policy.check_host_requirements(manifest)
    assert len(issues) == 1
    assert issues[0].key == "package:linktools-cntr"
    with pytest.raises(ContainerIncompatible):
        fresh_manager.manifest_policy.ensure_loadable(manifest)


def test_incompatible_python_requirement_blocks_loading(fresh_manager, tmp_path):
    _write(tmp_path, _cntr_manifest(top_requires={"python": ">=99.0"}))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    with pytest.raises(ContainerIncompatible):
        fresh_manager.manifest_policy.ensure_loadable(manifest)


def test_unrecognized_requirement_key_now_blocks_fail_closed(fresh_manager, tmp_path):
    """Breaking change from the old cntr manifest system: an unrecognized
    requirement key used to be reported as INFO only (kept, not enforced);
    the new generic-manifest-backed policy fails closed instead -- the
    manifest declared something this cntr version can't verify, so
    compatibility must never be assumed."""
    _write(tmp_path, _cntr_manifest(requires={"some-other-tool": ">=1.0"}))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    issues = fresh_manager.manifest_policy.check_host_requirements(manifest)
    assert len(issues) == 1
    assert issues[0].key == "some-other-tool"
    with pytest.raises(ContainerIncompatible):
        fresh_manager.manifest_policy.ensure_loadable(manifest)


def test_ensure_loadable_is_noop_for_legacy_repo(fresh_manager, tmp_path):
    fresh_manager.manifest_policy.ensure_loadable(None)  # must not raise


def test_ai_component_unrecognized_requirement_does_not_affect_cntr(fresh_manager, tmp_path):
    data = _cntr_manifest()
    data["components"]["ai"] = {
        "schema_version": 1, "requires": {"some-ai-only-tool": ">=1.0"}, "config": {}, "metadata": {},
        "extensions": {},
    }
    _write(tmp_path, data)
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    fresh_manager.manifest_policy.ensure_loadable(manifest)  # must not raise


# -- runtime requirement compatibility (docker-engine/docker-compose) -------

def test_runtime_requirements_use_prefixed_keys(fresh_manager, tmp_path, monkeypatch):
    _write(tmp_path, _cntr_manifest(requires={"runtime:docker-compose": ">=2.20"}))
    manifest = fresh_manager.manifest_policy.load(tmp_path)
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_compose_version", lambda *a, **k: "2.30.0")
    assert fresh_manager.manifest_policy.check_runtime_requirements(manifest) == []


def test_service_constants():
    assert ContainerManifestPolicy.component_name == "cntr"
    assert ContainerManifestPolicy.supported_component_versions == (1,)


# -- ContainerLoader pre-import gating ----------------------------------------

def _make_local_repo(tmp_path, manifest_data=None, with_compose=True):
    repo_dir = tmp_path / "repo_src"
    repo_dir.mkdir()
    if manifest_data is not None:
        (repo_dir / ".linktools.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    if with_compose:
        (repo_dir / "container.py").write_text(
            "from linktools.cntr.container import BaseContainer\n\n\n"
            "class Container(BaseContainer):\n    pass\n",
            encoding="utf-8",
        )
    return repo_dir


def test_repo_add_cleans_up_on_incompatible_manifest(fresh_manager, tmp_path):
    from linktools.cntr.container import ContainerError
    repo_dir = _make_local_repo(
        tmp_path, manifest_data=_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"}))

    before = dict(fresh_manager.repo_store.get_all())
    with pytest.raises(ContainerError):
        fresh_manager.repo_store.add(str(repo_dir), force=True)

    after = fresh_manager.repo_store.get_all()
    assert after == before
    # The symlink created for the new repo must have been removed again.
    repo_root = fresh_manager.data_path / "repo"
    if repo_root.exists():
        assert not any(
            os.path.realpath(str(repo_root / name)) == str(repo_dir.resolve())
            for name in os.listdir(repo_root)
        )


def test_repo_add_succeeds_for_compatible_manifest(fresh_manager, tmp_path):
    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    assert str(repo_dir.resolve()) in fresh_manager.repo_store.get_all()


def test_repo_add_succeeds_for_legacy_repo_without_manifest(fresh_manager, tmp_path):
    repo_dir = _make_local_repo(tmp_path, manifest_data=None)
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    assert str(repo_dir.resolve()) in fresh_manager.repo_store.get_all()


def test_repo_add_fails_when_manifest_has_no_cntr_component(fresh_manager, tmp_path):
    from linktools.cntr.container import ContainerError
    data = {"schema_version": 1, "kind": "linktools-project", "components": {
        "ai": {"schema_version": 1, "requires": {}, "config": {}, "metadata": {}, "extensions": {}},
    }}
    repo_dir = _make_local_repo(tmp_path, manifest_data=data)
    with pytest.raises(ContainerError):
        fresh_manager.repo_store.add(str(repo_dir), force=True)


def test_incompatible_repo_container_py_is_not_imported(fresh_manager, tmp_path, monkeypatch):
    repo_dir = _make_local_repo(
        tmp_path, manifest_data=_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"}))
    # Bypass RepoStore.add's own manifest gate (already covered above) to
    # simulate a repo that became incompatible after being installed (e.g.
    # this host's cntr was downgraded) -- the loader must still refuse it.
    repos = dict(fresh_manager.repo_store.get_all())
    repos[str(repo_dir)] = dict(type="local", repo_path=str(repo_dir), repo_name="repo_src")
    monkeypatch.setattr(fresh_manager.repo_store, "get_all", lambda: repos)

    import_calls = []
    import linktools.cntr.registry.loader as loader_module
    real_import = loader_module.import_module_file

    def spy_import(name, path):
        import_calls.append(path)
        return real_import(name, path)

    monkeypatch.setattr(loader_module, "import_module_file", spy_import)

    fresh_manager.__dict__.pop("containers", None)
    containers = fresh_manager.containers
    assert str(repo_dir / "container.py") not in import_calls
    assert "example" not in {c.name for c in containers.values()}


# -- RepoStore.update() re-validates the manifest ---------------------------

def test_update_reports_incompatible_when_manifest_becomes_incompatible(fresh_manager, tmp_path, monkeypatch):
    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)

    # Simulate the update pulling in a manifest that is now incompatible.
    (repo_dir / ".linktools.json").write_text(
        json.dumps(_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"})), encoding="utf-8")
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    results = fresh_manager.repo_store.update()

    assert len(results) == 1
    assert results[0].updated is True
    assert results[0].compatible is False
    assert "incompatible" in results[0].error


def test_update_reports_compatible_when_manifest_stays_compatible(fresh_manager, tmp_path, monkeypatch):
    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    results = fresh_manager.repo_store.update()

    assert len(results) == 1
    assert results[0].updated is True
    assert results[0].compatible is True
    assert results[0].error is None


def test_update_does_not_perform_git_rollback(fresh_manager, tmp_path, monkeypatch):
    """update explicitly does not implement automatic Git rollback -- an
    incompatible manifest after update is only reported."""
    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    (repo_dir / ".linktools.json").write_text(
        json.dumps(_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"})), encoding="utf-8")
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    fresh_manager.repo_store.update()  # must not raise or revert the file

    with open(repo_dir / ".linktools.json") as f:
        data = json.load(f)
    assert data["components"]["cntr"]["requires"]["package:linktools-cntr"] == ">=9999.0.0"


def test_update_reports_corrupt_manifest_json_as_incompatible(fresh_manager, tmp_path, monkeypatch):
    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    (repo_dir / ".linktools.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    results = fresh_manager.repo_store.update()

    assert len(results) == 1
    assert results[0].compatible is False
    assert "invalid" in results[0].error


def test_update_aggregates_multiple_repos_without_stopping_at_first_failure(fresh_manager, tmp_path, monkeypatch):
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    good_dir = _make_local_repo(tmp_path / "good", manifest_data=_cntr_manifest())
    bad_dir = _make_local_repo(tmp_path / "bad", manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(good_dir), force=True)
    fresh_manager.repo_store.add(str(bad_dir), force=True)
    # Simulate the update pulling in a manifest that is now incompatible --
    # add() itself would reject an already-incompatible manifest, so this
    # must happen after add() succeeds.
    (bad_dir / ".linktools.json").write_text(
        json.dumps(_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"})), encoding="utf-8")
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    results = fresh_manager.repo_store.update()

    by_url = {r.url: r for r in results}
    assert len(results) == 2
    assert by_url[str(good_dir)].compatible is True
    assert by_url[str(bad_dir)].compatible is False


def test_update_command_exits_non_zero_when_any_repo_incompatible(fresh_manager, tmp_path, monkeypatch):
    from linktools.cntr.commands.repo import RepoCommand
    from linktools.cntr.container import ContainerError
    import linktools.cntr.commands._shared as cntr_shared
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    (repo_dir / ".linktools.json").write_text(
        json.dumps(_cntr_manifest(requires={"package:linktools-cntr": ">=9999.0.0"})), encoding="utf-8")
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    with pytest.raises(ContainerError):
        RepoCommand().on_command_update()


def test_update_command_succeeds_when_all_repos_compatible(fresh_manager, tmp_path, monkeypatch):
    from linktools.cntr.commands.repo import RepoCommand
    import linktools.cntr.commands._shared as cntr_shared
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(repo_dir), force=True)
    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", lambda *a, **k: None)

    RepoCommand().on_command_update()  # must not raise


def test_update_reports_sync_failure_without_stopping_other_repos(fresh_manager, tmp_path, monkeypatch):
    from linktools.cntr.container import ContainerError

    (tmp_path / "good").mkdir()
    (tmp_path / "broken").mkdir()
    good_dir = _make_local_repo(tmp_path / "good", manifest_data=_cntr_manifest())
    broken_dir = _make_local_repo(tmp_path / "broken", manifest_data=_cntr_manifest())
    fresh_manager.repo_store.add(str(good_dir), force=True)
    fresh_manager.repo_store.add(str(broken_dir), force=True)

    real_sync = fresh_manager.repo_store.sync.sync

    def flaky_sync(url, meta, branch=None, reset=False):
        if url == str(broken_dir):
            raise ContainerError("network unreachable")
        return real_sync(url, meta, branch=branch, reset=reset)

    monkeypatch.setattr(fresh_manager.repo_store.sync, "sync", flaky_sync)

    results = fresh_manager.repo_store.update()
    by_url = {r.url: r for r in results}
    assert len(results) == 2
    assert by_url[str(good_dir)].updated is True
    assert by_url[str(broken_dir)].updated is False
    assert "network unreachable" in by_url[str(broken_dir)].error


# -- describe_repository() field completeness -------------------------------

def test_describe_repository_reports_git_revision_and_dirty_state(fresh_manager, tmp_path, monkeypatch):
    from linktools.cntr.repo.status import describe_repository
    import linktools.git as git_module

    class _FakeGitRepository:
        def __init__(self, environ, repo_path):
            pass

        def head_sha(self):
            return "deadbeef"

        def is_dirty(self):
            return True

    monkeypatch.setattr(git_module, "GitRepository", _FakeGitRepository)

    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    meta = {"type": "git", "repo_path": str(repo_dir)}
    info = describe_repository(fresh_manager, "https://example.invalid/repo.git", meta)

    assert info["revision"] == "deadbeef"
    assert info["dirty"] is True


def test_describe_repository_reports_components_and_project_fields(fresh_manager, tmp_path):
    from linktools.cntr.repo.status import describe_repository

    repo_dir = _make_local_repo(tmp_path, manifest_data=_cntr_manifest())
    meta = {"type": "local", "repo_path": str(repo_dir)}
    info = describe_repository(fresh_manager, str(repo_dir), meta)

    assert info["manifest"] == "present"
    assert info["kind"] == "linktools-project"
    assert info["project_name"] == "example"
    assert info["components"] == ["cntr"]
    assert info["cntr_component_schema_version"] == 1
    assert info["compatible"] is True


def test_describe_repository_reports_manifest_error_for_missing_cntr_component(fresh_manager, tmp_path):
    from linktools.cntr.repo.status import describe_repository

    data = {"schema_version": 1, "kind": "linktools-project", "components": {}}
    repo_dir = _make_local_repo(tmp_path, manifest_data=data)
    meta = {"type": "local", "repo_path": str(repo_dir)}
    info = describe_repository(fresh_manager, str(repo_dir), meta)

    assert info["compatible"] is False
    assert "manifest_error" in info
