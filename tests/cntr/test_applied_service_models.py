#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Last-applied service models retain resolved values and partial apply state."""
import copy
import stat
import traceback
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr.artifacts import AppliedServiceModels, sha256_of
from linktools.cntr.container import ContainerError


@pytest.fixture
def manager(tmp_path):
    entries = {}
    return SimpleNamespace(data_path=tmp_path, entries=entries,
                           artifact_index=SimpleNamespace(record=entries.update))


def _model(value="old", network="bridge"):
    return {"name": "project", "services": {
        "app": {"image": "app:1", "environment": {"TOKEN": value}},
        "worker": {"image": "worker:1", "environment": {"TOKEN": value}},
    }, "networks": {"default": {"driver": network}}}


def _path(manager, service):
    return manager.data_path / "compose" / "applied" / "services" / (service.encode().hex() + ".yml")


def test_environment_change_and_missing_snapshot_are_changed(manager):
    initial = AppliedServiceModels(manager, _model())
    assert initial.changed_services == frozenset({"app", "worker"})
    initial.record(("app", "worker"))
    changed = _model()
    changed["services"]["app"]["environment"]["TOKEN"] = "new"
    assert AppliedServiceModels(manager, changed).changed_services == frozenset({"app"})


def test_decoded_previous_model_cannot_mutate_snapshot_or_other_reads(manager):
    AppliedServiceModels(manager, _model()).record(("app",))
    store = AppliedServiceModels(manager, _model("new"))
    original = store.previous["app"]
    decoded = store.previous_model("app")
    decoded["services"]["app"]["environment"]["TOKEN"] = "mutated"
    assert store.previous["app"] == original
    assert store.previous_model("app") == _model()
    assert store.previous_model("worker") is None


def test_normalization_ignores_mapping_order_and_yaml_formatting(manager):
    initial = AppliedServiceModels(manager, _model())
    initial.record(("app", "worker"))
    saved = yaml.safe_load(initial.current["app"])
    _path(manager, "app").write_text(yaml.safe_dump(saved, sort_keys=False, default_flow_style=True))
    model = _model()
    model["services"]["app"] = dict(reversed(tuple(model["services"]["app"].items())))
    assert not AppliedServiceModels(manager, dict(reversed(tuple(model.items())))).changed_services


def test_shared_network_change_marks_all_services_changed(manager):
    initial = AppliedServiceModels(manager, _model())
    initial.record(("app", "worker"))
    assert AppliedServiceModels(manager, _model(network="host")).changed_services == frozenset({"app", "worker"})


def test_partial_record_preserves_stopped_sibling_and_its_shared_definitions(manager):
    initial = AppliedServiceModels(manager, _model())
    initial.record(("app", "worker"))
    old_worker = _path(manager, "worker").read_text()
    current = AppliedServiceModels(manager, _model(value="new", network="host"))
    current.record(("app",))
    assert _path(manager, "worker").read_text() == old_worker
    assert AppliedServiceModels(manager, _model(value="new", network="host")).changed_services == frozenset({"worker"})
    assert yaml.safe_load(_path(manager, "app").read_text()) == _model(value="new", network="host")


def test_other_service_change_does_not_mark_unchanged_target_dirty(manager):
    initial = AppliedServiceModels(manager, _model())
    initial.record(("app", "worker"))
    model = _model()
    model["services"]["worker"]["image"] = "worker:2"
    current = AppliedServiceModels(manager, model)
    assert current.current["app"] != current.previous["app"]
    assert current.changed_services == frozenset({"worker"})


def test_saved_project_retains_removed_dependencies_for_rollback(manager):
    model = _model()
    model["services"]["app"]["depends_on"] = {"worker": {"condition": "service_started"}}
    AppliedServiceModels(manager, model).record(("app",))
    candidate = copy.deepcopy(model)
    candidate["services"].pop("worker")
    candidate["services"]["app"].pop("depends_on")
    current = AppliedServiceModels(manager, candidate)
    saved = yaml.safe_load(current.previous["app"])
    assert saved == model
    assert set(saved["services"]["app"]["depends_on"]) <= set(saved["services"])
    assert yaml.safe_load(current.current["app"]) == candidate
    assert current.changed_services == frozenset({"app"})


def test_retained_service_is_previous_only_and_does_not_change_desired_services(manager):
    AppliedServiceModels(manager, _model()).record(("app", "worker"))
    desired = _model()
    desired["services"].pop("worker")
    current = AppliedServiceModels(manager, desired, retained_services=("app", "worker", "worker"))
    assert set(current.current) == {"app"}
    assert set(current.previous) == {"app", "worker"}
    assert not current.changed_services
    assert not current.untracked_services
    desired["services"]["app"]["environment"]["TOKEN"] = "new"
    current.set_model(desired)
    assert set(current.current) == {"app"}
    assert current.changed_services == {"app"}
    assert yaml.safe_load(current.previous["worker"]) == _model()


def test_missing_retained_snapshot_is_untracked_but_not_desired(manager):
    current = AppliedServiceModels(manager, _model(), retained_services=("app", "orphan"))
    assert current.untracked_services == {"app", "orphan"}
    assert set(current.current) == {"app", "worker"}
    assert current.changed_services == {"app", "worker"}
    assert not current.previous


def test_removed_service_snapshot_is_not_retained_without_observation(manager):
    AppliedServiceModels(manager, _model()).record(("app", "worker"))
    desired = _model()
    desired["services"].pop("worker")
    current = AppliedServiceModels(manager, desired, retained_services=("app",))
    assert set(current.previous) == {"app"}
    assert not current.untracked_services


def test_previous_environment_stays_resolved_after_external_file_changes(manager, tmp_path):
    env_file = tmp_path / "app.env"
    env_file.write_text("TOKEN=old\n")
    initial = AppliedServiceModels(manager, _model(value=env_file.read_text().strip().split("=", 1)[1]))
    initial.record(("app",))
    env_file.write_text("TOKEN=new\n")
    model = _model(value=env_file.read_text().strip().split("=", 1)[1])
    current = AppliedServiceModels(manager, model)
    model["services"]["app"]["environment"]["TOKEN"] = "later"
    assert yaml.safe_load(current.previous["app"])["services"]["app"]["environment"] == {"TOKEN": "old"}
    assert yaml.safe_load(current.current["app"])["services"]["app"]["environment"] == {"TOKEN": "new"}
    with pytest.raises(TypeError):
        current.current["app"] = "mutated"


def test_snapshots_are_private_and_indexed_even_when_replacing_permissive_file(manager):
    initial = AppliedServiceModels(manager, _model())
    initial.record(("app",))
    path = _path(manager, "app")
    path.chmod(0o644)
    path.parent.chmod(0o755)
    initial.record(("app",))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    entry = manager.entries[str(path.relative_to(manager.data_path))]
    assert entry == {"kind": "compose-applied-service", "container": "app",
                     "sha256": sha256_of(initial.current["app"])}


@pytest.mark.parametrize("content", ["[", "[]", "{}", "services: []", "services: {worker: {}}", "services: {app: null}"])
def test_malformed_saved_state_fails_closed(manager, content):
    path = _path(manager, "app")
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(ContainerError, match="[Cc]ompose model"):
        AppliedServiceModels(manager, _model())


def test_unknown_record_service_does_not_partially_write(manager):
    current = AppliedServiceModels(manager, _model())
    with pytest.raises(ContainerError, match="absent"):
        current.record(("app", "missing"))
    assert not _path(manager, "app").exists()


def test_malformed_snapshot_error_does_not_disclose_environment_values(manager):
    path = _path(manager, "app")
    path.parent.mkdir(parents=True)
    path.write_text("services: {app: {environment: {TOKEN: private-value")
    with pytest.raises(ContainerError) as caught:
        AppliedServiceModels(manager, _model())
    rendered = "".join(traceback.format_exception(type(caught.value), caught.value, caught.tb))
    assert "private-value" not in rendered


def test_dangling_snapshot_link_fails_closed(manager):
    path = _path(manager, "app")
    path.parent.mkdir(parents=True)
    path.symlink_to(path.parent / "missing")
    with pytest.raises(ContainerError, match="Cannot read"):
        AppliedServiceModels(manager, _model())


def test_yaml_alias_identity_does_not_affect_normalization(manager):
    model = _model()
    values = {"TOKEN": "old"}
    model["services"]["app"].update(environment=values, labels=values)
    AppliedServiceModels(manager, model).record(("app", "worker"))
    model["services"]["app"]["labels"] = copy.deepcopy(values)
    assert not AppliedServiceModels(manager, model).changed_services


def test_unreferenced_named_volume_does_not_recreate_other_services(manager):
    model = _model()
    model["services"]["app"]["volumes"] = [
        {"type": "volume", "source": "appdata", "target": "/data"}]
    model["volumes"] = {"appdata": {"name": "appdata-v1"}}
    AppliedServiceModels(manager, model).record(("app", "worker"))

    updated = copy.deepcopy(model)
    updated["volumes"]["appdata"]["name"] = "appdata-v2"
    assert AppliedServiceModels(manager, updated).changed_services == frozenset({"app"})

    updated["networks"]["default"]["driver"] = "overlay"
    assert AppliedServiceModels(manager, updated).changed_services == frozenset({"app", "worker"})


def test_secrets_and_configs_affect_only_referencing_service(manager):
    model = _model()
    model["secrets"] = {"token": {"file": "/host/token-v1"}}
    model["configs"] = {"rules": {"file": "/host/rules-v1"}}
    model["services"]["app"]["secrets"] = [{"source": "token", "target": "api_token"}]
    model["services"]["app"]["configs"] = [{"source": "rules", "target": "/etc/rules"}]
    AppliedServiceModels(manager, model).record(("app", "worker"))
    changed = copy.deepcopy(model)
    changed["secrets"]["token"]["file"] = "/host/token-v2"
    changed["configs"]["rules"]["file"] = "/host/rules-v2"
    assert AppliedServiceModels(manager, changed).changed_services == frozenset({"app"})
