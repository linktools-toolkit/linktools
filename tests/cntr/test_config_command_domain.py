#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr config get/explain/validate``: config's own
command domain, distinct from `compose config`/`compose validate` -- config
validate must never run `docker compose config`."""
import json

import pytest

import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.commands.config import ConfigCommand
from linktools.cntr.container import ContainerError


@pytest.fixture(autouse=True)
def _patch_shared_manager(fresh_manager, monkeypatch):
    # ConfigCommand's subcommand methods read the module-level `_shared.manager`
    # singleton, not whatever fixture happens to be passed in -- every command
    # test in this file needs it pointed at the isolated fresh_manager.
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)


def test_get_prints_resolved_value(fresh_manager, capsys):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    ConfigCommand().on_command_get(keys=["HOST"], show_secret=True)
    assert "HOST=10.0.0.5" in capsys.readouterr().out


def test_get_supports_multiple_keys(fresh_manager, capsys):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    fresh_manager.env_config.persist("DOCKER_USER", "alice")
    ConfigCommand().on_command_get(keys=["HOST", "DOCKER_USER"], show_secret=True)
    out = capsys.readouterr().out
    assert "HOST=10.0.0.5" in out
    assert "DOCKER_USER=alice" in out


def test_explain_reports_source_and_resolved_value(fresh_manager):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    info = fresh_manager.env_config.explain("HOST")
    assert info["resolved_value"] == "10.0.0.5"
    assert info["selected_source"] == "persistent"


def test_explain_json_output_is_serializable(fresh_manager, capsys):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    ConfigCommand().on_command_explain(key="HOST", as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved_value"] == "10.0.0.5"


def test_validate_passes_when_all_persisted_values_are_well_typed(fresh_manager):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    ConfigCommand().on_command_validate()  # must not raise


def test_validate_json_reports_valid_true_when_clean(fresh_manager, capsys):
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    ConfigCommand().on_command_validate(as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_validate_raises_on_a_malformed_persisted_value(fresh_manager):
    from linktools.core import ConfigField

    # A synthetic field with a real numeric cast: most built-in cntr fields
    # use permissive string/path casts that never fail on arbitrary text, so
    # this is the only reliable way to exercise a cast failure.
    fresh_manager.env_config.define(ConfigField(name="TEST_PORT", cast=int))
    store = fresh_manager.environ._config_store
    store.set("container.TEST_PORT", "not-a-number")
    fresh_manager.env_config.reload()

    with pytest.raises(ContainerError):
        ConfigCommand().on_command_validate()


def test_validate_does_not_invoke_docker_compose(fresh_manager, monkeypatch):
    """config validate must never run `docker compose
    config` -- that's compose validate's job."""
    def fail(*a, **k):
        raise AssertionError("config validate must not create a docker compose process")

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)
    fresh_manager.env_config.persist("HOST", "10.0.0.5")
    ConfigCommand().on_command_validate()


def test_validate_does_not_prompt_for_unconfigured_fields(fresh_manager, monkeypatch):
    import linktools.rich as rich

    def fail(*a, **k):
        raise AssertionError("config validate must not prompt for an unconfigured field")

    monkeypatch.setattr(rich, "prompt", fail)
    monkeypatch.setattr(rich, "choose", fail)
    ConfigCommand().on_command_validate()  # DOCKER_DOWNLOAD_PATH etc. are unset; must not prompt
