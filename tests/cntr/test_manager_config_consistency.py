#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manager config-derived value consistency (review P2-06, P2-07)."""
import os


def test_debug_env_var_zero_is_false(fresh_manager, monkeypatch):
    monkeypatch.setenv("DEBUG", "0")
    fresh_manager.env_config.reload()
    assert fresh_manager.debug is False


def test_debug_env_var_false_string_is_false(fresh_manager, monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    fresh_manager.env_config.reload()
    assert fresh_manager.debug is False


def test_debug_env_var_true_string_is_true(fresh_manager, monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    fresh_manager.env_config.reload()
    assert fresh_manager.debug is True


def test_debug_env_var_1_is_true(fresh_manager, monkeypatch):
    monkeypatch.setenv("DEBUG", "1")
    fresh_manager.env_config.reload()
    assert fresh_manager.debug is True


def test_debug_defaults_to_environ_debug_when_unset(fresh_manager, monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    fresh_manager.env_config.reload()
    assert fresh_manager.debug == fresh_manager.environ.debug


# -- P2-06: config-derived properties must not go stale within a process ----

def test_container_type_reflects_a_runtime_override_immediately(fresh_manager):
    before = fresh_manager.container_type
    fresh_manager.env_config.set("DOCKER_TYPE", "docker-rootless")
    assert fresh_manager.container_type == "docker-rootless"
    assert fresh_manager.container_type != before or before == "docker-rootless"


def test_host_reflects_a_runtime_override_immediately(fresh_manager):
    fresh_manager.env_config.set("HOST", "first.example.com")
    assert fresh_manager.host == "first.example.com"
    fresh_manager.env_config.set("HOST", "second.example.com")
    assert fresh_manager.host == "second.example.com"


def test_project_name_reflects_a_runtime_override_immediately(fresh_manager):
    fresh_manager.env_config.set("COMPOSE_PROJECT_NAME", "custom-project")
    assert fresh_manager.project_name == "custom-project"


def test_app_path_reflects_a_runtime_override_immediately(fresh_manager, tmp_path):
    new_path = str(tmp_path / "custom-app")
    fresh_manager.env_config.set("DOCKER_APP_PATH", new_path)
    assert str(fresh_manager.app_path) == os.path.abspath(new_path)


# -- P2-07: container temp_path must not double-nest "container" ------------

def test_container_temp_path_is_not_double_nested(fresh_manager):
    container = fresh_manager.containers["portainer"]
    path = str(container.get_temp_path("x"))

    manager_temp = str(fresh_manager.temp_path)
    assert path == os.path.join(manager_temp, "portainer", "x")
    assert os.path.join("container", "container") not in path
