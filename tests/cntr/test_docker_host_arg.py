#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""create_docker_process must actually connect to a configured DOCKER_HOST.

Regression: DOCKER_HOST was registered as a config option (and shown by
`ct-cntr config list`), but create_docker_process() never passed it to the
docker CLI in any form -- `ct-cntr config set DOCKER_HOST=tcp://...`
had no effect on which daemon commands actually ran against.

The built-in default ("/var/run/docker.sock") is left unexpressed so
docker-rootless's own `docker context` resolution (which the plain default
string does not itself describe) is not overridden by accident.
"""
import pytest


def _record(monkeypatch, manager):
    calls = []
    monkeypatch.setattr(manager.runtime, "create_process", lambda *a, **k: calls.append(a))
    return calls


def test_default_docker_host_emits_no_explicit_host_arg(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker")
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.create_docker_process("ps")

    assert calls == [("docker", "ps")]


@pytest.mark.parametrize("container_type,flag", [
    ("docker", "-H"),
    ("docker-rootless", "-H"),
])
def test_custom_docker_host_is_passed_to_the_cli(fresh_manager, monkeypatch, container_type, flag):
    monkeypatch.setattr(fresh_manager, "container_type", container_type)
    monkeypatch.setattr(fresh_manager.env_config, "get",
                        lambda key, type=None, default=None: "tcp://10.0.0.1:2376" if key == "DOCKER_HOST" else default)
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.create_docker_process("ps")

    assert calls == [("docker", flag, "tcp://10.0.0.1:2376", "ps")]


def test_bare_socket_path_gets_a_unix_scheme(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager, "container_type", "docker")
    monkeypatch.setattr(fresh_manager.env_config, "get",
                        lambda key, type=None, default=None: "/custom/docker.sock" if key == "DOCKER_HOST" else default)
    calls = _record(monkeypatch, fresh_manager)

    fresh_manager.runtime.create_docker_process("ps")

    assert calls == [("docker", "-H", "unix:///custom/docker.sock", "ps")]


def test_podman_container_type_raises_explicit_error_not_silent_fallback(fresh_manager, monkeypatch):
    """Spec section 48: a legacy DOCKER_TYPE=podman must fail loudly, never
    silently resolve to docker."""
    from linktools.cntr.container import ContainerError
    monkeypatch.setattr(fresh_manager, "container_type", "podman")
    calls = _record(monkeypatch, fresh_manager)

    with pytest.raises(ContainerError, match="Podman is no longer supported"):
        fresh_manager.runtime.create_docker_process("ps")

    assert calls == []
