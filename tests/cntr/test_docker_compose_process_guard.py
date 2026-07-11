#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""create_docker_compose_process must never invoke `docker compose`
with zero --file arguments.

Regression: with no --file, compose falls back to searching the current
working directory for a compose file. If the targeted containers produced
none (e.g. an installed set of pure-Dockerfile containers with no compose
definitions), that fallback could hit a completely unrelated project in
whatever directory the command happened to be run from.
"""
import pytest

from linktools.cntr.container import ContainerError


class _NoComposeContainer:
    def get_docker_compose_file(self):
        return None


def test_empty_container_list_raises_instead_of_running_bare_compose(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(fresh_manager.runtime, "create_process", lambda *a, **k: calls.append(a))

    with pytest.raises(ContainerError):
        fresh_manager.runtime.create_docker_compose_process([], "up")

    assert calls == []


def test_containers_without_compose_files_raise_instead_of_running_bare_compose(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(fresh_manager.runtime, "create_process", lambda *a, **k: calls.append(a))

    with pytest.raises(ContainerError):
        fresh_manager.runtime.create_docker_compose_process([_NoComposeContainer(), _NoComposeContainer()], "up")

    assert calls == []


def test_container_with_compose_file_still_runs(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(fresh_manager.runtime, "create_process", lambda *a, **k: (calls.append(a), None)[1])

    container = fresh_manager.containers["portainer"]
    fresh_manager.runtime.create_docker_compose_process([container], "up")

    assert calls
    assert "--file" in calls[0]
