#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose YAML write safety (refactor spec §5.4).

The compose file written by ``get_docker_compose_file`` must use ``safe_dump``
(sort_keys=True, allow_unicode=False) so the output is byte-identical to the
previous ``yaml.dump`` call but cannot leak Python object tags.
"""
import yaml


def test_compose_write_is_safe_yaml(fresh_manager):
    container = fresh_manager.containers["nginx"]
    path = container.get_docker_compose_file()
    assert path is not None
    text = path.read_text(encoding="utf-8")

    # No Python-specific tags may leak into user-facing compose files.
    assert "!!python/" not in text

    # Byte-identical to safe_dump with the documented parameters.
    expected = yaml.safe_dump(
        container.docker_compose, sort_keys=True, allow_unicode=False
    )
    assert text == expected


def test_compose_write_round_trips(fresh_manager):
    container = fresh_manager.containers["portainer"]
    path = container.get_docker_compose_file()
    text = path.read_text(encoding="utf-8")
    # safe_load must reproduce the in-memory compose exactly.
    assert yaml.safe_load(text) == container.docker_compose
