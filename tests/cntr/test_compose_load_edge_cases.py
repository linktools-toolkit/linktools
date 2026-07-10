#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge cases in docker_compose loading: the Compose `build: <string>`
shorthand, and a compose file whose YAML root isn't a mapping.
"""
import pytest

from linktools.cntr.container import BaseContainer, ContainerError


def _make_container(fresh_manager, tmp_path, compose_body):
    (tmp_path / "docker-compose.yml").write_text(compose_body)
    return BaseContainer(fresh_manager, tmp_path, name="999-edge-case")


def test_build_string_shorthand_is_left_untouched(fresh_manager, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    build: ./custom-context
""")
    service = container.docker_compose["services"]["app"]
    assert service["build"] == "./custom-context"


def test_build_mapping_form_still_gets_context_and_dockerfile(fresh_manager, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    build: {}
""")
    build = container.docker_compose["services"]["app"]["build"]
    assert build["context"] == str(container.get_docker_context_path())
    assert build["dockerfile"] == str(container.get_docker_file_path())


def test_build_omitted_still_gets_default_completion(fresh_manager, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    restart: "no"
""")
    build = container.docker_compose["services"]["app"]["build"]
    assert build["context"] == str(container.get_docker_context_path())
    assert build["dockerfile"] == str(container.get_docker_file_path())


def test_empty_compose_file_yields_no_services(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, "")
    assert container.docker_compose == {}


def test_non_mapping_compose_root_raises_container_error(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ContainerError):
        container.docker_compose
