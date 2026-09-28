#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge cases in docker_compose loading: the Compose `build: <string>`
shorthand, and a compose file whose YAML root isn't a mapping.

linktools-cntr does not resolve, normalize, or otherwise interpret paths an
author writes in their own Compose (build/env_file/volumes) -- those are
passed through verbatim. It only fills in defaults the author omitted
entirely (a Dockerfile-based build, or a source-directory .env).
"""
import pytest

from linktools.cntr.container import BaseContainer, ContainerError


def _make_container(fresh_manager, tmp_path, compose_body):
    (tmp_path / "docker-compose.yml").write_text(compose_body)
    return BaseContainer(fresh_manager, tmp_path, name="999-edge-case")


def test_build_string_is_preserved(fresh_manager, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    build: ./custom-context
""")
    service = container.docker_compose["services"]["app"]
    assert service["build"] == "./custom-context"


def test_build_context_is_preserved(fresh_manager, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    build:
      context: ./custom-context
""")
    build = container.docker_compose["services"]["app"]["build"]
    assert build["context"] == "./custom-context"


def test_existing_env_file_string_is_preserved(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    env_file: ./extra.env
""")
    service = container.docker_compose["services"]["app"]
    assert service["env_file"] == "./extra.env"


def test_existing_env_file_list_is_preserved(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    env_file:
      - ./extra.env
      - /already/absolute.env
""")
    service = container.docker_compose["services"]["app"]
    assert service["env_file"] == ["./extra.env", "/already/absolute.env"]


def test_existing_volume_short_syntax_is_preserved(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    volumes:
      - ./data:/data
      - ../shared:/shared
      - named-volume:/var/lib/data
      - ${HOST_PATH}:/mnt/host
      - /already/absolute:/absolute
""")
    service = container.docker_compose["services"]["app"]
    assert service["volumes"] == [
        "./data:/data",
        "../shared:/shared",
        "named-volume:/var/lib/data",
        "${HOST_PATH}:/mnt/host",
        "/already/absolute:/absolute",
    ]


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


def test_build_string_shorthand_does_not_raise_when_defaulting_dockerfile(fresh_manager, tmp_path):
    # `image` absent + `build` already a string shorthand: default
    # completion only fills in the mapping form, never touches a string.
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    build: ./custom-context
""")
    service = container.docker_compose["services"]["app"]
    assert service["build"] == "./custom-context"


def test_missing_env_file_still_auto_injects_source_dir_dot_env(fresh_manager, tmp_path):
    (tmp_path / ".env").write_text("FOO=bar\n")
    container = _make_container(fresh_manager, tmp_path, """\
services:
  app:
    image: nginx
""")
    service = container.docker_compose["services"]["app"]
    assert service["env_file"] == [str(tmp_path / ".env")]


def test_empty_compose_file_yields_no_services(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, "")
    assert container.docker_compose == {}


def test_non_mapping_compose_root_raises_container_error(fresh_manager, tmp_path):
    container = _make_container(fresh_manager, tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ContainerError):
        container.docker_compose
