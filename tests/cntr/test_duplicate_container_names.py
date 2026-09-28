#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duplicate container names must fail loudly, never silently overwrite
(review P1-08).

Before this, `ContainerManager.containers` kept whichever container loaded
last for a given name (only a debug-level log noted the collision) --
dependency resolution, config, artifact, and compose paths could then all
silently point at the wrong implementation depending on scan/install order.
"""
import os

import _harness
from linktools.cntr.container import ContainerError


def _fresh_standalone_manager(tmp_path):
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    return ContainerManager(Environ(), name="aio")


def _python_container_repo(tmp_path, repo_name, container_name):
    repo = tmp_path / repo_name
    c_dir = repo / f"100-{container_name}"
    c_dir.mkdir(parents=True)
    (c_dir / "container.py").write_text(
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    pass\n",
        encoding="utf-8",
    )
    return repo


def _simple_container_repo(tmp_path, repo_name, container_name):
    repo = tmp_path / repo_name
    c_dir = repo / f"100-{container_name}"
    c_dir.mkdir(parents=True)
    (c_dir / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    return repo


def test_third_party_repo_colliding_with_builtin_name_raises(tmp_path):
    # "nginx" is a real builtin container name.
    repo = _python_container_repo(tmp_path, "repo", "nginx")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))

    try:
        manager.containers
        raised = False
    except ContainerError as exc:
        raised = True
        message = str(exc)

    assert raised
    assert "nginx" in message
    assert "builtin" in message


def test_two_different_repos_same_container_name_raises(tmp_path):
    repo_a = _python_container_repo(tmp_path, "repo_a", "dup")
    repo_b = _python_container_repo(tmp_path, "repo_b", "dup")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))

    try:
        manager.containers
        raised = False
    except ContainerError as exc:
        raised = True
        message = str(exc)

    assert raised
    assert "dup" in message
    assert "repo_a" in message
    assert "repo_b" in message


def test_same_repo_two_directories_same_name_raises(tmp_path):
    repo = tmp_path / "repo"
    for sub in ("group1", "group2"):
        c_dir = repo / sub / "100-dup"
        c_dir.mkdir(parents=True)
        (c_dir / "container.py").write_text(
            "from linktools.cntr import BaseContainer\n"
            "class Container(BaseContainer):\n"
            "    pass\n",
            encoding="utf-8",
        )
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))

    try:
        manager.containers
        raised = False
    except ContainerError:
        raised = True

    assert raised


def test_simple_container_colliding_with_python_container_raises(tmp_path):
    repo_a = _python_container_repo(tmp_path, "repo_a", "dup")
    repo_b = _simple_container_repo(tmp_path, "repo_b", "dup")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))

    try:
        manager.containers
        raised = False
    except ContainerError:
        raised = True

    assert raised


def test_error_lists_both_origins_neither_silently_kept(tmp_path):
    repo_a = _python_container_repo(tmp_path, "repo_a", "dup")
    repo_b = _python_container_repo(tmp_path, "repo_b", "dup")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_a))
    manager.repos.add(str(repo_b))

    try:
        manager.containers
        message = None
    except ContainerError as exc:
        message = str(exc)

    assert message is not None
    # Both repo_names appear -- the error is not a one-sided "kept A" report.
    assert "repo_a" in message and "repo_b" in message
