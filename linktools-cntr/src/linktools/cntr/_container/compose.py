#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose/Dockerfile data loading, default completion, and file output for
a container."""
import hashlib
import os
from typing import TYPE_CHECKING

import yaml

from linktools import utils

if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from ..container import BaseContainer
    from ..manager import ContainerManager


def load_docker_compose(container: "BaseContainer") -> "dict[str, Any] | None":
    # A plain read -- no write happens in this function -- so it needs no
    # transaction of its own. Rendering may resolve settings from another
    # container; holding a store-wide transaction open for the whole render
    # would collide with that container's own transaction (CacheStore's
    # nesting guard is store-wide, not per-namespace).
    mount_paths = container.settings.get("mount_paths", {})
    for name in container.manager.docker_compose_names:
        path = container.get_source_path(name)
        if not os.path.exists(path):
            continue
        data = yaml.safe_load(container.render_template(path))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            from ..container import ContainerError
            raise ContainerError(f"Compose root must be a mapping: {path}")
        if "services" in data and isinstance(data["services"], dict):
            for name, service in data["services"].items():
                if not isinstance(service, dict):
                    continue
                service.setdefault("container_name", f"{container.project_name}-{name}")
                service.setdefault("restart", container.get_config("SERVICE_RESTART_POLICY"))
                service.setdefault("logging", {
                    "driver": container.get_config("SERVICE_LOG_DRIVER"),
                    "options": {
                        "max-size": container.get_config("SERVICE_LOG_MAX_SIZE"),
                    }
                })
                network_mode = service.get("network_mode")
                if not (isinstance(network_mode, str) and (
                    network_mode.startswith("container:")
                    or network_mode.startswith("service:")
                )):
                    service.setdefault("hostname", name)
                if "image" not in service:
                    # Referenced without writing anything -- this function
                    # must stay a pure read (Plan/Doctor/config-list all
                    # access `container.docker_compose` and must never
                    # trigger a Dockerfile write as a side effect). The
                    # real write happens only via write_docker_compose_file,
                    # for actual execution.
                    if container.docker_file:
                        dockerfile = container.get_docker_file_destination()
                        build = service.get("build")
                        if build is None:
                            build = service["build"] = {}
                        # A string `build: ./context` is valid Compose shorthand;
                        # only fill in context/dockerfile for the mapping form.
                        if isinstance(build, dict):
                            build.setdefault("context", str(container.get_docker_context_path()))
                            build.setdefault("dockerfile", str(dockerfile))
                if "env_file" not in service:
                    path = container.get_source_path(".env")
                    if path and os.path.exists(path):
                        service["env_file"] = [str(path)]
                container_paths = mount_paths.get(service.get("container_name"), {})
                if container_paths:
                    volumes = service.setdefault("volumes", [])
                    for container_path in container_paths.values():
                        if container_path not in volumes:
                            volumes.append(container_path)
        if "networks" in data and isinstance(data["networks"], dict):
            networks = data["networks"]
            for name in list(networks.keys()):
                network = networks[name]
                if network is None:
                    network = networks[name] = {}
                if not isinstance(network, dict):
                    continue
                network.setdefault("name", container.get_service_name(name))
        return data
    return None


def load_docker_file(container: "BaseContainer") -> "str | None":
    path = container.get_source_path("Dockerfile")
    if os.path.exists(path):
        return container.render_template(path)
    return None


def get_services(container: "BaseContainer") -> "dict[str, dict[str, Any]]":
    services = utils.get_item(container.docker_compose, "services")
    if not services or not isinstance(services, dict):
        return {}
    return services


def _record_artifact(container: "BaseContainer", destination, kind: str, content: str, source_names) -> None:
    from ..artifacts import sha256_of
    manager = container.manager
    rel_path = os.path.relpath(str(destination), str(manager.data_path))
    source = None
    for name in source_names:
        candidate = container.get_source_path(name)
        if candidate and os.path.exists(candidate):
            source = str(candidate)
            break
    entry = dict(kind=kind, container=container.name, sha256=sha256_of(content), source=source)
    repository = container.repo_context
    if repository is not None and not repository.builtin and repository.url:
        # repo_name + a hash of the (never credential-bearing) root_path --
        # never repository.url itself, which may embed a Git credential
        # (`https://user:token@host/repo.git`). This must stay identifying
        # enough to tell repositories apart without ever being able to leak
        # a secret through the Artifact Index.
        entry["repo_name"] = repository.repo_name
        if repository.root_path:
            entry["repo_id"] = hashlib.sha256(
                str(repository.root_path).encode("utf-8")).hexdigest()[:8]
        revision = _git_revision(manager, repository.root_path)
        if revision is not None:
            entry["repository_revision"] = revision
    manager.artifact_index.record({rel_path: entry})


def _git_revision(manager: "ContainerManager", repo_path) -> "str | None":
    if not repo_path or not os.path.exists(str(repo_path)):
        return None
    from linktools.git import GitRepository
    try:
        repo = GitRepository.open_if_valid(manager.environ, str(repo_path))
        return repo.head_sha() if repo is not None else None
    except Exception:  # noqa: BLE001 - artifact recording must never fail a real write
        return None


def write_docker_compose_file(container: "BaseContainer") -> "Path | None":
    destination = None
    if container.docker_compose:
        from ..artifacts import atomic_write_text_if_changed
        destination = utils.join_path(container.manager.data_path, "compose", f"{container.name}.yml")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # safe_dump (not dump) so non-serializable values raise instead of
        # leaking a Python object tag into the written YAML.
        content = yaml.safe_dump(container.docker_compose, sort_keys=True, allow_unicode=False)
        atomic_write_text_if_changed(destination, content)
        _record_artifact(container, destination, "compose", content, container.manager.docker_compose_names)
        # The compose model's `build.dockerfile` field (see
        # load_docker_compose) may reference docker_file_destination()
        # without that file having been written yet -- real execution
        # needs it to actually exist on disk, so write it alongside the
        # compose file itself. A no-op when this container has no
        # Dockerfile template.
        write_docker_file(container)
    return destination


def docker_file_destination(container: "BaseContainer") -> "Path":
    """Pure path computation, no write -- see
    ``BaseContainer.get_docker_file_destination``."""
    return utils.join_path(container.manager.data_path, "dockerfile", f"{container.name}.Dockerfile")


def write_docker_file(container: "BaseContainer") -> "Path | None":
    destination = None
    if container.docker_file:
        from ..artifacts import atomic_write_text_if_changed
        destination = docker_file_destination(container)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text_if_changed(destination, container.docker_file)
        _record_artifact(container, destination, "dockerfile", container.docker_file, ("Dockerfile",))
    return destination
