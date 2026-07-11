#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose/Dockerfile data loading, default completion, and file output for
a container."""
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
                    dockerfile = container.get_docker_file_path()
                    if dockerfile and os.path.exists(dockerfile):
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
    repository = getattr(container, "_repository_context", None)
    if repository is not None and not repository.builtin and repository.url:
        entry["repository_url"] = repository.url
        revision = _git_revision(manager, repository.root_path)
        if revision is not None:
            entry["repository_revision"] = revision
    manager.artifact_index.record({rel_path: entry})


def _git_revision(manager: "ContainerManager", repo_path) -> "str | None":
    if not repo_path or not os.path.exists(str(repo_path)):
        return None
    try:
        from dulwich.errors import NotGitRepository
        from linktools.git import GitRepository
        return GitRepository(manager.environ, str(repo_path)).head_sha()
    except NotGitRepository:
        return None
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
    return destination


def write_docker_file(container: "BaseContainer") -> "Path | None":
    destination = None
    if container.docker_file:
        from ..artifacts import atomic_write_text_if_changed
        destination = utils.join_path(container.manager.data_path, "dockerfile", f"{container.name}.Dockerfile")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text_if_changed(destination, container.docker_file)
        _record_artifact(container, destination, "dockerfile", container.docker_file, ("Dockerfile",))
    return destination
