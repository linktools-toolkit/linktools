#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Docker/Compose state inspection (Spec Part II).

Wraps ``docker version``/``docker compose version``/``docker compose ps``
through StructuredCommandRunner; never mutates state, never a second
runtime. Only stable, documented ``compose ps`` fields are relied upon --
unknown fields (added by newer Compose minor versions) are ignored, and the
raw/unnormalized JSON is never exposed as public API.
"""
import json
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..manager import ContainerManager


@dataclass(frozen=True)
class DockerEngineVersion:
    client: "str | None"
    server: "str | None"
    api: "str | None"


@dataclass(frozen=True)
class ServiceRuntimeState:
    logical_container: "str | None"
    service: str
    runtime_name: str
    state: str
    health: "str | None"
    image: "str | None"
    exit_code: "int | None"
    labels: "dict[str, str]"


_RUNNING_STATES = ("running", "restarting")


@dataclass(frozen=True)
class ProjectRuntimeState:
    project: str
    services: "tuple[ServiceRuntimeState, ...]"
    backend: str

    @property
    def running_container_names(self) -> "list[str]":
        """Logical container names with >=1 service running/restarting.

        Kept binary on purpose, to back the pre-existing two-valued
        list/get_actual contract; ``ct-cntr status`` shows the full
        running/degraded/exited/missing/unknown aggregation instead.
        """
        names = {
            service.logical_container
            for service in self.services
            if service.logical_container and service.state.lower() in _RUNNING_STATES
        }
        return sorted(names)


def _parse_labels(value) -> "dict[str, str]":
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str) and value:
        labels = {}
        for entry in value.split(","):
            key, sep, val = entry.partition("=")
            if sep:
                labels[key.strip()] = val.strip()
        return labels
    return {}


def _parse_ps_output(text: str) -> "list[dict]":
    """Tolerate a JSON array, a single JSON object, line-delimited JSON
    objects, or empty output -- across observed Compose v2 minor versions."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]

    items: "list[dict]" = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


class DockerInspector:
    """Organizes read-only Docker/Compose queries behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def get_engine_version(self, allow_sudo_prompt: bool = False) -> "DockerEngineVersion":
        process = self.manager.runtime.create_docker_process(
            "version", "--format", "{{json .}}",
            capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        try:
            data = self.manager.structured_runner.execute_json(process, check=False)
        except StructuredCommandError:
            return DockerEngineVersion(client=None, server=None, api=None)
        if not isinstance(data, dict):
            return DockerEngineVersion(client=None, server=None, api=None)
        client = data.get("Client") if isinstance(data.get("Client"), dict) else {}
        server = data.get("Server") if isinstance(data.get("Server"), dict) else {}
        return DockerEngineVersion(
            client=client.get("Version"),
            server=server.get("Version"),
            api=client.get("ApiVersion") or server.get("ApiVersion"),
        )

    def get_compose_version(self, allow_sudo_prompt: bool = False) -> "str | None":
        process = self.manager.runtime.create_docker_process(
            "compose", "version", "--short",
            capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        try:
            result = self.manager.structured_runner.execute_text(process, check=False)
        except StructuredCommandError:
            return None
        if not result.succeeded:
            return None
        return result.stdout.strip() or None

    def get_project_state(
            self,
            containers: "Iterable[BaseContainer]",
            allow_sudo_prompt: bool = False,
    ) -> "ProjectRuntimeState":
        containers = tuple(containers)
        if not containers:
            # Nothing to build a --file set from -- there is trivially
            # nothing running, not an unavailable/unqueryable runtime.
            return ProjectRuntimeState(
                project=self.manager.project_name, services=(), backend=self.manager.container_type,
            )

        service_owners: "dict[str, str]" = {}
        for container in containers:
            for service_name in container.services.keys():
                # Duplicate service names across containers are a deferred
                # issue (Spec section 4); first-registered container wins,
                # matching the existing Compose-merge-dependent behavior.
                service_owners.setdefault(service_name, container.name)

        process = self.manager.runtime.create_docker_compose_process(
            containers, "ps", "--all", "--format", "json",
            capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        result = self.manager.structured_runner.execute_text(process, check=True)

        services = []
        for item in _parse_ps_output(result.stdout):
            service = str(item.get("Service") or "")
            services.append(ServiceRuntimeState(
                logical_container=service_owners.get(service),
                service=service,
                runtime_name=str(item.get("Name") or ""),
                state=str(item.get("State") or ""),
                health=item.get("Health") or None,
                image=item.get("Image") or None,
                exit_code=item.get("ExitCode") if isinstance(item.get("ExitCode"), int) else None,
                labels=_parse_labels(item.get("Labels")),
            ))

        return ProjectRuntimeState(
            project=self.manager.project_name,
            services=tuple(services),
            backend=self.manager.container_type,
        )

    def validate_compose(
            self,
            containers: "Iterable[BaseContainer]",
            allow_sudo_prompt: bool = False,
    ):
        process = self.manager.runtime.create_docker_compose_process(
            tuple(containers), *self.manager.compose_runner.config_args(quiet=True),
            privilege=False, capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        return self.manager.structured_runner.execute_text(process, check=False)

    def preflight_candidates(self, candidate_files: "dict[str, str]") -> str:
        """Validate candidate compose file *content* (path -> text, not yet
        written anywhere real) by rendering it into a temp directory and
        running ``docker compose config --quiet`` there. Never touches the
        real generated files. Returns "passed"/"skipped"/"failed" -- shared
        by Plan preflight and Lock's own preflight so they can't diverge.
        """
        manager = self.manager
        try:
            with tempfile.TemporaryDirectory(prefix="cntr-preflight-") as tmp_dir:
                args = []
                for dest, content in candidate_files.items():
                    if not dest.endswith((".yml", ".yaml")):
                        continue
                    tmp_path = os.path.join(tmp_dir, os.path.basename(dest))
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    args.extend(["--file", tmp_path])
                if not args:
                    return "skipped"
                args.extend(["--project-name", manager.project_name, "config", "--quiet"])
                process = manager.runtime.create_docker_process(
                    "compose", *args, privilege=False, capture_output=True, sudo_non_interactive=True,
                )
                result = manager.structured_runner.execute_text(process, check=False)
                return "passed" if result.succeeded else "failed"
        except (StructuredCommandError, OSError):
            return "failed"
