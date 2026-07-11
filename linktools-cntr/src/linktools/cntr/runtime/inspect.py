#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Docker/Compose state inspection.

Wraps ``docker version``/``docker compose version``/``docker compose ps
--quiet``/``docker inspect`` through StructuredCommandRunner; never mutates
state, never a second runtime. Actual project state is built from a
Compose-project container id list plus a batch ``docker inspect``, since
``docker inspect`` has one stable JSON-array shape across Docker versions
(unlike ``docker compose ps``'s json/JSON-lines variance).
"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError
from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..manager import ContainerManager


class RuntimeInspectionError(ContainerError):
    pass


class RuntimeInspectionUnavailable(RuntimeInspectionError):
    """Docker/Compose could not be queried at all: missing binary, `sudo -n`
    denial, no compose file for these containers, a timeout, or an
    unparsable container-id list. Callers treat this as "state unknown" and
    may fall back to persisted state."""


class RuntimeInspectionOutputError(RuntimeInspectionError):
    """Docker responded, but its output is structurally invalid (non-array
    root, non-object item). Never treated as "nothing is running" -- that
    would let a corrupted response masquerade as every container having
    disappeared."""


@dataclass(frozen=True)
class DockerEngineVersion:
    client: "str | None"
    server: "str | None"
    api: "str | None"


@dataclass(frozen=True)
class ServiceRuntimeState:
    logical_container: "str | None"
    service: "str | None"
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
    source: str = "docker-inspect"

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


_CONTAINER_ID_RE = re.compile(r"^[0-9a-fA-F]{12,64}$")


def _parse_container_ids(text: str) -> "list[str]":
    """Stably deduped, order-preserving container ids from ``compose ps
    --quiet`` output. An empty result is a legitimate "nothing here" -- only
    a non-empty line that isn't a plausible hex id is an error."""
    ids: "list[str]" = []
    seen = set()
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if not _CONTAINER_ID_RE.match(line):
            raise RuntimeInspectionOutputError(f"Unexpected `compose ps --quiet` output line: {line!r}")
        if line not in seen:
            seen.add(line)
            ids.append(line)
    return ids


def _looks_like_not_found(message: str) -> bool:
    """Best-effort detection of Docker's own "no such object/container"
    message. Not a stable structured error code -- used only to decide
    whether one failed inspect in the recovery path is an ignorable
    disappearance or a real error that must still propagate."""
    lowered = (message or "").lower()
    return "no such" in lowered and ("container" in lowered or "object" in lowered)


def _normalize_state(state_data: dict) -> str:
    status = str(state_data.get("Status") or "").strip().lower()
    if status:
        return status
    if state_data.get("Running") is True:
        return "running"
    if state_data.get("Restarting") is True:
        return "restarting"
    if state_data.get("Paused") is True:
        return "paused"
    if state_data.get("Dead") is True:
        return "dead"
    return "unknown"


def _normalize_exit_code(value) -> "int | None":
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _map_inspect_item(
        item: dict, service_owners: "dict[str, str]", project_name: str,
) -> "ServiceRuntimeState | None":
    """None means "not this project" -- filtered out by the caller."""
    config = item.get("Config")
    if not isinstance(config, dict):
        config = {}
    state_data = item.get("State")
    if not isinstance(state_data, dict):
        state_data = {}
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        labels = {}

    if labels.get("com.docker.compose.project") != project_name:
        return None

    service = labels.get("com.docker.compose.service") or None
    health_data = state_data.get("Health")
    health = health_data.get("Status") if isinstance(health_data, dict) else None

    name = str(item.get("Name") or "")
    if name.startswith("/"):
        name = name[1:]

    return ServiceRuntimeState(
        logical_container=service_owners.get(service) if service else None,
        service=service,
        runtime_name=name,
        state=_normalize_state(state_data),
        health=health,
        image=config.get("Image"),
        exit_code=_normalize_exit_code(state_data.get("ExitCode")),
        labels={str(k): str(v) for k, v in labels.items()},
    )


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

    def _list_project_container_ids(
            self, containers: "Iterable[BaseContainer]", allow_sudo_prompt: bool = False,
    ) -> "list[str]":
        process = self.manager.runtime.create_docker_compose_process(
            containers, "ps", "--all", "--quiet",
            capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        result = self.manager.structured_runner.execute_text(process, check=True)
        return _parse_container_ids(result.stdout)

    def _inspect_containers(self, container_ids: "Iterable[str]", allow_sudo_prompt: bool = False) -> "list[dict]":
        process = self.manager.runtime.create_docker_process(
            "inspect", "--type", "container", *container_ids,
            capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
        )
        data = self.manager.structured_runner.execute_json(process, check=True)
        if not isinstance(data, list):
            raise RuntimeInspectionOutputError("`docker inspect` output root is not a JSON array")
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeInspectionOutputError("`docker inspect` item is not a JSON object")
        return data

    def _inspect_containers_recovering(
            self, container_ids: "Iterable[str]", allow_sudo_prompt: bool = False,
    ) -> "list[dict]":
        """One container may have disappeared between listing ids and
        inspecting them (Spec section 13): retry one id at a time, ignore a
        single "no such container/object", and let any other failure
        propagate immediately."""
        items: "list[dict]" = []
        for container_id in container_ids:
            process = self.manager.runtime.create_docker_process(
                "inspect", "--type", "container", container_id,
                capture_output=True, sudo_non_interactive=not allow_sudo_prompt,
            )
            try:
                result = self.manager.structured_runner.execute_json(process, check=True)
            except StructuredCommandError as exc:
                if _looks_like_not_found(str(exc)):
                    continue
                raise RuntimeInspectionUnavailable(str(exc)) from exc
            except OSError as exc:
                raise RuntimeInspectionUnavailable(str(exc)) from exc
            if isinstance(result, list):
                items.extend(item for item in result if isinstance(item, dict))
            elif isinstance(result, dict):
                items.append(result)
        return items

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
                # issue; first-registered container wins, matching the
                # existing Compose-merge-dependent behavior.
                service_owners.setdefault(service_name, container.name)

        try:
            container_ids = self._list_project_container_ids(containers, allow_sudo_prompt=allow_sudo_prompt)
        except (StructuredCommandError, OSError) as exc:
            raise RuntimeInspectionUnavailable(str(exc)) from exc

        if not container_ids:
            return ProjectRuntimeState(
                project=self.manager.project_name, services=(), backend=self.manager.container_type,
            )

        try:
            items = self._inspect_containers(container_ids, allow_sudo_prompt=allow_sudo_prompt)
        except (StructuredCommandError, OSError):
            # Every id just came from `compose ps`, so an empty recovery
            # result here means every one of them disappeared in the
            # meantime -- a legitimate empty project state, not an error.
            items = self._inspect_containers_recovering(container_ids, allow_sudo_prompt=allow_sudo_prompt)
        else:
            if not items:
                raise RuntimeInspectionOutputError(
                    "`docker inspect` returned no results for known container ids")

        services = []
        for item in items:
            mapped = _map_inspect_item(item, service_owners, self.manager.project_name)
            if mapped is not None:
                services.append(mapped)

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
