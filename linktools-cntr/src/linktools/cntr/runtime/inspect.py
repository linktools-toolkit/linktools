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
from .structured import StructuredCommandError, StructuredCommandOutputError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..manager import ContainerManager


class RuntimeInspectionError(ContainerError):
    pass


class RuntimeInspectionUnavailable(RuntimeInspectionError):
    """Docker/Compose could not be queried at all: missing binary, a denied
    sudo policy, no compose file for these containers, a timeout, or an
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


def validate_inspect_payload(payload, expected_non_empty: bool = False) -> "list[dict]":
    """Shared shape validator for both the batch and per-ID ``docker
    inspect`` paths: root must be a JSON array, every item must be a JSON
    object. ``expected_non_empty=True`` additionally rejects a
    structurally-valid-but-empty array -- a successful (non-error) inspect
    of a known, non-empty id list/single id returning ``[]`` is corrupted
    output, not a legitimate "nothing here" (a genuine disappearance is
    only ever signalled by the command itself failing, handled separately
    by the recovery path)."""
    if not isinstance(payload, list):
        raise RuntimeInspectionOutputError("`docker inspect` output must be a JSON array")
    result = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeInspectionOutputError("`docker inspect` item %d must be an object" % index)
        result.append(item)
    if expected_non_empty and not result:
        raise RuntimeInspectionOutputError("`docker inspect` returned no objects")
    return result


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

    def get_engine_version(self) -> "DockerEngineVersion":
        process = self.manager.runtime.create_docker_process(
            "version", "--format", "{{json .}}",
            capture_output=True,
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

    def get_compose_version(self) -> "str | None":
        process = self.manager.runtime.create_docker_process(
            "compose", "version", "--short",
            capture_output=True,
        )
        try:
            result = self.manager.structured_runner.execute_text(process, check=False)
        except StructuredCommandError:
            return None
        if not result.succeeded:
            return None
        return result.stdout.strip() or None

    def _list_project_container_ids(self, containers: "Iterable[BaseContainer]") -> "list[str]":
        process = self.manager.runtime.create_docker_compose_process(
            containers, "ps", "--all", "--quiet",
            capture_output=True,
        )
        result = self.manager.structured_runner.execute_text(process, check=True)
        return _parse_container_ids(result.stdout)

    def _inspect_containers(self, container_ids: "Iterable[str]") -> "list[dict]":
        process = self.manager.runtime.create_docker_process(
            "inspect", "--type", "container", *container_ids,
            capture_output=True,
        )
        try:
            payload = self.manager.structured_runner.execute_json(process, check=True)
        except StructuredCommandOutputError as exc:
            # The command ran but produced unparsable JSON -- this is a
            # corrupted response, never a signal that a container
            # disappeared. Translating it to RuntimeInspectionOutputError
            # here (instead of letting the StructuredCommandError subclass
            # propagate) keeps it out of get_project_state's
            # StructuredCommandError recovery branch below.
            raise RuntimeInspectionOutputError(str(exc)) from exc
        return validate_inspect_payload(payload, expected_non_empty=False)

    def _inspect_containers_recovering(self, container_ids: "Iterable[str]") -> "list[dict]":
        """One container may have disappeared between listing ids and
        inspecting them: retry one id at a time, ignore a
        single "no such container/object", and let any other failure
        propagate immediately."""
        items: "list[dict]" = []
        for container_id in container_ids:
            process = self.manager.runtime.create_docker_process(
                "inspect", "--type", "container", container_id,
                capture_output=True,
            )
            try:
                payload = self.manager.structured_runner.execute_json(process, check=True)
            except StructuredCommandOutputError as exc:
                raise RuntimeInspectionOutputError(str(exc)) from exc
            except StructuredCommandError as exc:
                if _looks_like_not_found(str(exc)):
                    continue
                raise RuntimeInspectionUnavailable(str(exc)) from exc
            except OSError as exc:
                raise RuntimeInspectionUnavailable(str(exc)) from exc
            # A successful (non-error) single-id inspect that returns an
            # empty array is NOT a legitimate "not found" signal -- a real
            # disappearance is always a non-zero exit, caught above. An
            # empty-but-successful result here is corrupted output.
            items.extend(validate_inspect_payload(payload, expected_non_empty=True))
        return items

    def get_project_state(self, containers: "Iterable[BaseContainer]") -> "ProjectRuntimeState":
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
            container_ids = self._list_project_container_ids(containers)
        except (StructuredCommandError, OSError) as exc:
            raise RuntimeInspectionUnavailable(str(exc)) from exc

        if not container_ids:
            return ProjectRuntimeState(
                project=self.manager.project_name, services=(), backend=self.manager.container_type,
            )

        try:
            items = self._inspect_containers(container_ids)
        except OSError as exc:
            # The process itself couldn't even run -- never a signal that a
            # container disappeared, so this never enters recovery.
            raise RuntimeInspectionUnavailable(str(exc)) from exc
        except StructuredCommandError:
            # A real command failure (non-zero exit; _inspect_containers
            # already turned an invalid-JSON response into
            # RuntimeInspectionOutputError before it could reach here, so
            # this branch is never entered for corrupted output). Every id
            # just came from `compose ps`, so an empty recovery result here
            # means every one of them disappeared in the meantime -- a
            # legitimate empty project state, not an error.
            items = self._inspect_containers_recovering(container_ids)
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

    def validate_compose(self, containers: "Iterable[BaseContainer]"):
        process = self.manager.runtime.create_docker_compose_process(
            tuple(containers), *self.manager.compose_runner.config_args(quiet=True),
            privilege=False, capture_output=True,
        )
        return self.manager.structured_runner.execute_text(process, check=False)

    def preflight_candidates(self, candidate_files: "dict[str, str]") -> str:
        """Validate candidate compose file *content* (path -> text, not yet
        written anywhere real) by rendering it into a temp directory and
        running ``docker compose config --quiet`` there. Never touches the
        real generated files. Returns "passed"/"skipped"/"failed"."""
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
                    "compose", *args, privilege=False, capture_output=True,
                )
                result = manager.structured_runner.execute_text(process, check=False)
                return "passed" if result.succeeded else "failed"
        except (StructuredCommandError, OSError):
            return "failed"
