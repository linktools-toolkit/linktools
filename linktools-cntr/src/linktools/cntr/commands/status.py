#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr status``: read-only aggregated Docker Compose actual-state
display.

Read-only: never writes persisted state, never runs a lifecycle hook, never
triggers build/up/down. If the configured docker type needs sudo, this
blocks on the password prompt like any other docker call.

User-supplied container names are validated against the installed set
BEFORE any Docker query runs (select_status_containers) -- an unknown name
must never trigger a Docker inspect round-trip first. A logical container's
displayed state is computed from its full *expected* service set (declared
by the container, spec-wise: container.services), not just whatever Docker
happened to return -- a declared-but-unobserved service becomes a synthetic
``ServiceStatus(observed=False, state="missing")`` entry that exists only in
this display layer, never written to persisted/runtime state.
"""
import json
from typing import TYPE_CHECKING

from linktools.cli import subcommand, subcommand_argument
from linktools.cli.argparse import LazyChoices
from ..container import ContainerError
from ..runtime.inspect import RuntimeInspectionUnavailable
from . import _shared
from ._order import ROOT_COMMAND_ORDER

if TYPE_CHECKING:
    from typing import Any
    from ..container import BaseContainer
    from ..manager import ContainerManager

STATUS_SCHEMA_VERSION = 1
_RUNNING_STATES = ("running", "restarting")
_EXITED_STATES = ("exited", "dead")


def select_status_containers(
        containers: "list[BaseContainer]", names: "list[str] | None",
) -> "tuple[BaseContainer, ...]":
    """Validate every user-supplied name against the installed set before
    any Docker query runs. Reports every unknown name in one error, not
    just the first. Duplicate names collapse to a single entry, in
    first-seen order. Empty/None ``names`` selects the full project.
    """
    by_name = {container.name: container for container in containers}

    if not names:
        return tuple(containers)

    missing = []
    selected = []
    seen = set()

    for name in names:
        if name not in by_name:
            missing.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        selected.append(by_name[name])

    if missing:
        raise ContainerError("Containers are not installed: %s" % ", ".join(missing))

    return tuple(selected)


class ServiceStatus(object):
    """One service's display-layer status for ``ct-cntr status`` -- a
    thin, Status-only projection over the real ``ServiceRuntimeState``
    (when observed) or a synthetic "declared but not observed" entry.
    Never written to ``ProjectRuntimeState``/``RunningStateStore``/
    ``DockerInspector`` -- it only ever exists inside this module's
    aggregation and text/JSON rendering.
    """

    def __init__(
            self,
            logical_container: "str | None",
            service: "str | None",
            state: str,
            observed: bool,
            runtime_name: "str | None" = None,
            health: "str | None" = None,
            image: "str | None" = None,
            exit_code: "int | None" = None,
    ):
        self.logical_container = logical_container
        self.service = service
        self.state = state
        self.observed = observed
        self.runtime_name = runtime_name
        self.health = health
        self.image = image
        self.exit_code = exit_code


def _aggregate_container_state(service_statuses: "list[ServiceStatus]") -> str:
    """Logical-container state from its full expected service set.

    - missing: no expected service was observed.
    - running: every expected service observed, all running/restarting, none unhealthy.
    - exited: every expected service observed, all exited/dead.
    - degraded: everything else (partial missing, mixed running/exited,
      paused/created/removing/unrecognized state, any unhealthy).

    "unknown" is never returned here -- it is reserved for a wholly
    unqueryable runtime, decided by the caller before this is even called.
    """
    observed = [s for s in service_statuses if s.observed]
    if not observed:
        return "missing"
    if len(observed) < len(service_statuses):
        return "degraded"

    states = [s.state.lower() for s in observed]
    any_unhealthy = any((s.health or "").lower() == "unhealthy" for s in observed)
    all_running = all(s in _RUNNING_STATES for s in states)
    all_exited = all(s in _EXITED_STATES for s in states)

    if all_running and not any_unhealthy:
        return "running"
    if all_exited:
        return "exited"
    return "degraded"


def collect_status(
        manager: "ContainerManager",
        names: "list[str] | None" = None,
        all_services: bool = False,
) -> "dict[str, Any]":
    """Build the JSON-shaped status payload; ``render_status`` formats it.

    Names are validated (select_status_containers) before the Docker query
    runs. Only ``RuntimeInspectionUnavailable`` (runtime unqueryable) is
    caught here and turned into ``queryable=false``/an ``error`` field with
    a default-zero exit code. ``RuntimeInspectionOutputError`` (a
    structurally invalid response) is left to propagate -- a corrupted
    response must never masquerade as "every container is missing"."""
    project_containers = tuple(manager.prepare_installed_containers())
    target = select_status_containers(list(project_containers), names)

    error = None
    try:
        state = manager.docker_inspector.get_project_state(project_containers)
        queryable = True
    except RuntimeInspectionUnavailable as exc:
        state = None
        queryable = False
        error = str(exc)

    services_by_container: "dict[str, list[Any]]" = {}
    if state is not None:
        for svc in state.services:
            if svc.logical_container:
                services_by_container.setdefault(svc.logical_container, []).append(svc)

    containers_payload = []
    for container in target:
        expected = list(container.services.keys())
        if not expected:
            continue
        observed_by_service = {svc.service: svc for svc in services_by_container.get(container.name, [])}

        service_statuses = []
        for service_name in expected:
            observed_svc = observed_by_service.get(service_name)
            if observed_svc is not None:
                service_statuses.append(ServiceStatus(
                    logical_container=container.name, service=service_name,
                    state=observed_svc.state, observed=True,
                    runtime_name=observed_svc.runtime_name, health=observed_svc.health,
                    image=observed_svc.image, exit_code=observed_svc.exit_code,
                ))
            else:
                service_statuses.append(ServiceStatus(
                    logical_container=container.name, service=service_name,
                    state="missing", observed=False,
                ))

        status = "unknown" if not queryable else _aggregate_container_state(service_statuses)
        containers_payload.append(dict(
            container=container.name,
            status=status,
            services=[
                dict(service=s.service, runtime_name=s.runtime_name, state=s.state,
                     health=s.health, observed=s.observed)
                for s in service_statuses
            ],
        ))

    orphans = []
    if all_services and state is not None:
        for svc in state.services:
            if svc.logical_container is None:
                orphans.append(dict(service=svc.service, runtime_name=svc.runtime_name,
                                    state=svc.state, health=svc.health, observed=True))

    payload = dict(
        schema_version=STATUS_SCHEMA_VERSION,
        project=manager.project_name,
        queryable=queryable,
        containers=containers_payload,
        orphan_services=orphans,
    )
    if error is not None:
        payload["error"] = error
    return payload


def render_status(logger, payload: "dict[str, Any]") -> None:
    if not payload["queryable"] and payload.get("error"):
        logger.warning(f"Unable to query actual runtime state: {payload['error']}")

    rows = []
    for entry in payload["containers"]:
        if not entry["services"]:
            rows.append((entry["container"], entry["status"], "-", "-", "-", "-"))
            continue
        for svc in entry["services"]:
            rows.append((
                entry["container"], entry["status"], svc["service"],
                svc["runtime_name"] or "-", svc["state"] or "-", svc["health"] or "-",
            ))
    for svc in payload.get("orphan_services", ()):
        rows.append(("(orphan)", "unknown", svc["service"], svc["runtime_name"] or "-",
                     svc["state"] or "-", svc["health"] or "-"))

    if not rows:
        logger.info("No container found")
        return

    header = ("Container", "Status", "Service", "Runtime name", "State", "Health")
    widths = [
        max(len(str(header[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(header))
    ]
    logger.info("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        logger.info("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


class StatusCommands:
    """Mixin providing the root ``status`` command."""

    @subcommand("status", order=ROOT_COMMAND_ORDER["status"], help="show actual Docker Compose status (read-only)")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("--all-services", dest="all_services", action="store_true", default=False,
                         help="also include services not owned by any installed container")
    def on_command_status(self, names: "list[str]" = None, as_json: bool = False, all_services: bool = False):
        payload = collect_status(_shared.manager, names=names, all_services=all_services)
        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            render_status(self.logger, payload)
