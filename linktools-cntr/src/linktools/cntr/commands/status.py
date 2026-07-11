#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr status`` / ``ct-cntr compose status``: read-only aggregated
Docker Compose actual-state display (Spec Part II section 18).

Read-only: never writes persisted state, never runs a lifecycle hook, never
triggers build/up/down. Defaults to non-interactive sudo (``--sudo-prompt``
opts into an interactive password prompt).
"""
import json
from typing import TYPE_CHECKING

from linktools.cli import subcommand, subcommand_argument
from linktools.cli.argparse import LazyChoices
from ..container import ContainerError
from ..runtime.structured import StructuredCommandError
from . import _shared

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager

STATUS_SCHEMA_VERSION = 1
_RUNNING_STATES = ("running", "restarting")


def _aggregate(services: "list[Any]") -> str:
    if not services:
        return "missing"
    states = [s.state.lower() for s in services]
    any_running = any(s in _RUNNING_STATES for s in states)
    all_running = all(s in _RUNNING_STATES for s in states)
    any_unhealthy = any((s.health or "").lower() == "unhealthy" for s in services)
    if all_running and not any_unhealthy:
        return "running"
    if any_running:
        return "degraded"
    return "exited"


def collect_status(
        manager: "ContainerManager",
        names: "list[str] | None" = None,
        sudo_prompt: bool = False,
        all_services: bool = False,
) -> "dict[str, Any]":
    """Build the JSON-shaped status payload; ``render_status`` formats it."""
    try:
        project_containers, state = manager.compose_operations.status(sudo_prompt=sudo_prompt)
        queryable = True
    except StructuredCommandError:
        project_containers = tuple(manager.prepare_installed_containers())
        state = None
        queryable = False

    if names:
        installed_names = {c.name for c in project_containers}
        unknown = [n for n in names if n not in installed_names]
        if unknown:
            raise ContainerError(f"Container(s) not installed: {', '.join(unknown)}")
        target = [c for c in project_containers if c.name in names]
    else:
        target = list(project_containers)

    services_by_container: "dict[str, list[Any]]" = {}
    if state is not None:
        for svc in state.services:
            if svc.logical_container:
                services_by_container.setdefault(svc.logical_container, []).append(svc)

    containers_payload = []
    for container in target:
        if not container.services:
            continue
        found = services_by_container.get(container.name, [])
        status = "unknown" if not queryable else _aggregate(found)
        containers_payload.append(dict(
            container=container.name,
            status=status,
            services=[
                dict(service=svc.service, runtime_name=svc.runtime_name,
                     state=svc.state, health=svc.health)
                for svc in found
            ],
        ))

    orphans = []
    if all_services and state is not None:
        for svc in state.services:
            if svc.logical_container is None:
                orphans.append(dict(service=svc.service, runtime_name=svc.runtime_name,
                                    state=svc.state, health=svc.health))

    return dict(
        schema_version=STATUS_SCHEMA_VERSION,
        project=manager.project_name,
        queryable=queryable,
        containers=containers_payload,
        orphan_services=orphans,
    )


def render_status(logger, payload: "dict[str, Any]") -> None:
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
    """Mixin providing ``status`` -- mounted on both the root Command and
    ComposeCommand so they share one implementation (single implementation
    principle, Spec Part IX)."""

    @subcommand("status", help="show actual Docker Compose status (read-only)")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("--sudo-prompt", dest="sudo_prompt", action="store_true", default=False,
                         help="allow an interactive sudo password prompt (default: never blocks on one)")
    @subcommand_argument("--all-services", dest="all_services", action="store_true", default=False,
                         help="also include services not owned by any installed container")
    def on_command_status(self, names: "list[str]" = None, as_json: bool = False,
                          sudo_prompt: bool = False, all_services: bool = False):
        payload = collect_status(
            _shared.manager, names=names, sudo_prompt=sudo_prompt, all_services=all_services,
        )
        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            render_status(self.logger, payload)
