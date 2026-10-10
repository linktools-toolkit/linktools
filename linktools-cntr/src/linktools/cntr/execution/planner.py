#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionPlanner builds an ``ExecutionPlan`` describing
what an ``up``/``restart``/``down`` would do, without doing any of it.

Reuses existing logic rather than re-implementing it: ``ComposeOperations
.select`` for target resolution, ``ComposeRunner.build_args``/``up_args``/
``config_args`` for Compose argument construction, and the Hook Registry's
own ``iter_phase`` for hook description. Plan never calls
``compose_runner.build/up/stop/down/config`` (the actual subprocess-running
methods), never runs a lifecycle hook, and never writes a generated
artifact or persisted state.
"""
import os
from typing import TYPE_CHECKING

from ..artifacts import collect_candidates, sha256_of
from ..container import ContainerError
from ..runtime.structured import StructuredCommandError, redact_command
from .model import ExecutionPlan, PlannedArtifact, PlannedCommand, PlannedHook

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager

PLAN_SCHEMA_VERSION = 1


class ExecutionPlanner:

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def plan(
            self,
            action: str,
            names: "list[str] | None" = None,
            pull: bool = False,
    ) -> "ExecutionPlan":
        from linktools.core import Config
        with Config.read_only_resolution():
            return self._plan(action, names=names, pull=pull)

    def _plan(
            self,
            action: str,
            names: "list[str] | None" = None,
            pull: bool = False,
    ) -> "ExecutionPlan":
        if action not in ("up", "restart", "down"):
            raise ContainerError(f"Unsupported plan action: {action!r}; expected up/restart/down")

        manager = self.manager
        # Metadata only -- Plan is documented read-only and must never run
        # a third-party container's on_prepare() (arbitrary file writes/
        # network access/hook registration) just to describe what a real
        # up/restart/down would do.
        selection = manager.compose_operations.select(names, metadata_only=True, for_start=action != "down")
        unresolved_selection = False
        try:
            start_selection = manager.compose_operations.start_selection(selection, privilege=False) if action != "down" else selection
        except (StructuredCommandError, OSError):
            unresolved_selection = True
            start_selection = selection

        candidates = collect_candidates(manager, selection.project_containers)
        artifact_index = manager.artifact_index.load()
        artifacts = [
            self._planned_artifact(dest, kind, container_name, content, artifact_index)
            for dest, (kind, container_name, content) in candidates.items()
        ]
        candidate_files = {dest: content for dest, (_, _, content) in candidates.items()}

        # `candidates` (and so `candidate_files`) is already in the same
        # order as `selection.project_containers` -- a real up/restart/down
        # forms its --file set the same way, one container at a time, and
        # Compose's multi-file merge is order-sensitive. Never re-sort this.
        compose_files = [p for p in candidate_files if p.endswith((".yml", ".yaml"))]
        file_args = manager.compose_runner.compose_args(compose_files)[1:]

        commands = []
        services = list(selection.services)
        stop_services = ()
        if not unresolved_selection and action == "restart" and (selection.full or services):
            stop_services = tuple(service for service in start_selection.services
                                  if selection.full or service in services)
            commands.append(self._planned_command("stop", [*file_args, "stop", *stop_services]))
        if not unresolved_selection and action in ("up", "restart"):
            services_to_start = start_selection.services
            for service in services_to_start:
                commands.append(self._planned_command(
                    "up", [*file_args, *manager.compose_runner.apply_service_args(service, remove_orphans=selection.full)]))
        elif action == "down":
            commands.append(self._planned_command("down", [*file_args, "down", *services]))

        hooks = []
        lifecycle_action = "up" if action == "restart" and not selection.full and not services else action
        stop_containers = selection.target_containers
        if action == "restart":
            stop_containers = tuple(container for container in selection.target_containers
                                    if set(container.services).intersection(stop_services))
        steps = () if unresolved_selection else manager.lifecycle.iter_steps(
            lifecycle_action, start_selection.target_containers, stop_containers=stop_containers)
        for step in steps:
            if step.phase is None:
                continue
            owner = step.container if step.container is not None else manager
            # Validate only buckets that dispatch actually visits, without
            # executing the callbacks that may register additional hooks.
            owner.hooks.validate(step.phase)
            ordered = list(owner.hooks.iter_phase(step.phase))
            if step.reverse:
                ordered.reverse()
            for hook in ordered:
                hooks.append(PlannedHook(
                    phase=step.phase.value,
                    container=step.container.name if step.container is not None else None,
                    name=hook.name, opaque=hook.opaque,
                ))

        warnings = []
        if unresolved_selection:
            warnings.append("Compose profile selection could not be resolved; startup commands and hooks "
                            "are omitted until native Compose configuration is available.")
        if action in ("up", "restart"):
            warnings.append("Only selected services, their requirements and running declared consumers are applied. "
                            "Prepared file content changes recreate affected consumers; ordinary configuration "
                            "updates do not invoke a container-specific reload lifecycle. Preparation, image "
                            "availability and native checks are resolved during execution, before restart stops.")
        if action == "restart":
            warnings.append("Restart prepares selected inputs and images, then checks "
                            "them before stopping only the explicitly selected services. Runtime providers "
                            "are included in startup hooks and application, not in the explicit stop set.")
        preflight = "skipped"
        if action in ("up", "restart") and candidate_files:
            preflight = manager.docker_inspector.preflight_candidates(candidate_files)
            if preflight == "failed":
                warnings.append("Compose preflight (docker compose config --quiet) failed")

        return ExecutionPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            action=action,
            project=manager.project_name,
            full=selection.full,
            targets=tuple(c.name for c in selection.target_containers) if not selection.full else (),
            resolved_containers=tuple(c.name for c in selection.project_containers),
            services=tuple(services),
            compose_files=tuple(compose_files),
            artifacts=tuple(artifacts),
            commands=tuple(commands),
            hooks=tuple(hooks),
            warnings=tuple(warnings),
            preflight=preflight,
        )

    def _planned_artifact(self, dest: str, kind: str, container: str, content: str,
                          artifact_index: "dict[str, dict[str, Any]]") -> "PlannedArtifact":
        rel_path = os.path.relpath(dest, str(self.manager.data_path))
        existing = artifact_index.get(rel_path)
        old_sha256 = existing.get("sha256") if existing else None
        new_sha256 = sha256_of(content)
        if old_sha256 is None:
            change = "added"
        elif old_sha256 != new_sha256:
            change = "changed"
        else:
            change = "unchanged"
        return PlannedArtifact(
            path=rel_path, kind=kind, container=container,
            old_sha256=old_sha256, new_sha256=new_sha256, change=change,
        )

    def _planned_command(self, phase: str, compose_args: "list[str]") -> "PlannedCommand":
        # Same builder create_docker_compose_process()/create_docker_process()
        # use, so Plan's argv (file order, --project-name, docker/compose
        # prefix, privilege) can never drift from what actually runs.
        spec = self.manager.runtime.docker_args("compose", *compose_args, privilege=None)
        # Plan only *describes* the command, it never actually runs it.
        display = self.manager.runtime.display_args(spec)
        return PlannedCommand(
            phase=phase,
            args=redact_command(spec.args),
            display_args=redact_command(display),
            privilege=spec.privilege,
            interactive=True,
        )
