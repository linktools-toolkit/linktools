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
from ..integration import consumer_type, order_services
from ..runtime.structured import redact_command
from .model import ExecutionPlan, PlannedArtifact, PlannedCommand, PlannedHook

if TYPE_CHECKING:
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
        start_selection = manager.compose_operations.start_selection(selection) if action != "down" else selection

        candidates = collect_candidates(manager, selection.project_containers)
        artifacts = [
            self._planned_artifact(dest, kind, container_name, content)
            for dest, (kind, container_name, content) in candidates.items()
        ]
        candidate_files = {dest: content for dest, (_, _, content) in candidates.items()}

        # `candidates` (and so `candidate_files`) is already in the same
        # order as `selection.project_containers` -- a real up/restart/down
        # forms its --file set the same way, one container at a time, and
        # Compose's multi-file merge is order-sensitive. Never re-sort this.
        compose_files = [p for p in candidate_files if p.endswith((".yml", ".yaml"))]
        file_args = []
        for path in compose_files:
            file_args.extend(["--file", path])
        file_args.extend(["--project-name", manager.project_name])

        commands = []
        services = list(selection.services)
        if action == "restart":
            commands.append(self._planned_command("stop", [*file_args, "stop", *services]))
        if action in ("up", "restart"):
            services_to_start = order_services(start_selection.project_containers, start_selection.services)
            for service in services_to_start:
                commands.append(self._planned_command(
                    "up", [*file_args, *manager.compose_runner.apply_service_args(service, remove_orphans=selection.full)]))
        elif action == "down":
            commands.append(self._planned_command("down", [*file_args, "down", *services]))

        hooks = []
        for step in manager.lifecycle.iter_steps(action, selection.target_containers):
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
        if action in ("up", "restart"):
            warnings.append("Configuration is reconciled across the complete installed project. "
                            "Other running services with pending configuration changes may also be updated; "
                            "unrelated stopped services stay stopped. Runtime inspection and native "
                            "candidate validation determine those additional updates during execution.")
            sync = manager.compose_operations.sync_selection(start_selection)
            for container in sync:
                if container.name in manager.generated_configs:
                    warnings.append("{}: generated candidate native validation is pending execution; "
                                    "no hooks, secrets or generated files were prepared".format(container.name))
            for container in sync:
                warnings.extend(consumer_type(container).plan_warnings())
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

    def _planned_artifact(self, dest: str, kind: str, container: str, content: str) -> "PlannedArtifact":
        rel_path = os.path.relpath(dest, str(self.manager.data_path))
        existing = self.manager.artifact_index.load().get(rel_path)
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
