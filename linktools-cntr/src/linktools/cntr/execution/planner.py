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
from ..lifecycle.hooks import HookPhase
from ..runtime.structured import redact_command
from .model import ExecutionPlan, PlannedArtifact, PlannedCommand, PlannedHook

if TYPE_CHECKING:
    from ..manager import ContainerManager

PLAN_SCHEMA_VERSION = 1

# Hook phases each action's dispatch touches, in the same order LifecycleDispatcher uses.
_ACTION_PHASES = {
    "up": (HookPhase.CHECK, HookPhase.BEFORE_START, HookPhase.AFTER_START),
    "restart": (
        HookPhase.BEFORE_STOP, HookPhase.AFTER_STOP,
        HookPhase.CHECK, HookPhase.BEFORE_START, HookPhase.AFTER_START,
    ),
    "down": (HookPhase.BEFORE_STOP, HookPhase.AFTER_STOP),
}


class ExecutionPlanner:

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def plan(
            self,
            action: str,
            names: "list[str] | None" = None,
            build: bool = True,
            pull: bool = False,
    ) -> "ExecutionPlan":
        if action not in ("up", "restart", "down"):
            raise ContainerError(f"Unsupported plan action: {action!r}; expected up/restart/down")

        manager = self.manager
        selection = manager.compose_operations.select(names)

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
            options = manager.compose_operations.build_options(action, selection, build, pull)
            if build:
                commands.append(self._planned_command(
                    "build", [*file_args, *manager.compose_runner.build_args(options)]))
            commands.append(self._planned_command(
                "up", [*file_args, *manager.compose_runner.up_args(options)]))
        elif action == "down":
            commands.append(self._planned_command("down", [*file_args, "down", *services]))

        hooks = []
        for phase in _ACTION_PHASES[action]:
            for container in selection.target_containers:
                # A hook order that couldn't actually execute (missing
                # required before/after reference, or a cycle) must fail
                # the plan, not be silently shown as if it would run.
                container.hooks.validate(phase)
                for hook in container.hooks.iter_phase(phase):
                    hooks.append(PlannedHook(phase=phase.value, container=container.name,
                                             name=hook.name, opaque=hook.opaque))
            manager.hooks.validate(phase)
            for hook in manager.hooks.iter_phase(phase):
                hooks.append(PlannedHook(phase=phase.value, container=None, name=hook.name, opaque=hook.opaque))

        warnings = []
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
