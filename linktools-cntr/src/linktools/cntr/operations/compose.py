#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single implementation behind both the root lifecycle shortcuts
(``ct-cntr up/restart/down``) and the ``ct-cntr compose ...`` namespace.

The CLI layer only defines arguments/help/routing; this module owns target
selection, hook dispatch and state updates so the two entry points can never
drift from each other (Spec Part IX, "single implementation principle").
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError
from ..context import EventContext
from ..execution.report import get_records, record_phase, render_report
from ..runtime.compose import ComposeOptions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from ..container import BaseContainer
    from ..manager import ContainerManager


@dataclass(frozen=True)
class ComposeSelection:
    """Resolved target selection for a single compose operation.

    ``project_containers`` is the full installed project (used to build the
    complete ``--file`` set); ``target_containers``/``services`` are the
    user's explicit selection (used for the trailing SERVICE filter and hook
    dispatch). ``full`` is True when the user selected nothing, i.e. the
    whole project is the target.
    """

    project_containers: "tuple[BaseContainer, ...]"
    target_containers: "tuple[BaseContainer, ...]"
    services: "tuple[str, ...]"
    full: bool


class ComposeOperations:
    """Compose lifecycle/inspection operations shared by root and ``compose`` commands."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def select(self, names: "Sequence[str] | None" = None, with_dependencies: bool = False) -> ComposeSelection:
        manager = self.manager
        project_containers = tuple(manager.prepare_installed_containers())

        if not names:
            return ComposeSelection(
                project_containers=project_containers,
                target_containers=project_containers,
                services=(),
                full=True,
            )

        installed_names = {c.name for c in project_containers}
        unknown = [name for name in names if name not in installed_names]
        if unknown:
            raise ContainerError(f"Container(s) not installed: {', '.join(unknown)}")

        target_containers = tuple(c for c in project_containers if c.name in names)
        if with_dependencies:
            target_containers = tuple(manager.resolver.resolve_dependencies(target_containers))

        services: "list[str]" = []
        seen: "set[str]" = set()
        for container in target_containers:
            for service_name in container.services.keys():
                if service_name not in seen:
                    seen.add(service_name)
                    services.append(service_name)
        if not services:
            names_desc = ", ".join(c.name for c in target_containers)
            raise ContainerError(f"No service found in container(s) `{names_desc}`")

        return ComposeSelection(
            project_containers=project_containers,
            target_containers=target_containers,
            services=tuple(services),
            full=False,
        )

    def _make_context(self, commands, selection: ComposeSelection) -> "EventContext":
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = list(selection.project_containers)
        context.target_containers = list(selection.target_containers)
        context.is_full_containers = selection.full
        return context

    def ensure_runtime_requirements(self, selection: ComposeSelection, action: str) -> None:
        """Block ``action`` when a repository's manifest declares a
        docker-engine/docker-compose requirement the actual runtime doesn't
        satisfy (or that can't even be verified -- an unqueryable runtime
        fails closed here, it never silently proceeds). Deduplicated by
        repository, and every repository's issues are collected before
        raising once, so a multi-repo project never fails partway through
        one repo only to leave the rest unreported.

        Only called from up/restart/render (compose)/lock -- down/status/
        doctor/remove keep their own warning-only reporting elsewhere, so a
        version bump can never block stopping or inspecting what's already
        running."""
        manager = self.manager
        seen_urls = set()
        problems: "list[str]" = []
        for container in selection.target_containers:
            repository = getattr(container, "_repository", None)
            if repository is None or repository.manifest is None:
                continue
            url = repository.url
            if url in seen_urls:
                continue
            seen_urls.add(url)
            for issue in manager.repo_manifest.check_runtime_requirements(repository.manifest):
                problems.append(f"Repository `{url}` requires {issue.key}{issue.required}: {issue.message}")
        if problems:
            raise ContainerError(
                f"Cannot {action}: runtime requirement(s) not satisfied.\n" + "\n".join(problems)
            )

    def build_options(
            self, action: str, selection: ComposeSelection, build: bool, pull: bool,
    ) -> ComposeOptions:
        """The exact ``ComposeOptions`` ``up``/``restart`` build for this
        action -- shared with ``ExecutionPlanner`` so a plan can never drift
        from what actually runs. ``down`` never builds a ComposeOptions at
        all (it doesn't build/pull/up anything)."""
        if action == "restart":
            # restart omits the --pull=false / --pull missing defaults that
            # `up` emits, and (unlike `up`/`exec up`/`exec restart`) never
            # includes proxy --build-args.
            return ComposeOptions(
                build=build, pull=pull, remove_orphans=selection.full,
                services=list(selection.services), emit_default_pull=False, include_proxy_build_args=False,
            )
        return ComposeOptions(
            build=build, pull=pull, remove_orphans=selection.full,
            services=list(selection.services), emit_default_pull=True,
        )

    def up(self, names: "Sequence[str] | None" = None, build: bool = True, pull: bool = False,
          report: bool = False) -> None:
        manager = self.manager
        selection = self.select(names)
        self.ensure_runtime_requirements(selection, "up")
        context = self._make_context(["up", pull and "pull", build and "build"], selection)
        options = self.build_options("up", selection, build, pull)

        container_scope = None if context.is_full_containers else ",".join(
            c.name for c in context.target_containers)

        with manager.lifecycle.notify_start(context):
            if build:
                with record_phase(context, "build", command=tuple(manager.compose_runner.build_args(options)),
                                  container=container_scope, logger=manager.logger):
                    manager.compose_runner.build(context, options)
            with record_phase(context, "up", command=tuple(manager.compose_runner.up_args(options)),
                              container=container_scope, logger=manager.logger):
                manager.compose_runner.up(context, options)

        with manager.lifecycle.notify_remove(context):
            pass

        # Record running state only after a successful up.
        manager.running_state.mark_started(context)
        if report:
            render_report(manager.logger, get_records(context))

    def restart(self, names: "Sequence[str] | None" = None, build: bool = True, pull: bool = False,
               report: bool = False) -> None:
        manager = self.manager
        selection = self.select(names)
        self.ensure_runtime_requirements(selection, "restart")
        context = self._make_context(["restart", pull and "pull", build and "build"], selection)
        options = self.build_options("restart", selection, build, pull)

        container_scope = None if context.is_full_containers else ",".join(
            c.name for c in context.target_containers)

        with manager.lifecycle.notify_stop(context):
            with record_phase(context, "stop", command=("stop", *selection.services),
                              container=container_scope, logger=manager.logger):
                manager.compose_runner.stop(context, selection.services)

        with manager.lifecycle.notify_start(context):
            if build:
                with record_phase(context, "build", command=tuple(manager.compose_runner.build_args(options)),
                                  container=container_scope, logger=manager.logger):
                    manager.compose_runner.build(context, options)
            with record_phase(context, "up", command=tuple(manager.compose_runner.up_args(options)),
                              container=container_scope, logger=manager.logger):
                manager.compose_runner.up(context, options)

        with manager.lifecycle.notify_remove(context):
            pass

        # restart ends with the targets running.
        manager.running_state.mark_started(context)
        if report:
            render_report(manager.logger, get_records(context))

    def down(self, names: "Sequence[str] | None" = None, report: bool = False) -> None:
        manager = self.manager
        selection = self.select(names)
        context = self._make_context("down", selection)
        container_scope = None if context.is_full_containers else ",".join(
            c.name for c in context.target_containers)

        with manager.lifecycle.notify_stop(context):
            with record_phase(context, "down", command=("down", *selection.services),
                              container=container_scope, logger=manager.logger):
                manager.compose_runner.down(context, selection.services)

        with manager.lifecycle.notify_remove(context):
            pass

        # Record stopped state only after a successful down.
        manager.running_state.mark_stopped(context)
        if report:
            render_report(manager.logger, get_records(context))

    def render(
            self,
            names: "Sequence[str] | None" = None,
            with_dependencies: bool = False,
            output_format: "str | None" = None,
            check: bool = False,
    ) -> "int | None":
        """``ct-cntr compose``: the final resolved Docker Compose model for
        the installed project (or ``--check`` to only validate it)."""
        selection = self.select(names, with_dependencies=with_dependencies)
        self.ensure_runtime_requirements(selection, "compose")
        context = self._make_context("compose", selection)
        return self.manager.compose_runner.config(
            context, selection.services, output_format=output_format, quiet=check,
        )

    def status(self, sudo_prompt: bool = False):
        """Full-project actual status (Spec section 18): always queries every
        installed container -- the CONTAINER filter for ``ct-cntr status`` is
        a display-only narrowing, applied by the caller."""
        project_containers = tuple(self.manager.prepare_installed_containers())
        state = self.manager.docker_inspector.get_project_state(project_containers, allow_sudo_prompt=sudo_prompt)
        return project_containers, state
