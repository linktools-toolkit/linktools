#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single implementation behind both the root lifecycle shortcuts
(``ct-cntr up/restart/down``) and the ``ct-cntr compose`` final-model
rendering command.

The CLI layer only defines arguments/help/routing; this module owns target
selection, hook dispatch and state updates so the two entry points can never
drift from each other.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .container import ContainerError
from .context import EventContext
from .execution.model import get_records, record_phase, render_report

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .container import BaseContainer
    from .manager import ContainerManager
    from .runtime.inspect import ProjectRuntimeState


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
    """Compose lifecycle operations and final-model rendering, shared by
    the root ``up``/``restart``/``down`` commands and the ``compose``
    command."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def select(self, names: "Sequence[str] | None" = None, with_dependencies: bool = False,
              metadata_only: bool = False, for_start: bool = False) -> ComposeSelection:
        """Resolve the target selection. ``metadata_only=True`` (used by
        ExecutionPlanner, which must stay read-only) registers config
        fields without running any container's ``on_prepare()`` -- real
        execution (``up``/``restart``/``down``/``compose``) always needs
        the full prepare instead."""
        manager = self.manager
        if metadata_only:
            project_containers = tuple(manager.load_installed_config_metadata())
            if not project_containers:
                from .container import NoContainerInstalledError
                raise NoContainerInstalledError("No container installed")
        else:
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
        if not services and not for_start:
            names_desc = ", ".join(c.name for c in target_containers)
            raise ContainerError(f"No service found in container(s) `{names_desc}`")

        return ComposeSelection(
            project_containers=project_containers,
            target_containers=target_containers,
            services=tuple(services),
            full=False,
        )

    def start_selection(self, selection: ComposeSelection) -> ComposeSelection:
        """Expand a partial start to installed dependency and integration consumers."""
        installed = {container.name: container for container in selection.project_containers}
        owners = {
            service: container
            for container in selection.project_containers
            for service in container.services
        }
        required = set(selection.target_containers)

        while True:
            before = set(required)
            for container in tuple(required):
                for dependency in container.dependencies:
                    if dependency not in installed:
                        raise ContainerError(
                            f"Required dependency {dependency!r} for {container.name} is not installed")
                    required.add(installed[dependency])

                for service in container.services.values():
                    depends_on = service.get("depends_on") or ()
                    for dependency in depends_on:
                        owner = owners.get(dependency)
                        if owner is None:
                            raise ContainerError(
                                f"Compose dependency {dependency!r} for {container.name} is not installed")
                        required.add(owner)

                for consumer_name, declarations in self.manager.integration_snapshot[container.name].items():
                    consumer = installed.get(consumer_name)
                    if consumer is None:
                        continue
                    if consumer_name == "nginx":
                        if not any(str(site.server_name) for site in declarations.values()):
                            continue
                    required.add(consumer)

            nginx = installed.get("nginx")
            if nginx in required:
                from ._nginx import NginxSite
                for producer, local_id, site in self.manager.iter_integrations("nginx"):
                    if not isinstance(site, NginxSite):
                        raise ContainerError(
                            f"Invalid nginx integration {producer.name}/{local_id}")
                    if not str(site.server_name):
                        continue
                    for capability, provider in (("auth", "authelia"), ("waf", "safeline")):
                        configured = getattr(site, capability)
                        if configured is None:
                            configured = self.manager.env_config.get(
                                "NGINX_" + capability.upper() + "_ENABLE", type=bool)
                        if configured:
                            if provider not in installed:
                                raise ContainerError(
                                    f"Nginx site {producer.name}/{local_id} requires {provider}")
                            required.add(installed[provider])
            if required == before:
                break

        ordered = tuple(self.manager.resolver.resolve_dependencies(required))
        services = tuple(
            name for container in ordered for name in container.services
        )
        if not services:
            names = ", ".join(c.name for c in selection.target_containers)
            raise ContainerError(f"No runnable service for {names}")
        return ComposeSelection(selection.project_containers, ordered, services, selection.full)

    def _make_context(self, commands, selection: ComposeSelection) -> "EventContext":
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = list(selection.project_containers)
        context.target_containers = list(selection.target_containers)
        context.is_full_containers = selection.full
        return context

    def sync_selection(self, selection: ComposeSelection) -> "tuple[BaseContainer, ...]":
        """Configuration edges expand synchronization, never the running set."""
        names = {container.name for container in selection.target_containers}
        sources = self.manager.config_source_snapshot
        while True:
            expanded = names | {name for name, values in sources.items() if names.intersection(values)}
            if expanded == names:
                break
            names = expanded
        return tuple(self.manager.resolver.resolve_dependencies(
            c for c in selection.project_containers if c.name in names))

    def up(self, names: "Sequence[str] | None" = None, pull: bool = False,
           report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=False)

    def restart(self, names: "Sequence[str] | None" = None, pull: bool = False,
                report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=True)

    def _start(self, names, pull: bool, report: bool, restart: bool) -> None:
        from .artifacts import GeneratedCandidate, collect_candidates
        manager = self.manager
        explicit = self.select(names, for_start=True)
        selection = self.start_selection(explicit)
        sync = self.sync_selection(selection)
        context = self._make_context(["restart" if restart else "up", pull and "pull"], selection)
        context.config_containers = list(sync)
        runner = manager.compose_runner
        import os
        context.saved_compose = {}
        context.compose_files = {}
        context.compose_owners = {}
        context.changed_compose_services = set()
        for path, (kind, owner, content) in collect_candidates(manager, selection.project_containers).items():
            if kind != "compose":
                continue
            context.compose_files[path] = content
            context.compose_owners[path] = owner
            try:
                applied = os.path.join(str(manager.data_path), "compose", "applied", owner + ".yml")
                with open(applied if os.path.exists(applied) else path, encoding="utf-8") as stream:
                    previous = stream.read()
                    context.saved_compose[path] = previous
            except FileNotFoundError:
                previous = None
            if previous != content:
                import yaml
                old_services = (yaml.safe_load(previous) or {}).get("services", {}) if previous else {}
                new_services = (yaml.safe_load(content) or {}).get("services", {})
                context.changed_compose_services.update(name for name, value in new_services.items()
                                                        if old_services.get(name) != value)
        actual = manager.docker_inspector.get_project_state(selection.project_containers)
        running = set(actual.running_container_names)
        context.initial_running = frozenset(running)
        context.initial_services = frozenset(service.service for service in getattr(actual, "services", ()))
        if not any(service.service == "nginx" and service.state == "running" for service in actual.services):
            running.discard("nginx")
        # A running synchronized nginx is also an applying consumer. Its
        # optional providers must join the same closure before image planning.
        nginx = next((c for c in sync if c.name == "nginx"), None)
        if nginx is not None and "nginx" in running and nginx not in selection.target_containers:
            selection = self.start_selection(ComposeSelection(selection.project_containers,
                selection.target_containers + (nginx,), selection.services, selection.full))
            sync = self.sync_selection(selection)
            context.target_containers = list(selection.target_containers)
            context.config_containers = list(sync)
        required = {c.name for c in selection.target_containers}

        # Startup callbacks and all validation precede any explicit restart stop.
        with manager.lifecycle.notify_start(context):
            model = runner.final_model(context)
            image_services = tuple(name for c in sync for name in c.services)
            image_plan = manager.image_preparer.plan(model, image_services, force_pull=pull)
            context.changed_image_services = set(image_plan.pull) | set(image_plan.build)
            if image_plan.pull:
                with record_phase(context, "pull", command=tuple(runner.pull_args(image_plan.pull)), logger=manager.logger):
                    runner.pull(context, image_plan.pull)
            if image_plan.build:
                with record_phase(context, "build", command=tuple(runner.build_args(
                        runner.options_for_build(image_plan.build, pull=pull))), logger=manager.logger):
                    runner.build(context, runner.options_for_build(image_plan.build, pull=pull))
            candidates = {}
            for container in sync:
                if container.generated_config_path is not None:
                    with record_phase(context, "prepare-config", container=container.name, logger=manager.logger):
                        container.prepare_generated_config(context)
                        candidates[container.name] = GeneratedCandidate(container)
            context.generated_candidates = candidates
            for container in sync:
                candidate = candidates.get(container.name)
                if candidate is not None:
                    with record_phase(context, "validate-config", container=container.name, logger=manager.logger):
                        container.validate_generated_config(candidate, context)

            if restart:
                stop_context = self._make_context(context.commands, explicit)
                with manager.lifecycle.notify_stop(stop_context):
                    with record_phase(context, "stop", command=("stop", *explicit.services), logger=manager.logger):
                        runner.stop(stop_context, explicit.services)
                        manager.running_state.mark_stopped(stop_context)
                running.difference_update(c.name for c in explicit.target_containers)

            nginx = next((c for c in sync if c.name == "nginx"), None)
            if nginx is not None and nginx.name in required and nginx.name not in running:
                with record_phase(context, "bootstrap", container="nginx", logger=manager.logger):
                    nginx.bootstrap_generated_config(context)
                    running.add("nginx")
                    if candidates.get("nginx") and not candidates["nginx"].previous_id:
                        candidates["nginx"].previous_id = context.nginx_bootstrap_id

            # The resolver owns strong dependency order. nginx full config is
            # deferred until its optional auth/WAF providers are ready.
            deferred = []
            for container in sync:
                if container.name == "nginx":
                    continue
                if (container.name not in required and container.name not in ("authelia", "safeline")) or container.name == "flare":
                    deferred.append(container)
                    continue
                candidate = candidates.get(container.name)
                should_run = container.name in required
                should_apply = should_run or container.name in running
                if candidate is not None:
                    if should_run:
                        prerequisites = [name for name in container.services
                                         if name != container.name and name != "authelia-admin"]
                        if prerequisites:
                            runner.apply_services(context, prerequisites)
                    with record_phase(context, "publish-config", container=container.name, logger=manager.logger):
                        self._publish_candidate(container, candidate, context, should_apply)
                elif should_run or container.name in running:
                    with record_phase(context, "up", container=container.name, logger=manager.logger):
                        self._apply_services_with_rollback(container, context, container.name in running)
                if should_run:
                    state_context = self._make_context(context.commands, ComposeSelection(
                        selection.project_containers, (container,), tuple(container.services), False))
                    manager.running_state.mark_started(state_context)
                if container.name in ("authelia", "safeline") and should_apply:
                    service = "authelia" if container.name == "authelia" else "safeline-mgt"
                    if service in container.services:
                        runner.wait_service_healthy(context, service)

            if nginx is not None:
                candidate = candidates.get("nginx")
                if candidate is not None:
                    with record_phase(context, "publish-config", container="nginx", logger=manager.logger):
                        self._publish_candidate(nginx, candidate, context, "nginx" in running)
                if nginx.name in required:
                    state_context = self._make_context(context.commands, ComposeSelection(
                        selection.project_containers, (nginx,), tuple(nginx.services), False))
                    manager.running_state.mark_started(state_context)
            for container in deferred:
                candidate = candidates.get(container.name)
                should_apply = container.name in required or container.name in running
                if candidate is not None:
                    with record_phase(context, "publish-config", container=container.name, logger=manager.logger):
                        self._publish_candidate(container, candidate, context, should_apply)
                elif should_apply:
                    self._apply_services_with_rollback(container, context, container.name in running)
                if container.name in required:
                    state_context = self._make_context(context.commands, ComposeSelection(
                        selection.project_containers, (container,), tuple(container.services), False))
                    manager.running_state.mark_started(state_context)
        with manager.lifecycle.notify_remove(context):
            pass
        if report:
            render_report(manager.logger, get_records(context))

    def _record_applied_compose(self, container, context) -> None:
        import os
        from .artifacts import atomic_write_text_if_changed, sha256_of
        for path, content in getattr(context, "compose_files", {}).items():
            if context.compose_owners[path] != container.name:
                continue
            applied = os.path.join(str(self.manager.data_path), "compose", "applied", container.name + ".yml")
            os.makedirs(os.path.dirname(applied), exist_ok=True)
            atomic_write_text_if_changed(applied, content)
            self.manager.artifact_index.record({os.path.relpath(applied, str(self.manager.data_path)): {
                "kind": "compose-applied", "container": container.name, "sha256": sha256_of(content)}})

    def _apply_services_with_rollback(self, container, context, was_running: bool) -> None:
        from .artifacts import atomic_write_text_if_changed
        runner = self.manager.compose_runner
        try:
            runner.apply_services(context, tuple(container.services))
            self._record_applied_compose(container, context)
        except Exception as error:
            previous = {path: content for path, content in context.saved_compose.items()
                        if context.compose_owners[path] == container.name}
            if previous and was_running:
                try:
                    files = dict(context.compose_files)
                    files.update(previous)
                    for path, content in previous.items():
                        atomic_write_text_if_changed(path, content)
                    runner.apply_saved_services(context, tuple(container.services), files)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; Compose rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
            raise

    def _publish_candidate(self, container, candidate, context, apply: bool) -> None:
        from copy import copy
        context.generated_candidates[container.name] = candidate
        candidate.publish()
        if not apply:
            return
        try:
            container.apply_generated_config(candidate, context)
            self._record_applied_compose(container, context)
        except Exception as error:
            try:
                candidate.restore()
            except Exception as rollback_error:
                raise ContainerError("{} apply failed: {}; rollback publication failed: {}".format(
                    container.name, error, rollback_error)) from error
            if candidate.previous_id:
                previous = copy(candidate)
                previous.generation_id = candidate.previous_id
                previous.path = __import__("os").path.join(candidate.root, candidate.previous_id)
                previous.changed = True
                old_compose = {path: content for path, content in getattr(context, "saved_compose", {}).items()
                               if context.compose_owners[path] == container.name}
                try:
                    context.generated_candidates[container.name] = previous
                    if old_compose:
                        context.rollback_compose_files = dict(context.compose_files)
                        context.rollback_compose_files.update(old_compose)
                    container.apply_generated_config(previous, context)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
                finally:
                    if hasattr(context, "rollback_compose_files"):
                        del context.rollback_compose_files
                    from .artifacts import atomic_write_text_if_changed
                    for path, content in old_compose.items():
                        atomic_write_text_if_changed(path, content)
            raise

    def down(self, names: "Sequence[str] | None" = None, report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._down(names, report)

    def _down(self, names: "Sequence[str] | None", report: bool) -> None:
        manager = self.manager
        selection = self.select(names)
        context = self._make_context("down", selection)
        container_scope = None if context.is_full_containers else ",".join(
            c.name for c in context.target_containers)

        with manager.lifecycle.notify_stop(context):
            with record_phase(context, "down", command=("down", *selection.services),
                              container=container_scope, logger=manager.logger):
                manager.compose_runner.down(context, selection.services)
            # See up()'s identical comment -- recorded before
            # on_stopped/AFTER_STOP hooks run.
            manager.running_state.mark_stopped(context)

        with manager.lifecycle.notify_remove(context):
            pass

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
        context = self._make_context("compose", selection)
        return self.manager.compose_runner.config(
            context, selection.services, output_format=output_format, quiet=check,
        )

    def status(self) -> "tuple[tuple[BaseContainer, ...], ProjectRuntimeState]":
        """Full-project actual status: always queries every
        installed container -- the CONTAINER filter for ``ct-cntr status`` is
        a display-only narrowing, applied by the caller."""
        project_containers = tuple(self.manager.prepare_installed_containers())
        state = self.manager.docker_inspector.get_project_state(project_containers)
        return project_containers, state
