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
from .runtime.compose import service_dependencies

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

    def start_selection(self, selection: ComposeSelection,
                        model: "dict | None" = None) -> ComposeSelection:
        """Resolve runtime providers without starting unrelated sibling services."""
        installed = {container.name: container for container in selection.project_containers}
        owners = {service: container for container in selection.project_containers for service in container.services}
        definitions = model["services"] if model is not None else {
            name: container.services[name] for name, container in owners.items()}
        required = set(selection.target_containers)
        services = set(selection.services) if selection.services else {
            service for container in required for service in container.services}

        while True:
            before = (set(required), set(services))
            for container in tuple(required):
                for dependency in container.dependencies:
                    if dependency not in installed:
                        raise ContainerError(
                            f"Required dependency {dependency!r} for {container.name} is not installed")
                    owner = installed[dependency]
                    required.add(owner)
                    services.update(owner.services)
                declarations = self.manager.integration_snapshot[container.name].get("nginx", {})
                nginx = installed.get("nginx")
                if nginx is not None and any(str(site.server_name) for site in declarations.values()):
                    required.add(nginx)
                    services.update(nginx.services)
            for name in tuple(services):
                owner = owners[name]
                for dependency in service_dependencies(definitions[name]):
                    provider = owners.get(dependency)
                    if provider is None:
                        raise ContainerError(f"Compose dependency {dependency!r} for {owner.name} is not installed")
                    required.add(provider)
                    services.add(dependency)
            if installed.get("nginx") in required:
                from .integration import NginxSite
                for producer, local_id, site in self.manager.iter_integrations("nginx"):
                    if not isinstance(site, NginxSite):
                        raise ContainerError(f"Invalid nginx integration {producer.name}/{local_id}")
                    if not str(site.server_name):
                        continue
                    for capability, provider in (("auth", "authelia"), ("waf", "safeline")):
                        configured = getattr(site, capability)
                        if configured is None:
                            configured = self.manager.env_config.get("NGINX_" + capability.upper() + "_ENABLE", type=bool)
                        if configured:
                            if provider not in installed:
                                raise ContainerError(f"Nginx site {producer.name}/{local_id} requires {provider}")
                            required.add(installed[provider])
                            services.update(("authelia",) if provider == "authelia" else installed[provider].services)
            if before == (required, services):
                break
        ordered = tuple(self.manager.resolver.resolve_dependencies(required))
        ordered_services = []
        visiting = set()

        def visit(name):
            if name in ordered_services:
                return
            if name in visiting:
                raise ContainerError("Compose dependency cycle at " + name)
            visiting.add(name)
            for dependency in owners[name].dependencies:
                for service in installed[dependency].services:
                    visit(service)
            for dependency in service_dependencies(definitions[name]):
                visit(dependency)
            visiting.remove(name)
            ordered_services.append(name)

        for container in ordered:
            for name in container.services:
                if name in services:
                    visit(name)
        if not ordered_services:
            names = ", ".join(c.name for c in selection.target_containers)
            raise ContainerError(f"No runnable service for {names}")
        return ComposeSelection(selection.project_containers, ordered, tuple(ordered_services), selection.full)

    def _reconcile_selection(self, explicit, context, changed_generations=()) -> ComposeSelection:
        services = set(explicit.services) if not explicit.full else {
            name for container in explicit.target_containers for name in container.services}
        targets = set(explicit.target_containers)
        for container in explicit.project_containers:
            changed = container.name in changed_generations
            pending = {name for name in container.services if name in context.initial_running_services and
                       (changed or name in context.changed_compose_services)}
            if pending:
                services.update(pending)
                targets.add(container)
        return self.start_selection(ComposeSelection(explicit.project_containers, tuple(targets),
                                                     tuple(services), explicit.full),
                                    getattr(context, "compose_model", None))

    def _make_context(self, commands, selection: ComposeSelection) -> "EventContext":
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = list(selection.project_containers)
        context.target_containers = list(selection.target_containers)
        context.is_full_containers = selection.full
        return context

    def sync_selection(self, selection: ComposeSelection) -> "tuple[BaseContainer, ...]":
        """Reconcile the full installed configuration without widening startup."""
        return selection.project_containers

    def up(self, names: "Sequence[str] | None" = None, pull: bool = False,
           report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=False)

    def restart(self, names: "Sequence[str] | None" = None, pull: bool = False,
                report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=True)

    def _start(self, names, pull: bool, report: bool, restart: bool) -> None:
        from .artifacts import AppliedServiceModels, GeneratedCandidate, collect_candidates
        manager = self.manager
        explicit = self.select(names, for_start=True)
        selection = self.start_selection(explicit)
        sync = self.sync_selection(selection)
        context = self._make_context(["restart" if restart else "up", pull and "pull"], selection)
        context.config_containers = list(sync)
        runner = manager.compose_runner
        import os
        context.saved_compose = {}
        context.applied_compose = {}
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
        actual = manager.docker_inspector.get_project_state(selection.project_containers)
        running = set(actual.running_container_names)
        context.initial_running = frozenset(running)
        context.initial_running_services = frozenset(
            service.service for service in actual.services if service.state in ("running", "restarting"))
        context.initial_services = frozenset(service.service for service in getattr(actual, "services", ()))
        if not any(service.service == "nginx" and service.state == "running" for service in actual.services):
            running.discard("nginx")
        # Image preparation may include running aggregate owners that later
        # prove unchanged; only the final candidate closure is applied.
        generations = manager.generated_configs
        context.changed_compose_services = set(context.initial_running_services)
        selection = self._reconcile_selection(explicit, context, generations)
        context.target_containers = list(selection.target_containers)

        # Hooks may prepare env_file inputs; capture the authoritative resolved
        # candidate only after startup preparation, and before any target stops.
        with manager.lifecycle.notify_start(context):
            model = runner.final_model(context)
            context.compose_model = model
            context.service_models = AppliedServiceModels(manager, model)
            context.changed_compose_services = set(context.service_models.changed_services)
            selection = self._reconcile_selection(explicit, context, generations)
            required_services = set(selection.services)
            image_services = tuple(name for c in sync for name in c.services
                                   if name in required_services or c.name in generations or
                                   (name in context.initial_running_services and
                                    name in context.changed_compose_services))
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
                owner = generations.get(container.name)
                if owner is not None:
                    with record_phase(context, "prepare-config", container=container.name, logger=manager.logger):
                        owner.prepare(context)
                        candidates[container.name] = GeneratedCandidate(container, owner.render)
            context.generated_candidates = candidates
            for container in sync:
                candidate = candidates.get(container.name)
                if candidate is not None:
                    with record_phase(context, "validate-config", container=container.name, logger=manager.logger):
                        generations[container.name].validate(candidate, context)

            selection = self._reconcile_selection(explicit, context,
                {name for name, candidate in candidates.items() if candidate.changed or
                 (name == "nginx" and getattr(context, "nginx_certificate_replaced", False))})
            required_services = set(selection.services)

            if restart:
                stop_context = self._make_context(context.commands, explicit)
                with manager.lifecycle.notify_stop(stop_context):
                    with record_phase(context, "stop", command=("stop", *explicit.services), logger=manager.logger):
                        runner.stop(stop_context, explicit.services)
                        manager.running_state.mark_stopped(stop_context)
                running.difference_update(c.name for c in explicit.target_containers)

            nginx = next((c for c in sync if c.name == "nginx"), None)
            if nginx is not None and "nginx" in required_services and nginx.name not in running:
                with record_phase(context, "bootstrap", container="nginx", logger=manager.logger):
                    try:
                        generations["nginx"].bootstrap(context)
                    except Exception as error:
                        self._rollback_candidate(nginx, candidates["nginx"], context, ("nginx",), error)
                        raise
                    running.add("nginx")
                    if candidates.get("nginx") and not candidates["nginx"].previous_id:
                        candidates["nginx"].previous_id = context.nginx_bootstrap_id

            owners = {service: container for container in sync for service in container.services}
            # nginx's bootstrap satisfies provider readiness; publish its full
            # generation only after auth/WAF services, then update navigation.
            services = tuple(name for name in selection.services if owners[name].name not in ("nginx", "flare")) + tuple(
                name for group in ("nginx", "flare") for name in selection.services if owners[name].name == group)
            context.applied_generation_services = {}
            for service in services:
                container = owners[service]
                candidate = candidates.get(container.name)
                if candidate is not None:
                    with record_phase(context, "publish-config", container=container.name, logger=manager.logger):
                        self._publish_candidate(container, candidate, context, (service,))
                else:
                    with record_phase(context, "up", container=container.name, logger=manager.logger):
                        self._apply_services_with_rollback(container, context, (service,))
                if service == "safeline-mgt":
                    runner.wait_service_healthy(context, service)
                state_context = self._make_context(context.commands, ComposeSelection(
                    selection.project_containers, (container,), (service,), False))
                manager.running_state.mark_started(state_context)
            for container in sync:
                if container.name in candidates and not any(name in required_services for name in container.services):
                    self._publish_candidate(container, candidates[container.name], context, ())
        with manager.lifecycle.notify_remove(context):
            pass
        if report:
            render_report(manager.logger, get_records(context))

    def _record_applied_compose(self, container, context, services) -> None:
        import os
        import yaml
        from .artifacts import atomic_write_text_if_changed, sha256_of
        if hasattr(context, "service_models"):
            context.service_models.record(services)
        for path, content in getattr(context, "compose_files", {}).items():
            if context.compose_owners[path] != container.name:
                continue
            # A synchronized owner may contain stopped sibling services. Keep
            # their last-applied models until those services are actually used.
            applied_content = getattr(context, "applied_compose", {}).get(path, context.saved_compose.get(path, ""))
            previous = yaml.safe_load(applied_content) or {}
            current = yaml.safe_load(content) or {}
            applied_services = dict(previous.get("services", {}))
            for service in services:
                if service in current.get("services", {}):
                    applied_services[service] = current["services"][service]
            current["services"] = applied_services
            content = yaml.safe_dump(current, sort_keys=False)
            applied = os.path.join(str(self.manager.data_path), "compose", "applied", container.name + ".yml")
            os.makedirs(os.path.dirname(applied), exist_ok=True)
            atomic_write_text_if_changed(applied, content)
            if hasattr(context, "applied_compose"):
                context.applied_compose[path] = content
            self.manager.artifact_index.record({os.path.relpath(applied, str(self.manager.data_path)): {
                "kind": "compose-applied", "container": container.name, "sha256": sha256_of(content)}})

    def _restore_applied_compose(self, container, context, previous) -> None:
        import os
        from .artifacts import atomic_write_text_if_changed, sha256_of
        applied = getattr(context, "applied_compose", {})
        for path, content in previous.items():
            if path not in applied:
                continue
            destination = os.path.join(str(self.manager.data_path), "compose", "applied", container.name + ".yml")
            atomic_write_text_if_changed(destination, content)
            applied[path] = content
            self.manager.artifact_index.record({os.path.relpath(destination, str(self.manager.data_path)): {
                "kind": "compose-applied", "container": container.name, "sha256": sha256_of(content)}})

    def _apply_services_with_rollback(self, container, context, services) -> None:
        from .artifacts import atomic_write_text_if_changed
        runner = self.manager.compose_runner
        try:
            runner.apply_services(context, services)
            self._record_applied_compose(container, context, services)
        except Exception as error:
            previous = {path: content for path, content in context.saved_compose.items()
                        if context.compose_owners[path] == container.name}
            running = tuple(service for service in services if service in context.initial_running_services)
            saved_models = getattr(context, "service_models", None)
            if running and (previous or (saved_models and any(name in saved_models.previous for name in running))):
                try:
                    files = dict(context.compose_files)
                    files.update(previous)
                    for path, content in previous.items():
                        atomic_write_text_if_changed(path, content)
                    for service in running:
                        model = context.service_models.previous.get(service) if hasattr(context, "service_models") else None
                        runner.apply_saved_services(context, (service,), {"previous.yml": model} if model else files)
                    if hasattr(context, "service_models"):
                        context.service_models.restore(running)
                    self._restore_applied_compose(container, context, previous)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; Compose rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
            raise

    def _publish_candidate(self, container, candidate, context, services) -> None:
        context.generated_candidates[container.name] = candidate
        candidate.publish()
        if not services:
            return
        try:
            self.manager.generated_configs[container.name].apply(candidate, context, services)
            self._record_applied_compose(container, context, services)
            applied = getattr(context, "applied_generation_services", {})
            applied.setdefault(container.name, []).extend(services)
        except Exception as error:
            self._rollback_candidate(container, candidate, context, services, error)
            raise

    def _rollback_candidate(self, container, candidate, context, services, error) -> None:
        from copy import copy
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
                if hasattr(context, "service_models"):
                    context.rollback_service_models = context.service_models.previous
                if old_compose:
                    context.rollback_compose_files = dict(context.compose_files)
                    context.rollback_compose_files.update(old_compose)
                affected = tuple(getattr(context, "applied_generation_services", {}).get(container.name, ())) + tuple(services)
                restore_services = tuple(dict.fromkeys(service for service in affected
                                                      if service in context.initial_running_services))
                if restore_services:
                    self.manager.generated_configs[container.name].apply(previous, context, restore_services)
                    if hasattr(context, "service_models"):
                        context.service_models.restore(restore_services)
                    self._restore_applied_compose(container, context, old_compose)
                    restored_context = copy(context)
                    restored_context.target_containers = [container]
                    restored_context.is_full_containers = False
                    self.manager.running_state.mark_started(restored_context)
            except Exception as rollback_error:
                raise ContainerError("{} apply failed: {}; rollback failed: {}".format(
                    container.name, error, rollback_error)) from error
            finally:
                if hasattr(context, "rollback_compose_files"):
                    del context.rollback_compose_files
                if hasattr(context, "rollback_service_models"):
                    del context.rollback_service_models
                from .artifacts import atomic_write_text_if_changed
                for path, content in old_compose.items():
                    atomic_write_text_if_changed(path, content)

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
