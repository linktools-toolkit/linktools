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
from .context import OperationContext
from .execution.model import get_records, record_phase, render_report
from .runtime.compose import order_services, service_dependencies

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
    native_roots: "frozenset[str]" = frozenset()


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
                        model: "dict | None" = None,
                        dependency_roots: "Sequence[BaseContainer] | None" = None) -> ComposeSelection:
        """Resolve requested container dependencies and all selected service dependencies."""
        installed = {container.name: container for container in selection.project_containers}
        owners = {service: container for container in selection.project_containers for service in container.services}
        definitions = model["services"] if model is not None else {
            name: container.services[name] for name, container in owners.items()}
        required = set(selection.target_containers)
        roots = set(selection.target_containers if dependency_roots is None else dependency_roots)
        services = set(selection.services) if selection.services else {
            service for container in required for service in container.services}

        while True:
            before = (set(required), set(services), set(roots))
            for container in tuple(roots):
                for dependency in container.dependencies:
                    if dependency not in installed:
                        raise ContainerError(
                            f"Required dependency {dependency!r} for {container.name} is not installed")
                    owner = installed[dependency]
                    required.add(owner)
                    roots.add(owner)
                    services.update(owner.services)
            required_names = {container.name for container in roots}
            for container in selection.project_containers:
                for provider, provider_services in container.get_runtime_requirements(required_names).items():
                    owner = installed[provider]
                    required.add(owner)
                    roots.add(owner)
                    services.update(provider_services)
            for name in tuple(services):
                owner = owners[name]
                for dependency, options in service_dependencies(definitions[name]).items():
                    if options.get("required", True) is False and dependency not in services:
                        continue
                    provider = owners.get(dependency)
                    if provider is None:
                        raise ContainerError(f"Compose dependency {dependency!r} for {owner.name} is not installed")
                    required.add(provider)
                    if owner in roots:
                        roots.add(provider)
                    services.add(dependency)
            if before == (required, services, roots):
                break
        ordered = tuple(container for container in selection.project_containers if container in required)
        selected_services = tuple(name for container in ordered for name in container.services if name in services)
        bootstrap = {name for container in ordered for name in container.bootstrap_services if name in services}
        ordered_services = order_services(
            selection.project_containers, selected_services, model, bootstrap,
            dependency_roots={container.name for container in roots})
        if not ordered_services:
            names = ", ".join(c.name for c in selection.target_containers)
            raise ContainerError(f"No runnable service for {names}")
        # Pre-start callbacks follow the same service dependencies as apply.
        # Keep owners without selected services in their original project order.
        ordered_owners = tuple(dict.fromkeys(owners[service] for service in ordered_services))
        ordered_owners += tuple(container for container in ordered if container not in ordered_owners)
        return ComposeSelection(selection.project_containers, ordered_owners, tuple(ordered_services),
                                selection.full, frozenset(c.name for c in roots))

    def _reconcile_selection(self, explicit, context, changed_generations=()) -> ComposeSelection:
        services = set(explicit.services) if not explicit.full else {
            name for container in explicit.target_containers for name in container.services}
        targets = set(explicit.target_containers)
        native_roots = set(explicit.target_containers)
        for container in explicit.project_containers:
            changed = container.name in changed_generations
            if container not in explicit.target_containers and not changed:
                continue
            pending = {name for name in container.services if name in context.initial_running_services and
                       (name in context.changed_compose_services or
                        (changed and name in container.generation_services))}
            if pending:
                services.update(pending)
                targets.add(container)
                if changed and pending.intersection(container.generation_services):
                    native_roots.add(container)
        return self.start_selection(ComposeSelection(explicit.project_containers, tuple(targets),
                                                     tuple(services), explicit.full),
                                    getattr(context, "compose_model", None),
                                    dependency_roots=tuple(native_roots))

    def _make_context(self, commands, selection: ComposeSelection) -> "OperationContext":
        context = OperationContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = list(selection.project_containers)
        context.target_containers = list(selection.target_containers)
        context.target_services = selection.services or tuple(
            service for container in selection.target_containers for service in container.services)
        context.is_full_containers = selection.full
        return context

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
        sync = selection.project_containers
        context = self._make_context(["restart" if restart else "up", pull and "pull"], selection)
        context.config_containers = list(sync)
        runner = manager.compose_runner
        import os
        context.saved_compose = {}
        context.applied_generation_services = {}
        context.locally_restored_services = set()
        context.started_services = set()
        context.bootstrapped_services = set()
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
                was_applied = os.path.exists(applied)
                with open(applied if was_applied else path, encoding="utf-8") as stream:
                    previous = stream.read()
                    context.saved_compose[path] = previous
            except FileNotFoundError:
                previous = None
        actual = manager.docker_inspector.get_project_state(selection.project_containers)
        context.initial_running = frozenset(actual.running_container_names)
        running_services = {service.service for service in actual.services if service.state == "running"}
        context.initial_running_services = frozenset(
            service.service for service in actual.services if service.state in ("running", "restarting"))
        context.initial_healthy_services = frozenset(
            service.service for service in actual.services if service.state == "running" and service.health == "healthy")
        context.native_running_images = {
            service.service: service.image_id for service in actual.services
            if service.state in ("running", "restarting") and service.image_id}
        context.initial_services = frozenset(service.service for service in actual.services)
        # Running owners prepare hook-dependent inputs before their changes
        # are knowable; application and after-start use the final closure.
        generations = manager.generated_configs
        context.changed_compose_services = set(context.initial_running_services)
        selection = self._reconcile_selection(explicit, context)
        # Running generated consumers may need new native providers after
        # their candidates are rendered. Prepare those providers' hooks before
        # resolving the final Compose model, without adding them to the apply scope.
        preparation_roots = tuple(container for container in sync
            if container in explicit.target_containers or
            (container.name in generations and any(
                service in context.initial_running_services
                for service in container.generation_services)))
        preparation = self.start_selection(selection, dependency_roots=preparation_roots)
        context.target_containers = list(preparation.target_containers)
        context.target_services = preparation.services

        # Hooks may prepare env_file inputs; capture the authoritative resolved
        # candidate only after startup preparation, and before any target stops.
        with manager.lifecycle.notify_start(context):
            model = runner.final_model(context)
            context.compose_model = model
            context.service_models = AppliedServiceModels(manager, model)
            context.changed_compose_services = set(context.service_models.changed_services)
            selection = self._reconcile_selection(explicit, context)
            required_services = set(selection.services)
            generation_targets = {
                container.name for container in sync
                if container.name in generations and any(
                    name in required_services or name in context.initial_running_services
                    for name in container.generation_services)
            }
            context.changed_image_services = set()
            context.image_preparation_targets = set()
            prepared_pulls, prepared_builds = set(), set()

            def prepare_images(services):
                image_plan = manager.image_preparer.plan(model, tuple(services), force_pull=pull)
                context.image_preparation_targets.update(image_plan.targets)
                pull_services = tuple(service for service in image_plan.pull if service not in prepared_pulls)
                build_services = tuple(service for service in image_plan.build if service not in prepared_builds)
                if pull_services:
                    with record_phase(context, "pull", command=tuple(runner.pull_args(pull_services)),
                                      logger=manager.logger):
                        runner.pull(context, pull_services)
                if build_services:
                    options = runner.options_for_build(build_services, pull=pull)
                    with record_phase(context, "build", command=tuple(runner.build_args(options)),
                                      logger=manager.logger):
                        runner.build(context, options)
                prepared_pulls.update(pull_services)
                prepared_builds.update(build_services)
                context.changed_image_services.update(pull_services)
                context.changed_image_services.update(build_services)

            prepare_images(selection.services)
            candidates = {}
            context.generated_candidates = candidates
            while True:
                pending = [
                    container for container in sync
                    if container.name in generations and container.name not in candidates
                    and (container.name in generation_targets or
                         any(name in selection.services for name in container.generation_services))
                ]
                if not pending:
                    break
                additional = set(selection.services) - required_services
                if additional:
                    prepare_images(tuple(name for name in selection.services if name in additional))
                    required_services.update(additional)
                for container in pending:
                    owner = generations[container.name]
                    with record_phase(context, "prepare-config", container=container.name, logger=manager.logger):
                        owner.on_prepare_config(context)
                        candidates[container.name] = GeneratedCandidate(container, owner.render_config)
                changed = {name for name, candidate in candidates.items() if candidate.changed}
                for name, candidate in candidates.items():
                    if name in changed:
                        continue
                    owner = generations[name]
                    if any(service in context.initial_running_services and
                           not owner.is_generation_current(context, service, candidate)
                           for service in owner.generation_services):
                        changed.add(name)
                selection = self._reconcile_selection(explicit, context, changed)
            final_services = set(selection.services)
            additional = final_services - required_services
            if additional:
                prepare_images(tuple(name for name in selection.services if name in additional))
            required_services = final_services
            context.target_containers = list(selection.target_containers)
            context.target_services = selection.services
            for container in sync:
                candidate = candidates.get(container.name)
                if candidate is not None:
                    with record_phase(context, "validate-config", container=container.name, logger=manager.logger):
                        generations[container.name].validate_config(context, candidate)

            for container in sync:
                services = tuple(name for name in container.services if name in required_services)
                self._require_rollback_model(container, context, services)

            bootstrap_candidates = {}
            available_after_stop = set(running_services)
            stopped_services = ({service for container in explicit.target_containers
                                 for service in container.services} if explicit.full
                                else set(explicit.services)) if restart else set()
            available_after_stop.difference_update(stopped_services)
            for container in sync:
                services = set(container.bootstrap_services) & required_services
                if services and not services.issubset(available_after_stop):
                    with record_phase(context, "validate-bootstrap", container=container.name, logger=manager.logger):
                        bootstrap = GeneratedCandidate(container, container.render_bootstrap)
                        container.validate_config(context, bootstrap)
                        bootstrap_candidates[container.name] = bootstrap

            pending_restart = set(stopped_services) & context.initial_running_services
            owners = {service: container for container in sync for service in container.services}
            stopped = False
            stop_attempted = False
            try:
                if restart and (explicit.full or explicit.services):
                    stop_context = self._make_context(context.commands, explicit)
                    with manager.lifecycle.notify_stop(stop_context):
                        with record_phase(context, "stop", command=("stop", *explicit.services), logger=manager.logger):
                            stop_attempted = True
                            runner.stop(stop_context, explicit.services)
                            stopped = True
                            manager.running_state.mark_stopped(stop_context)
                    running_services.difference_update(stopped_services)

                bootstrap_available = set()
                for container in sync:
                    services = tuple(service for service in container.bootstrap_services if service in required_services)
                    if not services:
                        continue
                    if all(service in running_services for service in services):
                        for service in services:
                            runner.wait_service_healthy(context, service)
                        bootstrap_available.update(services)
                        continue
                    with record_phase(context, "bootstrap", container=container.name, logger=manager.logger):
                        final_candidate = candidates[container.name]
                        bootstrap = bootstrap_candidates[container.name]
                        self._publish_candidate(container, bootstrap, context, services, record_applied=False)
                        context.generated_candidates[container.name] = final_candidate
                        running_services.update(services)
                        bootstrap_available.update(services)
                        context.bootstrapped_services.update(services)
                        if (not final_candidate.previous_id and
                                not any(service in context.initial_running_services for service in services)):
                            final_candidate.previous_id = bootstrap.generation_id
                            final_candidate.bootstrap_fallback = True

                services = order_services(
                    sync, selection.services, context.compose_model, bootstrap_available,
                    dependency_roots=selection.native_roots)
                for service in services:
                    context.applying_service = service
                    container = owners[service]
                    candidate = candidates.get(container.name)
                    if candidate is not None:
                        with record_phase(context, "publish-config", container=container.name, logger=manager.logger):
                            self._publish_candidate(container, candidate, context, (service,))
                    else:
                        with record_phase(context, "up", container=container.name, logger=manager.logger):
                            self._apply_services_with_rollback(container, context, (service,))
                    state_context = self._make_context(context.commands, ComposeSelection(
                        selection.project_containers, (container,), (service,), False))
                    manager.running_state.mark_started(state_context)
                    context.started_services.add(service)
                    pending_restart.discard(service)
                    context.applying_service = None
                for container in sync:
                    if container.name in candidates and not any(name in required_services for name in container.services):
                        self._publish_candidate(container, candidates[container.name], context, ())
            except Exception as error:
                if stop_attempted and not stopped:
                    from collections import Counter
                    try:
                        observed = manager.docker_inspector.get_project_state(sync)
                        originally_running = Counter(service.service for service in actual.services
                                                     if service.state in ("running", "restarting"))
                        still_running = Counter(service.service for service in observed.services
                                                if service.state in ("running", "restarting"))
                        pending_restart.intersection_update(
                            service for service in pending_restart
                            if still_running[service] < originally_running[service])
                        # A failed stop may have changed actual state even when
                        # no complete Compose stop was acknowledged.
                        stopped_owners = tuple(container for container in explicit.target_containers
                                               if container.name in context.initial_running and
                                               not any(container.name in item.logical_containers and
                                                       item.state in ("running", "restarting")
                                                       for item in observed.services))
                        if stopped_owners:
                            status = self._make_context(context.commands, ComposeSelection(
                                selection.project_containers, stopped_owners, (), False))
                            manager.running_state.mark_stopped(status)
                    except Exception as inspection_error:
                        raise ContainerError("Restart failed: {}; recovery inspection failed: {}".format(
                            error, inspection_error)) from error
                failed = getattr(context, "applying_service", None)
                if stopped and failed in context.initial_running_services:
                    pending_restart.add(failed)
                pending_restart.difference_update(context.locally_restored_services)
                if (stopped or stop_attempted) and pending_restart:
                    from copy import copy
                    try:
                        restore = tuple(service for service in selection.services
                                        if service in pending_restart)
                        models = runner.saved_service_models(context, restore)
                        restore_context = copy(context)
                        restore_context.generated_candidates = {}
                        for name, candidate in candidates.items():
                            if context.applied_generation_services.get(name):
                                restore_context.generated_candidates[name] = candidate
                            elif candidate.previous_id is not None:
                                old = copy(candidate)
                                old.generation_id = candidate.previous_id
                                restore_context.generated_candidates[name] = old
                        restored_native = set()
                        for service, model in models.items():
                            container = owners[service]
                            if (container.name in candidates and container.name not in restored_native
                                    and not context.applied_generation_services.get(container.name)):
                                candidates[container.name].restore()
                                container.rollback_config(context)
                                restored_native.add(container.name)
                            runner.apply_saved_services(restore_context, (service,), {"previous.yml": model})
                            if service in context.initial_healthy_services:
                                runner.wait_service_healthy(restore_context, service)
                            else:
                                runner.wait_service_running(restore_context, service)
                            container.on_service_started(restore_context, service)
                            context.service_models.restore((service,))
                            state_context = self._make_context(context.commands, ComposeSelection(
                                selection.project_containers, (container,), (service,), False))
                            manager.running_state.mark_started(state_context)
                    except Exception as rollback_error:
                        raise ContainerError("Restart failed: {}; recovery failed: {}".format(
                            error, rollback_error)) from error
                raise
        with manager.lifecycle.notify_remove(context):
            pass
        if report:
            render_report(manager.logger, get_records(context))
        if candidates:
            try:
                observed = manager.docker_inspector.get_project_state(sync)
            except ContainerError as exc:
                manager.logger.warning("Generated configuration cleanup skipped: %s", exc)
            else:
                running = {service.service for service in observed.services
                           if service.state in ("running", "restarting")}
                for candidate in candidates.values():
                    owner = candidate.container
                    try:
                        if any(service in running and
                               not owner.is_generation_current(context, service, candidate)
                               for service in owner.generation_services):
                            manager.logger.warning("Cannot prune unconfirmed generated configuration for %s",
                                                   owner.name)
                            continue
                        candidate.prune()
                    except (OSError, ContainerError) as exc:
                        manager.logger.warning("Unable to prune generated configuration for %s: %s",
                                               owner.name, exc)

    def _apply_services_with_rollback(self, container, context, services) -> None:
        from copy import copy
        from .artifacts import atomic_write_text_if_changed
        runner = self.manager.compose_runner
        try:
            runner.apply_services(context, services)
            for service in services:
                container.on_service_started(context, service)
            context.service_models.record(services)
        except Exception as error:
            previous = {path: content for path, content in context.saved_compose.items()
                        if context.compose_owners[path] == container.name}
            running = tuple(service for service in services if service in context.initial_running_services)
            started = tuple(service for service in services if service not in context.initial_running_services)
            saved_models = context.service_models
            if started:
                try:
                    runner.stop(context, started)
                    saved_models.restore(started)
                    remaining = (set(context.initial_running_services) |
                                 context.started_services) - set(started)
                    if not any(name in remaining for name in container.services):
                        stopped_context = copy(context)
                        stopped_context.target_containers = [container]
                        stopped_context.is_full_containers = False
                        self.manager.running_state.mark_stopped(stopped_context)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; Compose rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
            if running and (previous or any(name in saved_models.previous for name in running)):
                try:
                    models = runner.saved_service_models(context, running)
                    for path, content in previous.items():
                        atomic_write_text_if_changed(path, content, mode=0o600)
                    for service, model in models.items():
                        runner.apply_saved_services(context, (service,), {"previous.yml": model})
                        if service in context.initial_healthy_services:
                            runner.wait_service_healthy(context, service)
                        else:
                            runner.wait_service_running(context, service)
                        container.on_service_started(context, service)
                    context.service_models.restore(running)
                    restored_context = copy(context)
                    restored_context.target_containers = [container]
                    restored_context.is_full_containers = False
                    self.manager.running_state.mark_started(restored_context)
                    if hasattr(context, "locally_restored_services"):
                        context.locally_restored_services.update(running)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; Compose rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
            raise

    def _require_rollback_model(self, container, context, services) -> None:
        running = tuple(service for service in services if service in context.initial_running_services)
        if not running:
            return
        for service in running:
            if not context.native_running_images.get(service):
                raise ContainerError(
                    "Cannot replace running service {} without its original image ID".format(service))
        import yaml
        saved_services = set()
        for path, owner in context.compose_owners.items():
            if owner != container.name or path not in context.saved_compose:
                continue
            old = yaml.safe_load(context.saved_compose[path]) or {}
            if isinstance(old, dict) and isinstance(old.get("services"), dict):
                saved_services.update(old["services"])
        for service in running:
            if service not in context.service_models.previous and service not in saved_services:
                raise ContainerError(
                    "Cannot replace running service {} without a previous Compose model".format(service))
        self.manager.compose_runner.saved_service_models(context, running)

    def _publish_candidate(self, container, candidate, context, services, record_applied=True) -> None:
        self._require_rollback_model(container, context, services)
        context.generated_candidates[container.name] = candidate
        try:
            candidate.publish()
            if not services:
                return
            container.apply_config(context, candidate, services)
            for service in services:
                container.on_service_started(context, service)
            if record_applied:
                context.service_models.record(services)
                applied = context.applied_generation_services
                applied.setdefault(container.name, []).extend(services)
        except Exception as error:
            self._rollback_candidate(container, candidate, context, services, error)
            raise

    def _rollback_candidate(self, container, candidate, context, services, error) -> None:
        from copy import copy
        from .artifacts import atomic_write_text_if_changed
        runner = self.manager.compose_runner
        try:
            candidate.restore()
        except Exception as rollback_error:
            raise ContainerError("{} apply failed: {}; rollback publication failed: {}".format(
                container.name, error, rollback_error)) from error

        old_compose = {path: content for path, content in context.saved_compose.items()
                       if context.compose_owners[path] == container.name}
        affected = tuple(dict.fromkeys((*context.applied_generation_services.get(container.name, ()), *services)))
        fallback = getattr(candidate, "bootstrap_fallback", False)
        restore_services = tuple(service for service in affected
                                 if service in context.initial_running_services or
                                 (fallback and service in getattr(context, "bootstrapped_services", ())))
        stop_services = tuple(service for service in affected if service not in restore_services)
        try:
            if candidate.previous_id:
                previous = copy(candidate)
                previous.generation_id = candidate.previous_id
                previous.path = __import__("os").path.join(candidate.root, candidate.previous_id)
                previous.changed = True
                context.generated_candidates[container.name] = previous
            else:
                context.generated_candidates.pop(container.name, None)
            context.rollback_service_models = {}
            if fallback:
                context.bootstrap_fallback_services = frozenset(
                    service for service in restore_services
                    if service in getattr(context, "bootstrapped_services", ())
                    and service not in context.initial_running_services)
            if stop_services:
                runner.stop(context, stop_services)
            if restore_services:
                context.rollback_service_models = runner.saved_service_models(context, restore_services)
            if not candidate.previous_id:
                container.rollback_config(context)
            for service, model in context.rollback_service_models.items():
                if candidate.previous_id:
                    container.apply_config(context, previous, (service,))
                else:
                    runner.apply_saved_services(context, (service,), {"previous.yml": model})
                    if service in context.initial_healthy_services:
                        runner.wait_service_healthy(context, service)
                    else:
                        runner.wait_service_running(context, service)
                container.on_service_started(context, service)

            context.service_models.restore(affected)
            if restore_services:
                restored_context = copy(context)
                restored_context.target_containers = [container]
                restored_context.is_full_containers = False
                self.manager.running_state.mark_started(restored_context)
                if hasattr(context, "locally_restored_services"):
                    context.locally_restored_services.update(restore_services)
            elif stop_services and not any(
                    name in (set(context.initial_running_services) |
                             context.started_services) - set(stop_services)
                    for name in container.services):
                stopped_context = copy(context)
                stopped_context.target_containers = [container]
                stopped_context.is_full_containers = False
                self.manager.running_state.mark_stopped(stopped_context)
        except Exception as rollback_error:
            raise ContainerError("{} apply failed: {}; rollback failed: {}".format(
                container.name, error, rollback_error)) from error
        finally:
            if hasattr(context, "bootstrap_fallback_services"):
                del context.bootstrap_fallback_services
            del context.rollback_service_models
            for path, content in old_compose.items():
                atomic_write_text_if_changed(path, content, mode=0o600)

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
