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
                for dependency in service_dependencies(definitions[name]):
                    provider = owners.get(dependency)
                    if provider is None:
                        raise ContainerError(f"Compose dependency {dependency!r} for {owner.name} is not installed")
                    required.add(provider)
                    if owner in roots:
                        roots.add(provider)
                    services.add(dependency)
            if before == (required, services, roots):
                break
        ordered = tuple(self.manager.resolver.resolve_dependencies(required))
        selected_services = tuple(name for container in ordered for name in container.services if name in services)
        bootstrap = {name for container in ordered for name in container.bootstrap_services if name in services}
        ordered_services = order_services(selection.project_containers, selected_services, model, bootstrap)
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
                       (name in context.changed_compose_services or
                        (changed and name in container.generation_services))}
            if pending:
                services.update(pending)
                targets.add(container)
        return self.start_selection(ComposeSelection(explicit.project_containers, tuple(targets),
                                                     tuple(services), explicit.full),
                                    getattr(context, "compose_model", None),
                                    dependency_roots=explicit.target_containers)

    def _make_context(self, commands, selection: ComposeSelection) -> "EventContext":
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = list(selection.project_containers)
        context.target_containers = list(selection.target_containers)
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
        context.original_applied_compose = {}
        context.applied_compose = {}
        context.applied_generation_services = {}
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
                    if was_applied:
                        context.original_applied_compose[path] = previous
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
        selection = self._reconcile_selection(explicit, context, generations)
        context.target_containers = list(selection.target_containers)

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
            for container in sync:
                owner = generations.get(container.name)
                if owner is not None and container.name in generation_targets:
                    with record_phase(context, "prepare-config", container=container.name, logger=manager.logger):
                        owner.on_prepare_config(context)
                        candidates[container.name] = GeneratedCandidate(container, owner.render_config)
            context.generated_candidates = candidates
            selection = self._reconcile_selection(explicit, context,
                {name for name, candidate in candidates.items() if candidate.changed})
            final_services = set(selection.services)
            additional = final_services - required_services
            if additional:
                prepare_images(tuple(name for name in selection.services if name in additional))
            required_services = final_services
            context.target_containers = list(selection.target_containers)
            for container in sync:
                candidate = candidates.get(container.name)
                if candidate is not None:
                    with record_phase(context, "validate-config", container=container.name, logger=manager.logger):
                        generations[container.name].validate_config(context, candidate)

            for container in sync:
                candidate = candidates.get(container.name)
                if candidate is not None:
                    services = tuple(name for name in container.services if name in required_services)
                    self._require_rollback_model(container, candidate, context, services)

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

            if restart and (explicit.full or explicit.services):
                stop_context = self._make_context(context.commands, explicit)
                with manager.lifecycle.notify_stop(stop_context):
                    with record_phase(context, "stop", command=("stop", *explicit.services), logger=manager.logger):
                        runner.stop(stop_context, explicit.services)
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

            owners = {service: container for container in sync for service in container.services}
            services = order_services(sync, selection.services, context.compose_model, bootstrap_available)
            for service in services:
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
        context.service_models.record(services)
        for path, content in context.compose_files.items():
            if context.compose_owners[path] != container.name:
                continue
            # A synchronized owner may contain stopped sibling services. Keep
            # their last-applied models until those services are actually used.
            applied_content = context.applied_compose.get(path, context.saved_compose.get(path, ""))
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
            context.applied_compose[path] = content
            self.manager.artifact_index.record({os.path.relpath(applied, str(self.manager.data_path)): {
                "kind": "compose-applied", "container": container.name, "sha256": sha256_of(content)}})

    def _restore_applied_compose(self, container, context, previous) -> None:
        import os
        from .artifacts import atomic_write_text_if_changed, sha256_of
        applied = context.applied_compose
        original = getattr(context, "original_applied_compose", previous)
        for path in tuple(applied):
            if context.compose_owners[path] != container.name:
                continue
            destination = os.path.join(str(self.manager.data_path), "compose", "applied", container.name + ".yml")
            if path not in original:
                if os.path.exists(destination):
                    os.unlink(destination)
                self.manager.artifact_index.record({}, remove=(
                    os.path.relpath(destination, str(self.manager.data_path)),))
                del applied[path]
                continue
            content = original[path]
            atomic_write_text_if_changed(destination, content)
            applied[path] = content
            self.manager.artifact_index.record({os.path.relpath(destination, str(self.manager.data_path)): {
                "kind": "compose-applied", "container": container.name, "sha256": sha256_of(content)}})

    def _apply_services_with_rollback(self, container, context, services) -> None:
        from copy import copy
        from .artifacts import atomic_write_text_if_changed
        runner = self.manager.compose_runner
        try:
            runner.apply_services(context, services)
            for service in services:
                container.on_service_started(context, service)
            self._record_applied_compose(container, context, services)
        except Exception as error:
            previous = {path: content for path, content in context.saved_compose.items()
                        if context.compose_owners[path] == container.name}
            running = tuple(service for service in services if service in context.initial_running_services)
            saved_models = context.service_models
            if running and (previous or any(name in saved_models.previous for name in running)):
                try:
                    files = dict(context.compose_files)
                    files.update(previous)
                    for path, content in previous.items():
                        atomic_write_text_if_changed(path, content)
                    for service in running:
                        model = context.service_models.previous.get(service)
                        runner.apply_saved_services(context, (service,), {"previous.yml": model} if model else files)
                    context.service_models.restore(running)
                    self._restore_applied_compose(container, context, previous)
                    restored_context = copy(context)
                    restored_context.target_containers = [container]
                    restored_context.is_full_containers = False
                    self.manager.running_state.mark_started(restored_context)
                except Exception as rollback_error:
                    raise ContainerError("{} apply failed: {}; Compose rollback failed: {}".format(
                        container.name, error, rollback_error)) from error
            raise

    def _require_rollback_model(self, container, candidate, context, services) -> None:
        if candidate.previous_id is not None:
            return
        import yaml
        saved_services = set()
        for path, owner in context.compose_owners.items():
            if owner != container.name or path not in context.saved_compose:
                continue
            old = yaml.safe_load(context.saved_compose[path]) or {}
            if isinstance(old, dict) and isinstance(old.get("services"), dict):
                saved_services.update(old["services"])
        for service in services:
            if (service in context.initial_running_services and
                    service not in context.service_models.previous and service not in saved_services):
                raise ContainerError(
                    "Cannot replace running service {} without a previous Compose model".format(service))

    def _publish_candidate(self, container, candidate, context, services, record_applied=True) -> None:
        self._require_rollback_model(container, candidate, context, services)
        context.generated_candidates[container.name] = candidate
        candidate.publish()
        if not services:
            return
        try:
            container.apply_config(context, candidate, services)
            for service in services:
                container.on_service_started(context, service)
            if record_applied:
                self._record_applied_compose(container, context, services)
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
            context.rollback_service_models = context.service_models.previous
            if fallback:
                context.bootstrap_fallback_services = frozenset(
                    service for service in restore_services
                    if service in getattr(context, "bootstrapped_services", ())
                    and service not in context.initial_running_services)
            if old_compose:
                context.rollback_compose_files = dict(context.compose_files)
                context.rollback_compose_files.update(old_compose)

            if stop_services:
                runner.stop(context, stop_services)
            if candidate.previous_id:
                if restore_services:
                    container.apply_config(context, previous, restore_services)
            else:
                container.rollback_config(context)
                saved_files = dict(context.compose_files)
                saved_files.update(context.saved_compose)
                for service in restore_services:
                    model = context.service_models.previous.get(service)
                    if model is not None:
                        runner.apply_saved_services(context, (service,), {"previous.yml": model})
                    elif context.saved_compose:
                        runner.apply_saved_services(context, (service,), saved_files)
                    else:
                        raise ContainerError("No previous Compose model available for service " + service)
                    if service in getattr(context, "initial_healthy_services", ()):
                        runner.wait_service_healthy(context, service)
                    else:
                        runner.wait_service_running(context, service)

            context.service_models.restore(affected)
            self._restore_applied_compose(container, context, old_compose)
            if restore_services:
                restored_context = copy(context)
                restored_context.target_containers = [container]
                restored_context.is_full_containers = False
                self.manager.running_state.mark_started(restored_context)
            elif stop_services and not any(
                    service in context.initial_running_services for service in container.services):
                stopped_context = copy(context)
                stopped_context.target_containers = [container]
                stopped_context.is_full_containers = False
                self.manager.running_state.mark_stopped(stopped_context)
        except Exception as rollback_error:
            raise ContainerError("{} apply failed: {}; rollback failed: {}".format(
                container.name, error, rollback_error)) from error
        finally:
            if hasattr(context, "rollback_compose_files"):
                del context.rollback_compose_files
            if hasattr(context, "bootstrap_fallback_services"):
                del context.bootstrap_fallback_services
            del context.rollback_service_models
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
