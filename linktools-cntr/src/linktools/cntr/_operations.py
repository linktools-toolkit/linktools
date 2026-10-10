#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One preparation, validation and service-application path for cntr commands."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import ContainerError
from .context import OperationContext
from .execution.model import get_records, record_phase, render_report
from .runtime.compose import order_services, order_service_subset, service_dependencies

if TYPE_CHECKING:
    from collections.abc import Sequence, Iterable
    from .container import BaseContainer
    from .manager import ContainerManager
    from .runtime.inspect import ProjectRuntimeState


@dataclass(frozen=True)
class ComposeSelection:
    project_containers: "tuple[BaseContainer, ...]"
    target_containers: "tuple[BaseContainer, ...]"
    services: "tuple[str, ...]"
    full: bool


class ComposeOperations:
    def __init__(self, manager: "ContainerManager") -> None:
        self.manager = manager

    def select(self, names: "Sequence[str] | None" = None, with_dependencies: bool = False,
               metadata_only: bool = False, for_start: bool = False) -> ComposeSelection:
        project = tuple(self.manager.load_installed_config_metadata())
        if not project:
            from .container import NoContainerInstalledError
            raise NoContainerInstalledError("No container installed")
        if not names:
            return ComposeSelection(project, project, (), True)
        installed = {container.name for container in project}
        unknown = [name for name in names if name not in installed]
        if unknown:
            raise ContainerError("Container(s) not installed: " + ", ".join(unknown))
        targets = tuple(container for container in project if container.name in names)
        if with_dependencies:
            targets = tuple(self.manager.resolver.resolve_dependencies(targets))
        services = tuple(dict.fromkeys(service for container in targets for service in container.services))
        if not services and not for_start:
            raise ContainerError("No service found in selected containers")
        return ComposeSelection(project, targets, services, False)

    def start_selection(self, selection: ComposeSelection, model: "dict | None" = None,
                        running_services: "Iterable[str] | None" = None) -> ComposeSelection:
        """Select container groups and providers, then order only real Compose edges."""
        installed = {container.name: container for container in selection.project_containers}
        owners = {service: container for container in selection.project_containers for service in container.services}
        definitions = model["services"] if model is not None else {
            service: container.services[service] for service, container in owners.items()}
        groups = set(selection.target_containers)
        providers = set(groups)
        targets = set(groups)
        services = set(selection.services) if selection.services else {
            service for container in groups for service in container.services}
        running = set(running_services or ())
        previous_consumers = {}
        if running:
            for entry in self.manager.artifact_index.load().values():
                if entry.get("kind") != "generated-config":
                    continue
                consumer = installed.get(entry.get("container"))
                if consumer is None or not set(consumer.services).intersection(running):
                    continue
                for producer in entry.get("producers", ()):
                    previous_consumers.setdefault(producer, set()).add(consumer)
        while True:
            before = set(groups), set(providers), set(targets), set(services)
            for container in tuple(groups):
                for name in container.dependencies:
                    if name not in installed:
                        raise ContainerError("Required dependency {!r} for {} is not installed".format(
                            name, container.name))
                    groups.add(installed[name])
                    providers.add(installed[name])
                    targets.add(installed[name])
                    services.update(installed[name].services)
            required = {container.name for container in providers}
            for container in selection.project_containers:
                for name, selected in container.get_runtime_requirements(required).items():
                    if name not in installed:
                        raise ContainerError("Required provider {!r} is not installed".format(name))
                    providers.add(installed[name])
                    targets.add(installed[name])
                    services.update(selected)
            if running:
                for producer in tuple(providers):
                    exposed = {consumer.name for consumer in previous_consumers.get(producer.name, ())}
                    for declaration in self.manager.integration_snapshot.get(producer.name, ()):
                        exposed.add(declaration.consumer)
                        attached = getattr(declaration, "expose", None)
                        if attached is not None:
                            exposed.add(attached.consumer)
                    for name in exposed:
                        consumer = installed.get(name)
                        if consumer is None:
                            continue
                        active = set(consumer.services) & running
                        if active:
                            providers.add(consumer)
                            targets.add(consumer)
                            services.update(active)
            for service in tuple(services):
                for dependency, options in service_dependencies(definitions[service]).items():
                    if options.get("required", True) is False and dependency not in services:
                        continue
                    if dependency not in owners:
                        raise ContainerError("Compose dependency {!r} for {} is not installed".format(
                            dependency, service))
                    services.add(dependency)
                    targets.add(owners[dependency])
            if before == (groups, providers, targets, services):
                break
        ordered = order_services(selection.project_containers,
                                 tuple(service for service in owners if service in services), model)
        if not ordered:
            raise ContainerError("No runnable service in the selected scope")
        target_order = tuple(dict.fromkeys(owners[service] for service in ordered))
        target_order += tuple(container for container in selection.project_containers
                              if container in targets and container not in target_order)
        return ComposeSelection(selection.project_containers, target_order, tuple(ordered), selection.full)

    def _make_context(self, commands, selection: ComposeSelection) -> OperationContext:
        return OperationContext(
            actions=[commands] if isinstance(commands, str) else list(filter(None, commands)),
            project_containers=list(selection.project_containers),
            target_containers=list(selection.target_containers),
            target_services=selection.services or tuple(
                service for container in selection.target_containers for service in container.services),
            is_full_project=selection.full,
        )

    def up(self, names: "Sequence[str] | None" = None, pull: bool = False,
           report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=False)

    def restart(self, names: "Sequence[str] | None" = None, pull: bool = False,
                report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            self._start(names, pull, report, restart=True)

    def _start(self, names, pull, report, restart):
        import os
        from .artifacts import (AppliedServiceModels, bind_prepared_files, collect_candidates,
                                publish_prepared_files, prune_prepared_files)

        manager = self.manager
        runner = manager.compose_runner
        explicit = self.select(names, for_start=True)
        compose_files, compose_owners, saved_compose = {}, {}, {}
        for path, (kind, owner, content) in collect_candidates(manager, explicit.project_containers).items():
            if kind != "compose":
                continue
            compose_files[path] = content
            compose_owners[path] = owner
            legacy = os.path.join(str(manager.data_path), "compose", "applied", owner + ".yml")
            source = legacy if os.path.exists(legacy) else path
            try:
                with open(source, encoding="utf-8") as stream:
                    saved_compose[path] = stream.read()
            except FileNotFoundError:
                pass
        # Runtime inspection renders Compose files, so retain migration inputs first.
        actual = manager.docker_inspector.get_project_state(explicit.project_containers)
        initial = {item.service for item in actual.services if item.state in ("running", "restarting")}
        selection = self.start_selection(explicit, running_services=initial)
        refresh = frozenset(self.start_selection(explicit).services) if pull else frozenset()
        context = self._make_context(["restart" if restart else "up", pull and "pull"], selection)
        context.refresh_services = refresh
        context.initial_runtime_state = actual
        context.initial_existing_services = frozenset(item.service for item in actual.services)
        context.initial_running_images = {item.service: item.image_id for item in actual.services
                                         if item.service in initial and item.image_id}
        context.initial_running_services = frozenset(initial)
        context.compose_files = compose_files
        context.compose_owners = compose_owners
        context.previous_compose_contents = saved_compose

        with manager.lifecycle.notify_start(context):
            if (tuple(context.target_containers) != selection.target_containers or
                    tuple(context.target_services) != selection.services):
                raise ContainerError("Lifecycle callbacks cannot change the resolved operation targets")
            raw_model = runner.final_model(context)
            ordered = order_services(selection.project_containers, selection.services, raw_model)
            selection = ComposeSelection(selection.project_containers, selection.target_containers,
                                         ordered, selection.full)
            context.target_services = selection.services
            owners = {service: container for container in selection.project_containers for service in container.services}
            model_store = AppliedServiceModels(manager, raw_model, retained_services=initial)
            context.service_models = model_store
            missing = initial.intersection(raw_model["services"]).difference(model_store.previous)
            legacy = set()
            if missing:
                import yaml
                for text in context.previous_compose_contents.values():
                    try:
                        old = yaml.safe_load(text) or {}
                    except yaml.YAMLError:
                        continue
                    if isinstance(old, dict) and isinstance(old.get("services", {}), dict):
                        legacy.update(missing.intersection(old.get("services", {})))
            if legacy:
                model_store.retain_previous(runner.saved_service_models(context, tuple(sorted(legacy))))
            model = bind_prepared_files(context, raw_model, model_store.previous)
            context.compose_model = manager.image_preparer.with_build_revisions(
                model, selection.project_containers, selection.services)
            model_store.set_model(context.compose_model)
            image_plan = manager.image_preparer.plan(
                context.compose_model, selection.services, force_pull=pull,
                refresh_services=context.refresh_services)
            if image_plan.pull:
                with record_phase(context, "pull", command=tuple(runner.pull_args(image_plan.pull)),
                                  logger=manager.logger):
                    runner.pull(context, image_plan.pull)
            if image_plan.build:
                refreshing = tuple(name for name in image_plan.build if name in context.refresh_services)
                unchanged = tuple(name for name in image_plan.build if name not in context.refresh_services)
                for services, update in ((unchanged, False), (refreshing, True)):
                    if not services:
                        continue
                    options = runner.options_for_build(services, pull=update)
                    with record_phase(context, "build", command=tuple(runner.build_args(options)),
                                      logger=manager.logger):
                        runner.build(context, options)
                        manager.image_preparer.verify_builds(context.compose_model, services)
            target_image_ids = {name: manager.image_preparer.image_id(
                context.compose_model["services"][name]["image"]) for name in selection.services}
            with record_phase(context, "check", logger=manager.logger):
                manager.lifecycle.check(context)
                self._require_rollback_models(context, selection.services)

            stop_set = set(explicit.services or (
                service for container in explicit.target_containers for service in container.services)) if restart else set()
            pending_restart = stop_set & initial
            running = set(initial)
            failed = None
            applied_services = []
            updated, recreated = set(), set()
            stop_attempted = False
            stopped = False
            try:
                if stop_set:
                    stop_context = self._make_context(context.actions, explicit)
                    stop_context.compose_model = context.compose_model
                    with manager.lifecycle.notify_stop(stop_context):
                        stop_attempted = True
                        with record_phase(context, "stop", command=("stop", *explicit.services), logger=manager.logger):
                            runner.stop(stop_context, explicit.services)
                        stopped = True
                        running.difference_update(stop_set)
                        self._update_running_state(context, running, explicit.target_containers)
                for service in selection.services:
                    failed = service
                    spec = context.compose_model["services"][service]
                    before_image = context.initial_running_images.get(service)
                    image_changed = service in initial and before_image != target_image_ids[service]
                    binds = {str(spec.get(key)).split(":", 1)[1] for key in
                             ("network_mode", "ipc", "pid")
                             if str(spec.get(key, "")).startswith("service:")}
                    binds.update(str(value).split(":", 1)[0] for value in
                                 spec.get("volumes_from", ())
                                 if not str(value).startswith("container:"))
                    recreate = (service in model_store.changed_services or image_changed or
                                bool(binds & recreated))
                    cascade = any(options.get("restart") and dependency in updated for
                                  dependency, options in service_dependencies(spec).items())
                    with record_phase(context, "up", container=owners[service].name, logger=manager.logger):
                        if cascade and service in initial and not recreate and service not in stop_set:
                            runner.restart_service(context, service)
                        else:
                            if manager.image_preparer.image_id(spec["image"]) != target_image_ids[service]:
                                raise ContainerError("Selected image changed during deployment: " + service)
                            runner.apply_service(context, service, recreate=recreate)
                        active = runner.wait_service_ready(context, service)
                        model_store.record((service,))
                    if active:
                        running.add(service)
                    else:
                        running.discard(service)
                    pending_restart.discard(service)
                    self._update_running_state(context, running, (owners[service],))
                    applied_services.append(service)
                    if recreate and service in initial:
                        recreated.add(service)
                    if recreate or cascade or service in stop_set or service not in initial:
                        updated.add(service)
                    failed = None
                for service, recreate in self._dependent_actions(
                        context, selection.services, initial, updated, recreated):
                    failed = service
                    owner = owners.get(service)
                    with record_phase(context, "restart-dependent", container=owner.name if owner else None,
                                      logger=manager.logger):
                        import yaml
                        saved = runner.saved_service_models(context, (service,))
                        previous_model = yaml.safe_load(saved[service])
                        if recreate:
                            runner.apply_saved_services(context, (service,),
                                                        {"previous.yml": saved[service]})
                        else:
                            runner.restart_service(context, service, model=previous_model)
                        runner.wait_service_ready(context, service, model=previous_model)
                    failed = None
            except Exception as error:
                try:
                    restore = set()
                    if stop_attempted and not stopped:
                        from collections import Counter
                        observed = manager.docker_inspector.get_project_state(selection.project_containers)
                        original_counts = Counter(item.service for item in actual.services
                                                  if item.state in ("running", "restarting"))
                        observed_counts = Counter(item.service for item in observed.services
                                                  if item.state in ("running", "restarting"))
                        restore.update(service for service in pending_restart
                                       if observed_counts[service] < original_counts[service])
                        running = set(observed_counts)
                    elif stopped:
                        restore.update(pending_restart)
                    cleanup = set()
                    if failed is not None:
                        affected = self._shared_input_consumers(context, failed, applied_services) | {failed}
                        restore.update(affected & initial)
                        cleanup = affected - initial
                        if cleanup:
                            discarded = set(cleanup)
                            for service, _ in self._dependent_actions(
                                    context, restore | cleanup, running, set(cleanup), discarded,
                                    applied=set(applied_services) - restore):
                                if service in initial:
                                    restore.add(service)
                                else:
                                    cleanup.add(service)
                                discarded.add(service)
                            stopped_new = tuple(service for service in reversed(selection.services) if service in cleanup)
                            runner.stop(context, stopped_new)
                            model_store.restore(stopped_new)
                            running.difference_update(cleanup)
                    touched = restore | cleanup
                    if restore:
                        import yaml
                        successful = set(applied_services) - restore - cleanup
                        actions = {service: True for service in (*context.compose_model["services"], *model_store.previous)
                                   if service in restore}
                        actions.update(self._dependent_actions(
                            context, restore, running, set(restore), set(restore), applied=successful))
                        # Capture all old inputs before mutation; order old and current actions together.
                        saved = runner.saved_service_models(context, tuple(
                            service for service in (*context.compose_model["services"], *model_store.previous)
                            if service in actions and service not in successful))
                        models = {service: (context.compose_model if service in successful else
                                            yaml.safe_load(saved[service])) for service in actions}
                        specifications = {service: model["services"][service] for service, model in models.items()}
                        for service in order_service_subset(context.project_containers, specifications):
                            if actions[service]:
                                if service in successful:
                                    spec = specifications[service]
                                    if manager.image_preparer.image_id(spec["image"]) != target_image_ids[service]:
                                        raise ContainerError("Selected image changed during recovery: " + service)
                                    runner.apply_service(context, service, recreate=True)
                                else:
                                    runner.apply_saved_services(context, (service,), {"previous.yml": saved[service]})
                            else:
                                runner.restart_service(context, service, model=models[service])
                            active = runner.wait_service_ready(context, service, model=models[service])
                            if service in restore:
                                model_store.restore((service,))
                            if active:
                                running.add(service)
                            else:
                                running.discard(service)
                            touched.add(service)
                    self._update_running_state(context, running, tuple(dict.fromkeys(
                        owners[service] for service in touched if service in owners)))
                except Exception as recovery_error:
                    raise ContainerError("Operation failed: {}; recovery failed: {}".format(
                        error, recovery_error)) from error
                raise
            try:
                publish_prepared_files(context, selection.services)
            except Exception as error:
                raise ContainerError("Services were applied; publishing file references failed: {}".format(error)) from error
        manager.lifecycle.reconcile_removed(context)
        try:
            prune_prepared_files(context, model_store)
        except (OSError, ContainerError) as error:
            manager.logger.warning("Prepared file cleanup failed: %s", error)
        if report:
            render_report(manager.logger, get_records(context))

    def _dependent_actions(self, context, selected, running, updated, recreated,
                           applied=()):
        """Propagate only declared Compose restart edges and stale namespace binds."""
        import yaml
        # Unknown legacy services have no trustworthy dependency model to act on.
        known = set(context.service_models.previous) | set(applied)
        if context.is_full_project:
            known.intersection_update(context.compose_model["services"])
        running = set(running).intersection(known)
        definitions = {}
        for name in running:
            saved = None if name in applied else context.service_models.previous.get(name)
            model = yaml.safe_load(saved) if saved else context.compose_model
            definitions[name] = model["services"][name]
        pending = {name: definitions[name] for name in sorted(running - set(selected))}
        for name in order_service_subset(context.project_containers, pending):
            spec = definitions[name]
            binds = {str(spec.get(key)).split(":", 1)[1] for key in ("network_mode", "ipc", "pid")
                     if str(spec.get(key, "")).startswith("service:")}
            binds.update(str(value).split(":", 1)[0] for value in spec.get("volumes_from", ())
                         if not str(value).startswith("container:"))
            rebuild = bool(binds & recreated)
            restart = any(options.get("restart") and parent in updated
                          for parent, options in service_dependencies(spec).items())
            if rebuild or restart:
                yield name, rebuild
                updated.add(name)
                if rebuild:
                    recreated.add(name)

    def _shared_input_consumers(self, context, failed, applied):
        """Restore applied peers only when they share a changed file input."""
        from pathlib import Path
        import yaml

        def changed_sources(service):
            if service not in context.compose_model["services"]:
                return []
            previous = context.service_models.previous.get(service)
            old = yaml.safe_load(previous)["services"][service] if previous is not None else {}
            mounts = {item["target"]: item["source"] for item in old.get("volumes", ())
                      if isinstance(item, dict) and item.get("type") == "bind"}
            return [Path(item["source"]) for item in context.compose_model["services"][service].get("volumes", ())
                    if isinstance(item, dict) and item.get("type") == "bind" and
                    mounts.get(item["target"]) != item["source"]]

        sources = {service: changed_sources(service) for service in applied}
        changed = changed_sources(failed)
        affected = set()
        while True:
            found = {service for service, inputs in sources.items() if service not in affected and any(
                source == other or (source.is_dir() and source in other.parents) or
                (other.is_dir() and other in source.parents)
                for source in changed for other in inputs)}
            if not found:
                return affected
            affected.update(found)
            changed.extend(source for service in found for source in sources[service])

    def _require_rollback_models(self, context, services):
        from .errors import ContainerError
        running = tuple(service for service in services if service in context.initial_running_services)
        for service in running:
            if not context.initial_running_images.get(service):
                raise ContainerError("Cannot replace running service {} without its original image ID".format(service))
        if running:
            self.manager.compose_runner.saved_service_models(context, running)

    def _update_running_state(self, context, running, containers):
        from copy import copy
        for container in containers:
            state = copy(context)
            state.target_containers = [container]
            state.is_full_project = False
            if set(container.services) & running:
                self.manager.running_state.mark_started(state)
            else:
                self.manager.running_state.mark_stopped(state)

    def down(self, names: "Sequence[str] | None" = None, report: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:project:" + self.manager.project_name):
            manager = self.manager
            selection = self.select(names)
            context = self._make_context("down", selection)
            with manager.lifecycle.notify_stop(context):
                with record_phase(context, "down", command=("down", *selection.services), logger=manager.logger):
                    manager.compose_runner.down(context, selection.services)
                manager.running_state.mark_stopped(context)
            manager.lifecycle.reconcile_removed(context)
            if report:
                render_report(manager.logger, get_records(context))

    def render(self, names: "Sequence[str] | None" = None, with_dependencies: bool = False,
               output_format: "str | None" = None, check: bool = False) -> "int | None":
        selection = self.select(names, with_dependencies=with_dependencies)
        context = self._make_context("compose", selection)
        return self.manager.compose_runner.config(
            context, selection.services, output_format=output_format, quiet=check)

    def status(self) -> "tuple[tuple[BaseContainer, ...], ProjectRuntimeState]":
        from linktools.core import Config
        with Config.read_only_resolution():
            containers = tuple(self.manager.load_installed_config_metadata())
            return containers, self.manager.docker_inspector.get_project_state(containers)
