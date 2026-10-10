#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Docker Compose command assembly.

Both the CLI (``ct-cntr up/restart/down``) and the per-container ``exec``
subcommands build the same kind of ``docker compose`` argument lists. This
module centralizes that assembly so the two paths cannot drift.

Proxy build arguments and action-specific command options are centralized here
so root, restart, and per-container execution share one command builder.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence, Mapping
    from typing import Any, Iterable, Iterator
    from linktools.runtime import Process
    from ..container import BaseContainer
    from ..context import OperationContext
    from .structured import CommandResult
    from ..manager import ContainerManager


_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


def namespace_dependencies(spec: "Mapping[str, Any]") -> "tuple[str, ...]":
    """Project services whose container identities are bound into this service."""
    result = [str(value).split(":", 1)[0] for value in spec.get("volumes_from") or ()
              if not str(value).startswith("container:")]
    result.extend(str(spec[key]).split(":", 1)[1] for key in ("network_mode", "ipc", "pid")
                  if str(spec.get(key, "")).startswith("service:"))
    return tuple(dict.fromkeys(result))


def service_dependencies(spec: "dict[str, Any]") -> "dict[str, dict[str, Any]]":
    """Normalize Compose's explicit and implicit service dependency edges."""
    declared = spec.get("depends_on") or {}
    if isinstance(declared, dict):
        result = {name: dict(options) if isinstance(options, dict) else {}
                  for name, options in declared.items()}
    else:
        result = {name: {} for name in declared}
    implicit = [value.split(":", 1)[0] for value in spec.get("links") or ()]
    implicit.extend(namespace_dependencies(spec))
    for name in implicit:
        result.setdefault(name, {})
    return result


def order_services(containers: "Iterable[BaseContainer]", services: "Iterable[str]",
                   model: "dict[str, Any] | None" = None) -> "tuple[str, ...]":
    """Order real Compose dependencies; container grouping adds no start edge."""
    from ..errors import ContainerError
    owners = {name: container for container in containers for name in container.services}
    selected = tuple(dict.fromkeys(services))
    pending = set(selected)
    definitions = model["services"] if model is not None else {
        name: owner.services[name] for name, owner in owners.items()}
    dependencies = {}
    for name in selected:
        edges = set()
        for dependency, options in service_dependencies(definitions[name]).items():
            if dependency not in pending:
                if options.get("required", True) is False:
                    continue
                raise ContainerError("Unselected Compose dependency {} for {}".format(dependency, name))
            edges.add(dependency)
        dependencies[name] = edges
    result = []
    positions = {name: index for index, name in enumerate(selected)}
    while pending:
        ready = [name for name in pending if not dependencies[name] & pending]
        if not ready:
            raise ContainerError("Compose dependency cycle at " + ", ".join(sorted(pending)))
        name = min(ready, key=positions.__getitem__)
        pending.remove(name)
        result.append(name)
    return tuple(result)

def order_service_subset(containers: "Iterable[BaseContainer]",
                         specifications: "dict[str, dict[str, Any]]") -> "tuple[str, ...]":
    """Order actions by their effective dependencies, leaving other services alone."""
    graph = {service: {"depends_on": {
        name: options for name, options in service_dependencies(spec).items()
        if name in specifications}} for service, spec in specifications.items()}
    return order_services(containers, tuple(specifications), {"services": graph})


@dataclass
class ComposeOptions:
    """Resolved options for a single compose build/up invocation."""

    pull: bool = False
    remove_orphans: bool = False
    services: "list[str]" = field(default_factory=list)
    # CLI `up` and both `exec up`/`exec restart` include proxy --build-args;
    # CLI `restart` deliberately never did.
    include_proxy_build_args: bool = True


class ComposeRunner:
    """Assemble and run docker-compose commands for a ContainerManager."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def collect_services(self, context: "OperationContext") -> "list[str]":
        """Service names for the targeted containers; empty for "all" runs."""
        if context.is_full_project:
            return []
        services: "list[str]" = []
        for container in context.target_containers:
            services.extend(container.services.keys())
        if not services:
            # Imported lazily to keep runtime.compose free of a module-level
            # dependency on ..container (which imports this module).
            from ..errors import ContainerError
            names = ",".join(c.name for c in context.target_containers)
            raise ContainerError(f"No service found in container `{names}`")
        return services

    def collect_proxy_build_args(self) -> "list[str]":
        """``--build-arg`` entries for configured HTTP proxies (both cases)."""
        options: "list[str]" = []
        for key in _PROXY_ENV_KEYS:
            if key in os.environ:
                options.extend(["--build-arg", f"{key}={os.environ[key]}"])
            upper = key.upper()
            if upper in os.environ:
                options.extend(["--build-arg", f"{upper}={os.environ[upper]}"])
        return options

    def build_args(self, options: ComposeOptions) -> "list[str]":
        args: "list[str]" = ["build"]
        if options.pull:
            args.append("--pull")
        if options.include_proxy_build_args:
            args.extend(self.collect_proxy_build_args())
        args.extend(options.services)
        return args

    def up_args(self, options: ComposeOptions) -> "list[str]":
        args: "list[str]" = ["up", "--detach", "--no-build"]
        args.extend(["--pull", "never"])
        if options.remove_orphans:
            args.append("--remove-orphans")
        args.extend(options.services)
        return args

    def build(self, context: "OperationContext", options: "ComposeOptions") -> int:
        from ..container import SourceContainer
        selected = set(options.services)
        for container in context.target_containers:
            if not selected or selected.intersection(container.services):
                if isinstance(container, SourceContainer):
                    container.prepare_build_context()
                container.get_docker_file_path()
        with self._model_args(context) as args:
            return self.manager.runtime.create_docker_process(*args, *self.build_args(options)).check_call()
    def pull_args(self, services: "Sequence[str]") -> "list[str]":
        return ["pull", "--ignore-buildable", *services]

    def pull(self, context: "OperationContext", services: "Sequence[str]") -> int:
        with self._model_args(context) as args:
            return self.manager.runtime.create_docker_process(*args, *self.pull_args(services)).check_call()
    def options_for_build(self, services: "Sequence[str]", pull: bool = False) -> ComposeOptions:
        return ComposeOptions(pull=pull, services=list(services))

    def final_model(self, context: "OperationContext", preserve_disabled: bool = False,
                    privilege: "bool | None" = None) -> "dict[str, Any]":
        from ..artifacts import collect_candidates
        files = [content for kind, owner, content in collect_candidates(
            self.manager, context.project_containers).values() if kind == "compose"]
        if not files:
            from ..errors import ContainerError
            raise ContainerError("No Compose files in selected project")
        services = ()
        if not context.is_full_project and any(
                spec.get("profiles") for container in context.project_containers for spec in container.services.values()):
            services = context.target_services or ()
        with self._saved_compose_args(context, files) as args:
            model = self._resolved_model(self.manager.runtime.create_docker_process(
                *args, *self.config_args(services=services, output_format="json"), privilege=privilege, capture_output=True))
            if preserve_disabled:
                # Native un-interpolated output retains disabled services without
                # loading their env_files. Keep those definitions for orphan detection.
                complete = self._resolved_model(self.manager.runtime.create_docker_process(
                    *args, *self.config_args(output_format="json"), "--no-interpolate", privilege=privilege, capture_output=True))
                for name, value in model.items():
                    if isinstance(value, dict) and isinstance(complete.get(name), dict):
                        complete[name].update(value)
                    else:
                        complete[name] = value
                model = complete
            return model
    def _resolved_model(self, process: "Process") -> "dict[str, Any]":
        result = self.manager.structured_runner.execute_json(process, check=True)
        if not isinstance(result, dict) or not isinstance(result.get("services"), dict):
            from ..errors import ContainerError
            raise ContainerError("Docker Compose returned an invalid final model")
        return result

    def stop(self, context: "OperationContext", services: "Sequence[str]") -> int:
        with self._model_args(context) as args:
            return self.manager.runtime.create_docker_process(*args, "stop", *services).check_call()
    def down(self, context: "OperationContext", services: "Sequence[str]") -> int:
        with self._model_args(context) as args:
            return self.manager.runtime.create_docker_process(*args, "down", *services).check_call()
    def config_args(
            self,
            services: "Sequence[str]" = (),
            output_format: "str | None" = None,
            quiet: bool = False,
    ) -> "list[str]":
        """Shared ``docker compose config`` argument builder.

        Reused by ``compose config``/``compose validate`` and by Plan
        preflight, so all three can never drift from each other.
        """
        args: "list[str]" = ["config"]
        if quiet:
            args.append("--quiet")
        if output_format:
            args.extend(["--format", output_format])
        args.extend(services)
        return args

    def config(self, context: "OperationContext", services: "Sequence[str]" = (),
               output_format: "str | None" = None, quiet: bool = False) -> int:
        from ..artifacts import collect_candidates
        from ..errors import ContainerError
        files = [content for kind, owner, content in collect_candidates(
            self.manager, context.project_containers).values() if kind == "compose"]
        if not files:
            raise ContainerError("No Compose files in selected project")
        with self._saved_compose_args(context, files) as args:
            return self.manager.runtime.create_docker_process(
                *args, *self.config_args(services, output_format, quiet), privilege=False).check_call()
    def isolated_service_args(self, model: "dict[str, Any]", service: str,
                              command: "Sequence[str]", environment: "Mapping[str, object] | None" = None,
                              network: bool = False,
                              mount_overrides: "Mapping[str, str] | None" = None) -> "list[str]":
        """Target image/env/mounts, deliberately excluding ports, IPs and dependencies."""
        spec = model["services"][service]

        def raw(value: "Any") -> "Any":
            # `compose config` serializes dollars for another Compose load.
            # Raw Docker arguments have no interpolation pass of their own.
            return value.replace("$$", "$") if isinstance(value, str) else value

        image = raw(spec.get("image")) or (self.manager.project_name + "-" + service)
        args = ["run", "--rm", "--network", "bridge" if network else "none"]
        overrides = dict(mount_overrides or {})
        for mount in spec.get("volumes", ()):
            if not isinstance(mount, dict) or mount.get("type") not in ("bind", "volume"):
                raise ValueError("Native validation requires resolved bind/volume mounts")
            source = mount.get("source")
            if mount["type"] == "volume":
                source = model.get("volumes", {}).get(source, {}).get("name", source)
            target = raw(mount["target"])
            if target in overrides:
                value = "type=bind,source={},target={}".format(overrides.pop(target), target)
            else:
                value = "type={},source={},target={}".format(mount["type"], raw(source), target)
            if mount.get("read_only"):
                value += ",readonly"
            args.extend(["--mount", value])
        if overrides:
            raise ValueError("No resolved service mount for " + ", ".join(overrides))
        for category in ("secrets", "configs"):
            for item in spec.get(category, ()):
                name = item if isinstance(item, str) else item["source"]
                definition = model.get(category, {}).get(name, {})
                source = definition.get("file")
                if not source:
                    from ..errors import ContainerError
                    raise ContainerError("Isolated validation requires a local {} file".format(category))
                target = (item.get("target") if isinstance(item, dict) else None) or name
                if not target.startswith("/"):
                    target = ("/run/secrets/" if category == "secrets" else "/") + target
                args.extend(["--mount", "type=bind,source={},target={},readonly".format(raw(source), raw(target))])
        values = {key: raw(value) for key, value in (spec.get("environment") or {}).items()}
        values.update(environment or {})
        for key, value in values.items():
            if value is not None:
                args.extend(["--env", "{}={}".format(key, value)])
        if spec.get("user"):
            args.extend(["--user", str(raw(spec["user"]))])
        if spec.get("working_dir"):
            args.extend(["--workdir", raw(spec["working_dir"])])
        args.extend(["--entrypoint", command[0], image, *command[1:]])
        return args

    def _native_validation_model(self, context: "OperationContext", service: str) -> "dict[str, Any]":
        return context.compose_model if context.compose_model is not None else self.final_model(context)
    def validate_service(self, context: "OperationContext", service: str,
                         command: "Sequence[str]", environment: "Mapping[str, object] | None" = None,
                         network: bool = False, check: bool = True,
                         mount_overrides: "Mapping[str, str] | None" = None) -> "CommandResult":
        model = self._native_validation_model(context, service)
        args = self.isolated_service_args(model, service, command, environment, network, mount_overrides)
        result = self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_process(*args, capture_output=True), check=False)
        if check and not result.succeeded:
            from ..errors import ContainerError
            # Command output can contain expanded credentials.
            raise ContainerError("Native validation failed for service {} (exit {})".format(
                service, result.returncode))
        return result

    def run_isolated_service(self, context: "OperationContext", service: str,
                             command: "Sequence[str]") -> None:
        """Run a one-shot service maintenance command with resolved mounts, no network or ports."""
        model = self._native_validation_model(context, service)
        args = self.isolated_service_args(model, service, command)
        result = self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_process(*args, capture_output=True), check=False)
        if not result.succeeded:
            from ..errors import ContainerError
            raise ContainerError("Isolated service command failed for {} (exit {})".format(
                service, result.returncode))

    def apply_service_args(self, service: str, recreate: bool = False, remove_orphans: bool = False) -> "list[str]":
        args = self.up_args(ComposeOptions(services=[], remove_orphans=remove_orphans))
        args.append("--no-deps")
        if recreate:
            args.append("--force-recreate")
        args.append(service)
        return args

    def apply_service(self, context: "OperationContext", service: str, recreate: bool = False) -> int:
        self.wait_service_dependencies(context, service)
        with self._model_args(context) as args:
            return self.manager.runtime.create_docker_process(
                *args, *self.apply_service_args(service, recreate, context.is_full_project)).check_call()
    def restart_service(self, context: "OperationContext", service: str,
                        model: "dict[str, Any] | None" = None) -> int:
        """Restart the existing container without applying pending Compose changes."""
        with self._model_args(context, model=model) as args:
            return self.manager.runtime.create_docker_process(
                *args, "restart", "--no-deps", service).check_call()

    def exec_service(self, context: "OperationContext", service: str, command: "Sequence[str]",
                     check: bool = True) -> "CommandResult":
        return self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_compose_process(
                context.project_containers, "exec", "-T", service, *command, capture_output=True), check=check)

    def wait_service_healthy(self, context: "OperationContext", service: str,
                             timeout: "int | None" = 30) -> None:
        import time
        from ..errors import ContainerError
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self.manager.docker_inspector.get_project_state(context.project_containers)
            matches = [item for item in state.services if item.service == service]
            if matches and all(item.state == "running" and item.health == "healthy" for item in matches):
                return
            if timeout is None:
                if not matches:
                    raise ContainerError("Dependency service {} is unavailable".format(service))
                if any(item.state not in ("running", "restarting") for item in matches):
                    raise ContainerError("Dependency service {} is not running".format(service))
                if any(item.health is None for item in matches):
                    raise ContainerError("Dependency service {} has no healthcheck".format(service))
                if any(item.health == "unhealthy" for item in matches):
                    raise ContainerError("Dependency service {} is unhealthy".format(service))
            if deadline is not None and time.monotonic() >= deadline:
                raise ContainerError("Service {} did not become healthy".format(service))
            time.sleep(0.5)

    @staticmethod
    def _is_scaled_to_zero(spec: "Mapping[str, Any]") -> bool:
        return spec.get("scale") == 0 or spec.get("deploy", {}).get("replicas") == 0

    def wait_service_dependencies(self, context: "OperationContext", service: str,
                                  model: "dict[str, Any] | None" = None) -> None:
        """Use the executing model's conditions without imposing a task deadline."""
        from ..errors import ContainerError
        restored = model is not None
        if model is None:
            model = getattr(context, "compose_model", None) or self.final_model(context)
        for dependency, options in service_dependencies(model["services"][service]).items():
            if self._is_scaled_to_zero(model["services"].get(dependency, {})):
                continue
            condition = options.get("condition", "service_started")
            if (options.get("required", True) is False and
                    dependency not in (getattr(context, "target_services", None) or ())):
                state = self.manager.docker_inspector.get_project_state(context.project_containers)
                matches = [item for item in state.services if item.service == dependency]
                available = any(
                    item.state in ("running", "restarting") or
                    (condition == "service_completed_successfully" and
                     item.state == "exited" and item.exit_code == 0)
                    for item in matches)
                if not available:
                    self.manager.logger.warning("Optional Compose dependency %s is unavailable", dependency)
                    continue
            if condition not in ("service_healthy", "service_completed_successfully", "service_started"):
                raise ContainerError("Unsupported Compose dependency condition: " + str(condition))
            try:
                if condition == "service_healthy":
                    self.wait_service_healthy(context, dependency, timeout=None)
                elif condition == "service_completed_successfully":
                    self.wait_service_completed(context, dependency, timeout=None)
                elif condition == "service_started":
                    # Normal application already acknowledged each ordered `up`.
                    # Rollback does not start dependencies outside its restore set.
                    if restored:
                        state = self.manager.docker_inspector.get_project_state(context.project_containers)
                        matches = [item for item in state.services if item.service == dependency]
                        if not matches or any(not (item.state == "running" or
                                (item.state == "exited" and item.exit_code == 0)) for item in matches):
                            raise ContainerError("Dependency service {} is unavailable".format(dependency))
            except ContainerError:
                if options.get("required", True) is False:
                    observed = self.manager.docker_inspector.get_project_state(context.project_containers)
                    matches = [item for item in observed.services if item.service == dependency]
                    unavailable = not matches or any(
                        item.state not in ("running", "restarting") or
                        (condition == "service_healthy" and item.health in (None, "unhealthy"))
                        for item in matches)
                    if unavailable:
                        self.manager.logger.warning("Optional Compose dependency %s is not ready", dependency)
                        continue
                raise

    def wait_service_completed(self, context: "OperationContext", service: str,
                               timeout: "int | None" = 30) -> None:
        import time
        from ..errors import ContainerError
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self.manager.docker_inspector.get_project_state(context.project_containers)
            matches = [item for item in state.services if item.service == service]
            if any(item.state == "dead" or
                   (item.state == "exited" and item.exit_code != 0) for item in matches):
                raise ContainerError("Dependency service {} failed".format(service))
            if matches and all(item.state == "exited" and item.exit_code == 0 for item in matches):
                return
            if timeout is None and (not matches or any(
                    item.state not in ("running", "restarting", "exited") for item in matches)):
                raise ContainerError("Dependency service {} is unavailable".format(service))
            if deadline is not None and time.monotonic() >= deadline:
                raise ContainerError("Service {} did not complete successfully".format(service))
            time.sleep(0.5)

    @contextmanager
    def _saved_compose_args(self, context: "OperationContext",
                            contents: "Iterable[str]") -> "Iterator[list[str]]":
        import tempfile
        with tempfile.TemporaryDirectory(prefix="cntr-compose-") as directory:
            paths = []
            for index, content in enumerate(contents):
                path = os.path.join(directory, "{}.yml".format(index))
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(content)
                paths.append(path)
            yield self.compose_args(paths)
    def _legacy_rollback_files(self, context: "OperationContext",
                               services: "Sequence[str]") -> "list[str]":
        """Keep only old Compose files needed by these services and their references."""
        import yaml
        from ..errors import ContainerError

        # Past service ownership comes from captured declarations, not today's owners.
        models, pinned = {}, set()
        actual = getattr(context, "initial_runtime_state", None)
        namespaces = {item.service: item.namespace_bindings for item in actual.services
                      if getattr(item, "namespace_bindings", None) is not None} if actual is not None else {}
        for path, text in context.previous_compose_contents.items():
            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError:
                continue
            if isinstance(data, dict) and isinstance(data.get("services", {}), dict):
                for service, spec in data.get("services", {}).items():
                    bindings = namespaces.get(service)
                    if bindings is not None and isinstance(spec, dict):
                        replacement = dict(spec)
                        for field in ("network_mode", "ipc", "pid", "volumes_from"):
                            if field not in bindings:
                                replacement.pop(field, None)
                                continue
                            value = bindings[field]
                            shared = (bool(value) if field == "volumes_from" else
                                      str(value).startswith(("service:", "container:")))
                            if field in replacement or shared:
                                replacement[field] = value
                        data["services"][service] = replacement
                        pinned.add(path)
                models[path] = data
        included, checked = set(), set()
        queue = list(services)
        while queue:
            name = queue.pop()
            if name in checked:
                continue
            checked.add(name)
            files = [path for path, model in models.items() if name in model.get("services", {})]
            if not files:
                raise ContainerError("No previous Compose declaration for service " + name)
            for path in files:
                spec = models[path]["services"][name]
                if not isinstance(spec, dict):
                    raise ContainerError("Invalid previous Compose service " + name)
                included.add(path)
                queue.extend(service_dependencies(spec))

        # Shared resources can be defined by another owner, independently of
        # its services. Include those declarations only when referenced.
        needed = {key: set() for key in ("networks", "volumes", "secrets", "configs")}
        for path in included:
            for name, spec in models[path].get("services", {}).items():
                if name not in checked or not isinstance(spec, dict):
                    continue
                networks = spec.get("networks")
                if networks:
                    needed["networks"].update(networks)
                for mount in spec.get("volumes", ()):
                    if isinstance(mount, dict) and mount.get("type") == "volume":
                        needed["volumes"].add(mount.get("source"))
                for category in ("secrets", "configs"):
                    needed[category].update(item if isinstance(item, str) else item.get("source")
                                            for item in spec.get(category, ()))
        for category, names in needed.items():
            defined = set().union(*(models[path].get(category, {}) for path in included))
            missing = names - defined
            for path, data in models.items():
                if path in included or not missing:
                    continue
                resources = data.get(category)
                if not isinstance(resources, dict) or not missing.intersection(resources):
                    continue
                included.add(path)
                missing.difference_update(resources)
        return [yaml.safe_dump(models[path]) if path in pinned else text
                for path, text in context.previous_compose_contents.items() if path in included]

    def saved_service_models(self, context: "OperationContext",
                             services: "Sequence[str]") -> "dict[str, str]":
        import yaml
        from ..errors import ContainerError
        texts, specifications = {}, {}
        legacy = None
        for service in dict.fromkeys(services):
            text = context.service_models.previous.get(service)
            if text is not None:
                model = yaml.safe_load(text)
            else:
                if legacy is None:
                    if not context.previous_compose_contents:
                        raise ContainerError("No previous Compose model available for service " + service)
                    old_files = self._legacy_rollback_files(context, services)
                    with self._saved_compose_args(context, old_files) as args:
                        legacy = self._resolved_model(self.manager.runtime.create_docker_process(
                            *args, *self.config_args(services=services, output_format="json"), capture_output=True))
                model = legacy
                text = yaml.safe_dump(model)
            texts[service] = text
            specifications[service] = model["services"][service]
        return {service: texts[service] for service in order_service_subset(context.project_containers, specifications)}
    def apply_saved_services(self, context: "OperationContext", services: "Sequence[str]",
                             files: "dict[str, str]", *,
                             image_ids: "Mapping[str, str] | None" = None) -> None:
        import yaml
        from ..errors import ContainerError
        services = tuple(dict.fromkeys(services))
        if not services:
            return
        if not files:
            raise ContainerError("No saved Compose files available")
        overlay = {}
        for service in services:
            image = (context.initial_runtime_state.running_images if image_ids is None else image_ids).get(service)
            if not image:
                raise ContainerError("No original image ID for service " + service)
            overlay[service] = {"image": image}
        contents = [*files.values(), yaml.safe_dump({"services": overlay})]
        with self._saved_compose_args(context, contents) as args:
            model = self._resolved_model(self.manager.runtime.create_docker_process(
                *args, *self.config_args(services=services, output_format="json"), capture_output=True))
            specifications = {service: model["services"][service] for service in services}
            for service in order_service_subset(context.project_containers, specifications):
                self.wait_service_dependencies(context, service, model=model)
                self.manager.runtime.create_docker_process(
                    *args, *self.apply_service_args(service, recreate=True)).check_call()

    def compose_args(self, files: "Sequence[str]") -> "list[str]":
        """Shared project/file argument construction for planning and execution."""
        from ..errors import ContainerError
        if not files:
            raise ContainerError("No Compose files in selected project")
        args = ["compose", "--project-directory", os.path.join(str(self.manager.data_path), "compose"),
                "--project-name", self.manager.project_name]
        for path in files:
            args.extend(["--file", str(path)])
        return args

    @contextmanager
    def _model_args(self, context: "OperationContext",
                    model: "dict[str, Any] | None" = None) -> "Iterator[list[str]]":
        import yaml
        if model is None:
            model = context.compose_model if context.compose_model is not None else self.final_model(context)
        with self._saved_compose_args(context, (yaml.safe_dump(model, sort_keys=True),)) as args:
            yield args

    def wait_service_ready(self, context: "OperationContext", service: str,
                           model: "dict[str, Any] | None" = None, timeout: "int | None" = None) -> bool:
        """Confirm readiness; return whether any instance remains running."""
        import time
        from ..errors import ContainerError
        if model is None:
            model = context.compose_model
        if self._is_scaled_to_zero(model["services"][service]):
            return False
        completed = any(
            service_dependencies(spec).get(service, {}).get("condition") == "service_completed_successfully"
            for name, spec in model["services"].items()
            if context.target_services is None or name in context.target_services)
        if completed:
            self.wait_service_completed(context, service, timeout=timeout)
            return False
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            actual = self.manager.docker_inspector.get_project_state(context.project_containers)
            matches = [item for item in actual.services if item.service == service]
            if matches and all(
                    (item.state == "running" and item.health in (None, "healthy")) or
                    (item.state == "exited" and item.exit_code == 0 and item.health is None)
                    for item in matches):
                return any(item.state == "running" for item in matches)
            if any(item.state in ("dead", "exited") or item.health == "unhealthy" for item in matches):
                raise ContainerError("Service {} failed to become ready".format(service))
            if timeout is None and (not matches or any(
                    item.state not in ("running", "restarting", "exited") for item in matches)):
                raise ContainerError("Service {} is unavailable".format(service))
            if deadline is not None and time.monotonic() >= deadline:
                raise ContainerError("Service {} did not become ready".format(service))
            time.sleep(0.5)
