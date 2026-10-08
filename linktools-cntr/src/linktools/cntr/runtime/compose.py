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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence, Mapping
    from typing import Any, Iterable
    from ..container import BaseContainer
    from ..context import EventContext
    from ..artifacts import GeneratedCandidate
    from .structured import CommandResult
    from ..manager import ContainerManager


_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


def service_dependencies(spec: "dict[str, Any]") -> "dict[str, dict[str, Any]]":
    """Normalize Compose's explicit and implicit service dependency edges."""
    declared = spec.get("depends_on") or {}
    if isinstance(declared, dict):
        result = {name: dict(options) if isinstance(options, dict) else {}
                  for name, options in declared.items()}
    else:
        result = {name: {} for name in declared}
    implicit = [value.split(":", 1)[0] for value in spec.get("links") or ()]
    implicit.extend(value.split(":", 1)[0] for value in spec.get("volumes_from") or ()
                    if not value.startswith("container:"))
    for key in ("network_mode", "ipc", "pid"):
        value = spec.get(key) or ""
        if value.startswith("service:"):
            implicit.append(value[len("service:"):])
    for name in implicit:
        result.setdefault(name, {})
    return result


def order_services(containers: "Iterable[BaseContainer]", services: "Iterable[str]",
                   model: "dict[str, Any] | None" = None,
                   available_services: "Iterable[str]" = ()) -> "tuple[str, ...]":
    """Topologically order applications; priority only breaks ready-node ties.

    Availability is reserved for acknowledged bootstrap services, never the
    general running set: ordinary dependencies must apply their new model first.
    """
    from ..container import ContainerError
    containers = tuple(containers)
    installed = {container.name: container for container in containers}
    owners = {name: container for container in containers for name in container.services}
    selected = tuple(dict.fromkeys(services))
    pending = set(selected)
    available = set(available_services)
    definitions = model["services"] if model is not None else {
        name: owner.services[name] for name, owner in owners.items()}
    required = {owners[name].name for name in selected}
    dependencies = {}
    for name in selected:
        owner = owners[name]
        edges = set()
        for dependency in owner.dependencies:
            edges.update(service for service in installed[dependency].services
                         if service in pending and service not in available)
        for dependency, options in service_dependencies(definitions[name]).items():
            condition = options.get("condition", "service_started")
            if dependency in available and condition in ("service_started", "service_healthy"):
                continue
            if dependency not in pending:
                raise ContainerError("Unselected Compose dependency {} for {}".format(dependency, name))
            edges.add(dependency)
        for provider, provider_services in owner.get_runtime_requirements(required).items():
            if provider != owner.name:
                edges.update(service for service in provider_services if service in pending)
        dependencies[name] = edges
    result = []
    positions = {name: index for index, name in enumerate(selected)}
    while pending:
        ready = [name for name in pending if not (dependencies[name] & pending)]
        if not ready:
            raise ContainerError("Compose dependency cycle at " + ", ".join(sorted(pending)))
        name = min(ready, key=lambda value: (owners[value].application_priority, positions[value]))
        pending.remove(name)
        result.append(name)
    return tuple(result)


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

    def collect_services(self, context: "EventContext") -> "list[str]":
        """Service names for the targeted containers; empty for "all" runs."""
        if context.is_full_containers:
            return []
        services: "list[str]" = []
        for container in context.target_containers:
            services.extend(container.services.keys())
        if not services:
            # Imported lazily to keep runtime.compose free of a module-level
            # dependency on ..container (which imports this module).
            from ..container import ContainerError
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

    def build(self, context: "EventContext", options: ComposeOptions) -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers, *self.build_args(options)
        ).check_call()

    def pull_args(self, services: "Sequence[str]") -> "list[str]":
        return ["pull", "--ignore-buildable", *services]

    def pull(self, context: "EventContext", services: "Sequence[str]") -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers, *self.pull_args(services)
        ).check_call()

    def options_for_build(self, services: "Sequence[str]", pull: bool = False) -> ComposeOptions:
        return ComposeOptions(pull=pull, services=list(services))

    def final_model(self, context: "EventContext") -> "dict[str, Any]":
        result = self.manager.structured_runner.execute_json(
            self.manager.runtime.create_docker_compose_process(
                context.containers, *self.config_args(output_format="json"),
                capture_output=True,
            ), check=True,
        )
        if not isinstance(result, dict) or not isinstance(result.get("services"), dict):
            from ..container import ContainerError
            raise ContainerError("Docker Compose returned an invalid final model")
        return result

    def up(self, context: "EventContext", options: ComposeOptions) -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers, *self.up_args(options)
        ).check_call()

    def stop(self, context: "EventContext", services: "Sequence[str]") -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers, "stop", *services
        ).check_call()

    def down(self, context: "EventContext", services: "Sequence[str]") -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers, "down", *services
        ).check_call()

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

    def config(
            self,
            context: "EventContext",
            services: "Sequence[str]" = (),
            output_format: "str | None" = None,
            quiet: bool = False,
    ) -> int:
        return self.manager.runtime.create_docker_compose_process(
            context.containers,
            *self.config_args(services=services, output_format=output_format, quiet=quiet),
            privilege=False,
        ).check_call()

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
                value = "type=bind,source={},target={},readonly".format(overrides.pop(target), target)
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
                    from ..container import ContainerError
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

    def validate_service(self, context: "EventContext", service: str,
                         command: "Sequence[str]", environment: "Mapping[str, object] | None" = None,
                         network: bool = False, check: bool = True,
                         mount_overrides: "Mapping[str, str] | None" = None) -> "CommandResult":
        model = getattr(context, "compose_model", None)
        if model is None:
            model = self.final_model(context)
        args = self.isolated_service_args(model, service, command, environment, network, mount_overrides)
        result = self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_process(*args, capture_output=True), check=False)
        if check and not result.succeeded:
            from ..container import ContainerError
            # Command output can contain expanded credentials.
            raise ContainerError("Native validation failed for service {} (exit {})".format(
                service, result.returncode))
        return result

    def run_isolated_service(self, context: "EventContext", service: str,
                             command: "Sequence[str]") -> None:
        """Run a one-shot service maintenance command with resolved mounts, no network or ports."""
        model = getattr(context, "compose_model", None) or self.final_model(context)
        args = self.isolated_service_args(model, service, command)
        result = self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_process(*args, capture_output=True), check=False)
        if not result.succeeded:
            from ..container import ContainerError
            raise ContainerError("Isolated service command failed for {} (exit {})".format(
                service, result.returncode))

    def apply_service_args(self, service: str, recreate: bool = False, remove_orphans: bool = False) -> "list[str]":
        args = self.up_args(ComposeOptions(services=[], remove_orphans=remove_orphans))
        args.append("--no-deps")
        if recreate:
            args.append("--force-recreate")
        args.append(service)
        return args

    def apply_service(self, context: "EventContext", service: str, recreate: bool = False) -> int:
        previous = getattr(context, "rollback_service_models", {}).get(service)
        if previous is not None:
            self.apply_saved_services(context, (service,), {"previous.yml": previous})
            return 0
        saved = getattr(context, "rollback_compose_files", None)
        if saved is not None:
            self.apply_saved_services(context, (service,), saved)
            return 0
        self.wait_service_dependencies(context, service)
        args = self.apply_service_args(service, recreate, context.is_full_containers)
        import tempfile
        import yaml
        candidates = getattr(context, "generated_candidates", {})
        candidate = next((c for c in candidates.values() if service == c.container.name), None)
        label = candidate.container.generation_label(service, candidate.generation_id) if candidate else None
        if label is None:
            return self.manager.runtime.create_docker_compose_process(context.containers, *args).check_call()
        overlay = {"services": {service: {"labels": {
            "io.linktools.cntr.generation": label}}}}
        with tempfile.TemporaryDirectory(prefix="cntr-apply-") as directory:
            path = os.path.join(directory, "generation.yml")
            with open(path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(overlay, stream)
            return self.manager.runtime.create_docker_compose_process(
                context.containers, "--file", path, *args).check_call()

    def is_generation_current(self, context: "EventContext", service: str, candidate: "GeneratedCandidate") -> bool:
        if (service in getattr(context, "changed_image_services", ()) or
                service in getattr(context, "changed_compose_services", ())):
            return False
        state = self.manager.docker_inspector.get_project_state(context.containers)
        matches = [item for item in state.services if item.service == service]
        if not matches or not all(item.state == "running" and
                item.labels.get("io.linktools.cntr.generation") == candidate.generation_id
                for item in matches):
            return False
        model = getattr(context, "compose_model", None)
        if model is None:
            model = self.final_model(context)
        spec = model["services"][service]
        target = spec.get("image") or (self.manager.project_name + "-" + service)
        result = self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_process(
                "image", "inspect", "--format", "{{.Id}}", target, capture_output=True), check=True)
        target_id = result.stdout.strip()
        return bool(target_id) and all(item.image_id == target_id for item in matches)

    def wait_service_running(self, context: "EventContext", service: str, timeout: int = 30) -> None:
        import time
        from ..container import ContainerError
        deadline = time.monotonic() + timeout
        while True:
            state = self.manager.docker_inspector.get_project_state(context.containers)
            matches = [item for item in state.services if item.service == service]
            if matches and all(item.state == "running" for item in matches):
                return
            if time.monotonic() >= deadline:
                raise ContainerError("Service {} did not become running".format(service))
            time.sleep(0.5)

    def exec_service(self, context: "EventContext", service: str, command: "Sequence[str]",
                     check: bool = True) -> "CommandResult":
        return self.manager.structured_runner.execute(
            self.manager.runtime.create_docker_compose_process(
                context.containers, "exec", "-T", service, *command, capture_output=True), check=check)

    def wait_service_healthy(self, context: "EventContext", service: str, timeout: int = 30) -> None:
        import time
        from ..container import ContainerError
        deadline = time.monotonic() + timeout
        while True:
            state = self.manager.docker_inspector.get_project_state(context.containers)
            matches = [item for item in state.services if item.service == service]
            if matches and all(item.state == "running" and item.health == "healthy" for item in matches):
                return
            if time.monotonic() >= deadline:
                raise ContainerError("Service {} did not become healthy".format(service))
            time.sleep(0.5)

    def wait_service_dependencies(self, context: "EventContext", service: str) -> None:
        """Honor Compose readiness conditions for every native apply path."""
        from ..container import ContainerError
        model = getattr(context, "compose_model", None) or self.final_model(context)
        for dependency, options in service_dependencies(model["services"][service]).items():
            condition = options.get("condition", "service_started")
            if condition == "service_healthy":
                self.wait_service_healthy(context, dependency)
            elif condition == "service_completed_successfully":
                self.wait_service_completed(context, dependency)
            elif condition == "service_started":
                # The ordered dependency's successful `up -d` acknowledged
                # startup; one-shot services may already have exited by now.
                continue
            else:
                raise ContainerError("Unsupported Compose dependency condition: " + str(condition))

    def wait_service_completed(self, context: "EventContext", service: str, timeout: int = 30) -> None:
        import time
        from ..container import ContainerError
        deadline = time.monotonic() + timeout
        while True:
            state = self.manager.docker_inspector.get_project_state(context.containers)
            matches = [item for item in state.services if item.service == service]
            if any(item.state in ("exited", "dead") and item.exit_code != 0 for item in matches):
                raise ContainerError("Dependency service {} failed".format(service))
            if matches and all(item.state == "exited" and item.exit_code == 0 for item in matches):
                return
            if time.monotonic() >= deadline:
                raise ContainerError("Service {} did not complete successfully".format(service))
            time.sleep(0.5)

    def apply_services(self, context: "EventContext", services: "Sequence[str]") -> None:
        """Apply the dependency-ordered selection supplied by the orchestrator."""
        for service in services:
            self.apply_service(context, service)

    def apply_saved_services(self, context: "EventContext", services: "Sequence[str]",
                             files: "dict[str, str]") -> None:
        """Apply the saved model through the same command builder on rollback."""
        import tempfile
        with tempfile.TemporaryDirectory(prefix="cntr-rollback-") as directory:
            file_args = []
            for index, content in enumerate(files.values()):
                path = os.path.join(directory, "{}.yml".format(index))
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(content)
                file_args.extend(["--file", path])
            import yaml
            candidates = getattr(context, "generated_candidates", {})
            labels = {
                candidate.container.name: {"labels": {
                    "io.linktools.cntr.generation": candidate.generation_id}}
                for candidate in candidates.values()
                if candidate.container.name in services and
                candidate.container.generation_label(candidate.container.name, candidate.generation_id) is not None
            }
            if labels:
                path = os.path.join(directory, "generation.yml")
                with open(path, "w", encoding="utf-8") as stream:
                    yaml.safe_dump({"services": labels}, stream)
                file_args.extend(["--file", path])
            for service in services:
                self.manager.runtime.create_docker_process(
                    "compose", *file_args, "--project-name", self.manager.project_name,
                    *self.apply_service_args(service, recreate=True)).check_call()
