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
    from collections.abc import Sequence
    from typing import Any
    from ..context import EventContext
    from ..manager import ContainerManager


_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


@dataclass
class ComposeOptions:
    """Resolved options for a single compose build/up invocation."""

    pull: bool = False
    remove_orphans: bool = False
    services: "list[str]" = field(default_factory=list)
    # Compatibility field for callers constructing options; preparation owns
    # all pull decisions now.
    emit_default_pull: bool = False
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
        return ComposeOptions(pull=pull, services=list(services),
                              emit_default_pull=False)

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
