#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Docker Compose command assembly.

Both the CLI (``ct-cntr up/restart/down``) and the per-container ``exec``
subcommands build the same kind of ``docker compose`` argument lists. This
module centralizes that assembly so the two paths cannot drift.

Two behavioral subtleties, each captured by a ``ComposeOptions`` flag so every
caller reproduces its own exact arguments:

- when ``--pull`` is not requested, CLI ``up`` still emits ``--pull=false``
  (to build) and ``--pull missing`` (to up); ``restart`` and ``exec`` emit
  nothing (``emit_default_pull``).
- CLI ``up`` and both ``exec up``/``exec restart`` include proxy
  ``--build-arg``s; CLI ``restart`` never did (``include_proxy_build_args``).
"""
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from ..context import EventContext
    from ..manager import ContainerManager


_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


@dataclass
class ComposeOptions:
    """Resolved options for a single compose build/up invocation."""

    build: bool = True
    pull: bool = False
    remove_orphans: bool = False
    services: "list[str]" = field(default_factory=list)
    # Only CLI ``up`` emits --pull=false / --pull missing when ``pull`` is False.
    # ``restart`` and ``exec`` emit no pull flags in that case.
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
        elif options.emit_default_pull:
            args.append("--pull=false")
        if options.include_proxy_build_args:
            args.extend(self.collect_proxy_build_args())
        args.extend(options.services)
        return args

    def up_args(self, options: ComposeOptions) -> "list[str]":
        args: "list[str]" = ["up", "--detach", "--no-build"]
        if options.pull:
            args.extend(["--pull", "always"])
        elif options.emit_default_pull:
            args.extend(["--pull", "missing"])
        if options.remove_orphans:
            args.append("--remove-orphans")
        args.extend(options.services)
        return args

    def build(self, context: "EventContext", options: ComposeOptions):
        return self.manager.create_docker_compose_process(
            context.containers, *self.build_args(options)
        ).check_call()

    def up(self, context: "EventContext", options: ComposeOptions):
        return self.manager.create_docker_compose_process(
            context.containers, *self.up_args(options)
        ).check_call()

    def stop(self, context: "EventContext", services: "Sequence[str]"):
        return self.manager.create_docker_compose_process(
            context.containers, "stop", *services
        ).check_call()

    def down(self, context: "EventContext", services: "Sequence[str]"):
        return self.manager.create_docker_compose_process(
            context.containers, "down", *services
        ).check_call()

    def config(self, context: "EventContext", services: "Sequence[str]" = ()):
        return self.manager.create_docker_compose_process(
            context.containers, "config", *services, privilege=False
        ).check_call()
