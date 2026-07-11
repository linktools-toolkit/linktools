#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import SubprocessError
from typing import TYPE_CHECKING

from dulwich.errors import GitProtocolError

from linktools.cli import (
    BaseCommandGroup, CommandGroupRef, CommandParser, SubCommandWrapper, subcommand, subcommand_argument,
)
from linktools.cli.argparse import BooleanOptionalAction, LazyChoices
from linktools.errors import ConfigError, GitError
from ..container import ContainerError
from ..doctor import WARN, Doctor
from . import _shared
from ._order import ROOT_COMMAND_ORDER
from .compose import ComposeCommand
from .config import ConfigCommand
from .exec_ import ExecCommand
from .plan import PlanCommand, maybe_dry_run
from .repo import RepoCommand
from .status import StatusCommands

if TYPE_CHECKING:
    from typing import Any


class Command(StatusCommands, BaseCommandGroup):
    """
    Deploy and manage Docker containers with ease
    """

    @property
    def name(self) -> str:
        return "cntr"

    @property
    def parent(self) -> "CommandGroupRef | str | None":
        return CommandGroupRef(
            id="common",
            name="ct",
            description="Common scripts",
        )

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ContainerError, ConfigError, GitError, SubprocessError, GitProtocolError, OSError, AssertionError,
        ]

    def init_arguments(self, parser: "CommandParser") -> None:
        self.add_subcommands(parser=parser, target=self.init_subcommands(), sort=True)

    def init_subcommands(self) -> "Any":
        return [
            self,
            SubCommandWrapper(ExecCommand(), order=ROOT_COMMAND_ORDER["exec"]),
            SubCommandWrapper(ComposeCommand(), order=ROOT_COMMAND_ORDER["compose"]),
            SubCommandWrapper(PlanCommand(), order=ROOT_COMMAND_ORDER["plan"]),
            SubCommandWrapper(ConfigCommand(), order=ROOT_COMMAND_ORDER["config"]),
            SubCommandWrapper(RepoCommand(), order=ROOT_COMMAND_ORDER["repo"]),
        ]

    @subcommand("list", order=ROOT_COMMAND_ORDER["list"], help="list all containers")
    @subcommand_argument("--detail", action="store_true", help="show container detail info")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_list(self, names: "list[str]" = None, detail: bool = False):
        manager = _shared.manager
        # Registers every resolved-installed container's own config defaults
        # before anything below can indirectly render a container's own
        # compose/Dockerfile template (get_effective -> get_actual ->
        # DockerInspector iterates container.services, a cached_property
        # that renders docker_compose) -- otherwise a template referencing
        # its own container's config key sees it as genuinely undefined.
        manager.prepare_installed_containers()
        install_containers = manager.installed_state.get(resolve=False)
        all_install_containers = manager.resolver.resolve_dependencies(install_containers)
        # Prefer live state, fall back to persisted when Docker is unavailable
        # so `list` never crashes.
        running_names = set(manager.running_state.get_effective(install_containers))
        for container in sorted(manager.containers.values(), key=lambda o: o.order):
            if names and container.name not in names:
                continue
            installed = container in all_install_containers
            added = container in install_containers
            running = container.name in running_names
            if not installed and running:
                style, symbol, label = "yellow bold", "[-]", "pending remove"
            elif not installed:
                style, symbol, label = "dim", "[ ]", None
            elif added and running:
                style, symbol, label = "green bold", "[*]", "added"
            elif added:
                style, symbol, label = "cyan bold", "[+]", "pending install"
            elif running:
                style, symbol, label = "green dim", "[-]", "dependency"
            else:
                style, symbol, label = "cyan dim", "[+]", "pending install, dependency"
            suffix = f" \\[{label}]" if label else ""
            message = f"[{style}]{symbol} {container.name}{suffix}[/]"
            if detail:
                message += f"{os.linesep}    [dim]Enable: {container.enable}[/]"
                message += f"{os.linesep}    [dim]Order: {container.order}[/]"
                message += f"{os.linesep}    [dim]Path: {container.root_path}[/]"
                message += f"{os.linesep}    [dim]Description: {container.description}[/]"
                message += f"{os.linesep}    [dim]Dependencies: \\[{', '.join(container.dependencies)}][/]"
                message += f"{os.linesep}    [dim]Configs: \\[{', '.join(container.configs.keys())}][/]"
            self.logger.info(message, extra={"markup": True})

    @subcommand("add", order=ROOT_COMMAND_ORDER["add"], help="add containers to installed list")
    @subcommand_argument("names", metavar="CONTAINER", nargs="+", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_add(self, names: "list[str]"):
        containers = _shared.manager.installed_state.add(*names)
        assert containers, "No container added"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Add {', '.join(result)} success")

    @subcommand("remove", order=ROOT_COMMAND_ORDER["remove"], help="remove containers from installed list")
    @subcommand_argument("-f", "--force", help="Force remove")
    @subcommand_argument("names", metavar="CONTAINER", nargs="+", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_remove(self, names: "list[str]", force: bool = False):
        containers = _shared.manager.installed_state.remove(*names, force=force)
        assert containers, "No container removed"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Remove {', '.join(result)} success")

    @subcommand("up", order=ROOT_COMMAND_ORDER["up"], help="deploy installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                         help="show what would happen, without doing it")
    @subcommand_argument("--report", action="store_true", default=False,
                         help="show a per-phase timing/outcome report after completion")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_up(self, names: "list[str]" = None, build: bool = True, pull: str = False,
                      dry_run: bool = False, report: bool = False):
        if maybe_dry_run(_shared.manager, self.logger, "up", names=names, build=build, pull=pull, dry_run=dry_run):
            return
        # Root `up` and `compose up` share one implementation (ComposeOperations)
        # so they cannot drift from each other.
        _shared.manager.compose_operations.up(names=names, build=build, pull=pull, report=report)

    @subcommand("restart", order=ROOT_COMMAND_ORDER["restart"], help="restart installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                         help="show what would happen, without doing it")
    @subcommand_argument("--report", action="store_true", default=False,
                         help="show a per-phase timing/outcome report after completion")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_restart(self, names: "list[str]" = None, build: bool = True, pull: str = False,
                           dry_run: bool = False, report: bool = False):
        if maybe_dry_run(_shared.manager, self.logger, "restart", names=names, build=build, pull=pull,
                         dry_run=dry_run):
            return
        _shared.manager.compose_operations.restart(names=names, build=build, pull=pull, report=report)

    @subcommand("down", order=ROOT_COMMAND_ORDER["down"], help="stop installed containers")
    @subcommand_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                         help="show what would happen, without doing it")
    @subcommand_argument("--report", action="store_true", default=False,
                         help="show a per-phase timing/outcome report after completion")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_down(self, names: "list[str]" = None, dry_run: bool = False, report: bool = False):
        if maybe_dry_run(_shared.manager, self.logger, "down", names=names, dry_run=dry_run):
            return
        _shared.manager.compose_operations.down(names=names, report=report)

    @subcommand("doctor", order=ROOT_COMMAND_ORDER["doctor"],
               help="read-only environment and security checks (changes nothing)")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("--check", action="store_true", default=False,
                         help="exit non-zero if any WARN-or-worse finding is present")
    @subcommand_argument("--runtime", action="store_true", default=False,
                         help="also validate compose config and manifest-declared "
                              "docker-engine/docker-compose requirements")
    @subcommand_argument("--sudo-prompt", dest="sudo_prompt", action="store_true", default=False,
                         help="allow an interactive sudo password prompt (default: never blocks on one)")
    def on_command_doctor(self, as_json: bool = False, check: bool = False, runtime: bool = False,
                          sudo_prompt: bool = False):
        findings = Doctor(_shared.manager).run(runtime=runtime, sudo_prompt=sudo_prompt)
        if as_json:
            import json
            payload = dict(
                schema_version=1,
                project=_shared.manager.project_name,
                findings=[
                    dict(severity=f.severity, code=f.code, component=f.component,
                        message=f.message, details=f.details)
                    for f in findings
                ],
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for finding in findings:
                self.logger.info(f"[{finding.severity}] {finding.message}")
            self.logger.info("[INFO] No change has been made. Suggestions are kept for compatibility.")
        # Default `doctor` exit behavior is unchanged; only --check turns
        # findings into a non-zero exit.
        if check and any(f.severity == WARN for f in findings):
            raise ContainerError("Doctor found WARN-or-worse issues")

    def _make_context(self, commands, names):
        # Kept as a thin wrapper over the manager facade so external
        # subclasses/tests calling Command._make_context keep working.
        return _shared.manager.create_event_context(commands, names=names)
