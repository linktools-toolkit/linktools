#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr compose up/restart/down`` -- identical to the root ``ct-cntr
up/restart/down`` shortcuts; both dispatch through ComposeOperations."""
from linktools.cli import subcommand, subcommand_argument
from linktools.cli.argparse import BooleanOptionalAction, LazyChoices
from .. import _shared


class ComposeLifecycleCommands:
    """Mixin providing the compose lifecycle subcommands."""

    @subcommand("up", help="deploy installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_up(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        return _shared.manager.compose_operations.up(names=names, build=build, pull=pull)

    @subcommand("restart", help="restart installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_restart(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        return _shared.manager.compose_operations.restart(names=names, build=build, pull=pull)

    @subcommand("down", help="stop installed containers")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_down(self, names: "list[str]" = None):
        return _shared.manager.compose_operations.down(names=names)
