#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandParser, SubCommandGroup
from linktools.cli.argparse import ArgParseComplete, LazyChoices
from . import _shared

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import SubCommand


class ExecCommand(BaseCommand):
    """
    exec container command
    """

    @property
    def name(self):
        return "exec"

    @property
    def config(self):
        return _shared.manager.env_config

    @property
    def _subparser(self) -> "CommandParser":
        parser = CommandParser()

        subcommands: "list[SubCommand]" = []
        for container in _shared.manager.get_installed_containers():
            subcommand_group = SubCommandGroup(container.name, container.description)
            subcommands.append(subcommand_group)
            subcommands.extend(self.walk_subcommands(container, parent_id=subcommand_group.id))
        self.add_subcommands(parser, target=subcommands)

        return parser

    def init_arguments(self, parser: "CommandParser") -> None:

        class Completer(ArgParseComplete.Completer):
            get_parser = lambda _: self._subparser
            get_args = lambda _, args, **kw: \
                [args.exec_name, *args.exec_args] \
                if args.exec_name \
                else None

        parser.add_argument("exec_name", nargs="?", metavar="CONTAINER", help="container name",
                            choices=LazyChoices(_shared.iter_installed_container_names))
        action = parser.add_argument("exec_args", nargs="...", metavar="ARGS", help="container exec args")
        action.completer = Completer()

    def run(self, args: "Namespace") -> "int | None":
        args = self._subparser.parse_args([args.exec_name, *args.exec_args] if args.exec_name else [])
        subcommand = self.parse_subcommand(args)
        if not subcommand or isinstance(subcommand, SubCommandGroup):
            return self.print_subcommands(args, root=subcommand, max_level=2)
        _shared.manager.prepare_installed_containers()
        return subcommand.run(args)
