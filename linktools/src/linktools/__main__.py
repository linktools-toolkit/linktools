#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

from linktools.cli import BaseCommandGroup, iter_entry_point_commands
from linktools.metadata import __scripts_group__

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommandGroup):

    """Top-level linktools command group."""
    def init_arguments(self, parser: "CommandParser") -> None:
        """Register top-level entry point commands.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
        """
        self.add_subcommands(
            parser=parser,
            target=iter_entry_point_commands(__scripts_group__, onerror="warn"),
            sort=True
        )


command = Command()
if __name__ == '__main__':
    raise SystemExit(command.main())
