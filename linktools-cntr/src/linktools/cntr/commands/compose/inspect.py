#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr compose config/validate`` -- read-only Docker Compose inspection.

Both always use the full installed project's ``--file`` set; an explicit
CONTAINER selection only narrows the trailing SERVICE filter, it never
narrows which compose files are loaded (Compose merge/interpolation must see
the whole project)."""
from linktools.cli import subcommand, subcommand_argument
from linktools.cli.argparse import LazyChoices
from .. import _shared


class ComposeInspectCommands:
    """Mixin providing the compose inspection subcommands."""

    @subcommand("config", help="show docker compose config for installed containers")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    @subcommand_argument("-d", "--with-dependencies", action="store_true", default=False,
                         help="include dependency containers in the selection")
    @subcommand_argument("--format", dest="output_format", choices=["yaml", "json"],
                         help="output format (default: yaml, docker compose's own default)")
    def on_command_config(self, names: "list[str]" = None, with_dependencies: bool = False,
                          output_format: str = None):
        return _shared.manager.compose_operations.config(
            names=names, with_dependencies=with_dependencies, output_format=output_format,
        )

    @subcommand("validate", help="validate docker compose config for installed containers (read-only)")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    @subcommand_argument("-d", "--with-dependencies", action="store_true", default=False,
                         help="include dependency containers in the selection")
    def on_command_validate(self, names: "list[str]" = None, with_dependencies: bool = False):
        return _shared.manager.compose_operations.validate(names=names, with_dependencies=with_dependencies)
