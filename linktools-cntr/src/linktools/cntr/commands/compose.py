#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr compose [CONTAINER...]`` -- the final resolved Docker Compose
model for the installed project, or ``--check`` to only validate it.

``--file`` always uses the complete installed project; an explicit
CONTAINER selection only narrows the trailing SERVICE filter passed to
``docker compose ... config``, it never narrows which compose files are
loaded (Compose merge/interpolation must see the whole project).
"""
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandParser
from linktools.cli.argparse import LazyChoices
from ..container import ContainerError
from . import _shared

if TYPE_CHECKING:
    from argparse import Namespace


class ComposeCommand(BaseCommand):
    """
    show the final resolved Docker Compose model (or --check it)
    """

    @property
    def name(self) -> str:
        return "compose"

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                            choices=LazyChoices(_shared.iter_installed_container_names))
        parser.add_argument("-d", "--with-dependencies", dest="with_dependencies",
                            action="store_true", default=False,
                            help="include dependency containers in the selection")
        parser.add_argument("--format", dest="output_format", choices=("yaml", "json"),
                            help="output format (default: yaml, docker compose's own default)")
        parser.add_argument("--check", action="store_true", default=False,
                            help="only validate the resolved model, do not print it")

    def run(self, args: "Namespace") -> "int | None":
        if args.check and args.output_format:
            raise ContainerError("--check and --format cannot be used together")

        return _shared.manager.compose_operations.render(
            names=args.names,
            with_dependencies=args.with_dependencies,
            output_format=args.output_format,
            check=args.check,
        )
