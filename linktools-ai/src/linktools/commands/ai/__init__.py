#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai` command package.

This package is discovered by `linktools.cli.command.iter_module_commands` as a
single command node (not a group) because it re-exports a ``command`` attribute
that is a `BaseCommand` instance. All subcommands live in `.chat`.
"""

from .chat import Command

command = Command()
