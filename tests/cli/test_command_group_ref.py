#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public CommandGroupRef registration semantics."""

import pytest

from linktools.cli import (
    BaseCommand,
    CommandGroupRef,
    CommandParser,
    SubCommandError,
    SubCommandGroup,
    SubCommandWrapper,
)


class _Root(BaseCommand):
    def init_arguments(self, parser: CommandParser) -> None:
        del parser

    def run(self, args: object) -> int:
        del args
        return 0


def _child(name: str, parent: object) -> SubCommandWrapper:
    class _Command(BaseCommand):
        @property
        def name(self) -> str:
            return name

        @property
        def parent(self) -> object:
            return parent

        def init_arguments(self, parser: CommandParser) -> None:
            del parser

        def run(self, args: object) -> int:
            del args
            return 0

    return SubCommandWrapper(_Command())


def _parser(root: _Root) -> CommandParser:
    return CommandParser(prog="test", command=root)


def test_missing_declared_parent_group_is_registered() -> None:
    root = _Root()
    parser = _parser(root)
    child = _child(
        "cntr",
        CommandGroupRef(
            id="common",
            name="ct",
            description="Common scripts",
            order="100-common",
        ),
    )

    root.add_subcommands(parser=parser, target=[child], required=True)

    parser.parse_args(["ct", "cntr"])


def test_real_group_wins_over_fallback_declaration() -> None:
    root = _Root()
    parser = _parser(root)
    real = SubCommandGroup(
        name="ct",
        description="Common scripts",
        id="common",
        order="100-common",
    )
    child = _child(
        "cntr",
        CommandGroupRef(id="common", name="other", order="999-other"),
    )

    root.add_subcommands(parser=parser, target=[real, child], required=True)

    parser.parse_args(["ct", "cntr"])


def test_consistent_fallback_declarations_share_one_group() -> None:
    root = _Root()
    parser = _parser(root)
    parent = CommandGroupRef(id="common", name="ct", order="100-common")

    root.add_subcommands(
        parser=parser,
        target=[_child("cntr", parent), _child("env", parent)],
        required=True,
    )

    parser.parse_args(["ct", "cntr"])
    parser.parse_args(["ct", "env"])


def test_conflicting_fallback_declarations_are_rejected() -> None:
    root = _Root()
    parser = _parser(root)

    with pytest.raises(SubCommandError):
        root.add_subcommands(
            parser=parser,
            target=[
                _child(
                    "cntr",
                    CommandGroupRef(id="common", name="ct", order="100-common"),
                ),
                _child(
                    "env",
                    CommandGroupRef(id="common", name="common", order="100-common"),
                ),
            ],
            required=True,
        )


def test_missing_plain_string_parent_is_rejected() -> None:
    root = _Root()
    parser = _parser(root)

    with pytest.raises(SubCommandError):
        root.add_subcommands(
            parser=parser,
            target=[_child("cntr", "common")],
            required=True,
        )
