#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest
from argparse import Namespace
from typing import Optional

from linktools.cli import BaseCommand, subcommand, subcommand_argument, commands, SubCommandWrapper, CommandParser


class TestCommands(unittest.TestCase):

    def test_help(self):
        class Command(BaseCommand):

            def __init__(self):
                self.subcommands = list(self.walk_subcommands(commands))

            def init_arguments(self, parser: CommandParser) -> None:
                self.add_subcommands(parser, self.subcommands)

            def run(self, args: Namespace) -> Optional[int]:
                self.run_subcommand(args)
                return 0

        command = Command()
        with self.subTest(command.name, command=command):
            self.assertEqual(command(["--help"]), 0)

        for subcommand in command.subcommands:
            if isinstance(subcommand, SubCommandWrapper):
                with self.subTest(subcommand.name, command=subcommand.command):
                    self.assertEqual(subcommand.command(["--help"]), 0)

    def test_sub_command(self):
        class SubCommand(BaseCommand):

            def init_arguments(self, parser):
                self.add_subcommands(parser)

            def run(self, args):
                self.run_subcommand(args)

            @subcommand("aaa", help="test subcommand")
            def aaa(self):
                print("SubCommand.aaa")

            @subcommand("bbb", help="test subcommand")
            def bbb(self):
                print("SubCommand.bbb")

            @subcommand("ccc", help="test subcommand")
            @subcommand_argument("-a", "--arg1")
            def ccc(self, arg1):
                print("SubCommand.ccc")

        class SubCommand2(SubCommand):

            @subcommand("ddd", help="test subcommand")
            def ddd(self):
                print("SubCommand2.ddd")

            @subcommand("aaa", help="test subcommand")
            @subcommand_argument("-a")
            def aaa(self, a: bool = True):
                print("SubCommand2.aaa")

            def ccc(self, arg1, arg2: str = 123):
                print("SubCommand2.ccc", arg1)

        command = SubCommand2()
        with self.subTest(command.name, command=command):
            self.assertEqual(command(["--help"]), 0)
            self.assertEqual(command(["-h"]), 0)
            self.assertEqual(command(["aaa"]), 0)
            self.assertEqual(command(["bbb"]), 0)
            self.assertEqual(command(["ccc", "--arg1", "test"]), 0)
            self.assertEqual(command(["ddd"]), 0)


if __name__ == '__main__':
    unittest.main()
