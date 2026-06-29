#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest

from linktools.cli import BaseCommand, subcommand, subcommand_argument, SubCommandWrapper, iter_entry_point_commands
from linktools.metadata import __scripts_group__
from linktools.__main__ import command


class TestCommands(unittest.TestCase):

    def test_help(self):
        # `--help` exits via argparse's SystemExit rather than returning, for
        # both the top-level command and every installed sub-package's CLI.
        with self.subTest(command.name, command=command):
            with self.assertRaises(SystemExit) as cm:
                command(["--help"])
            self.assertEqual(cm.exception.code, 0)

        for subcommand in command.walk_subcommands(iter_entry_point_commands(__scripts_group__, onerror="warn")):
            if isinstance(subcommand, SubCommandWrapper):
                with self.subTest(subcommand.name, command=subcommand.command):
                    with self.assertRaises(SystemExit) as cm:
                        subcommand.command(["--help"])
                    self.assertEqual(cm.exception.code, 0)

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

        command_ = SubCommand2()
        with self.subTest(command_.name, command=command_):
            with self.assertRaises(SystemExit) as cm:
                command_(["--help"])
            self.assertEqual(cm.exception.code, 0)

            with self.assertRaises(SystemExit) as cm:
                command_(["-h"])
            self.assertEqual(cm.exception.code, 0)

            self.assertEqual(command_(["aaa"]), 0)
            self.assertEqual(command_(["bbb"]), 0)
            self.assertEqual(command_(["ccc", "--arg1", "test"]), 0)
            self.assertEqual(command_(["ddd"]), 0)


if __name__ == '__main__':
    unittest.main()
