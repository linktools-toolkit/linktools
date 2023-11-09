#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from typing import Optional

from linktools.cli import IOSCommand


class Command(IOSCommand):
    """
    Sib supports managing multiple ios devices
    """

    _GENERAL_COMMANDS = [
        "completion",
        "devices",
        "help",
        "version",
        "remote",
    ]

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('sib_args', nargs='...', metavar="args", help="sib args")

    def run(self, args: Namespace) -> Optional[int]:
        if args.sib_args and args.sib_args[0] not in self._GENERAL_COMMANDS:
            device = args.parse_device()
            process = device.popen(*args.sib_args, capture_output=False)
            return process.call()

        process = args.parse_device.bridge.popen(*args.sib_args, capture_output=False)
        return process.call()


command = Command()
if __name__ == "__main__":
    command.main()
