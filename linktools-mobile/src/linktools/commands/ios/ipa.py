#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.mobile.ios import IPA

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser



class Command(BaseCommand):
    """
    Parse and extract detailed information from IPA files
    """

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("path", help="ipa file path")

    def run(self, args: "Namespace") -> "int | None":
        path = os.path.abspath(os.path.expanduser(args.path))
        ipa = IPA(path)
        self.logger.info(
            json.dumps(
                ipa.get_info_plist(),
                indent=2,
                ensure_ascii=False
            )
        )
        return 0


command = Command()
if __name__ == "__main__":
    command.main()
