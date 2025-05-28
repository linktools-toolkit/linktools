#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from argparse import ArgumentParser, Namespace
from typing import Optional

from linktools.cli import BaseCommand
from linktools.mobile.ios import IPA


class Command(BaseCommand):
    """
    Parse and extract detailed information from IPA files
    """

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("path", help="ipa file path")

    def run(self, args: Namespace) -> Optional[int]:
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
