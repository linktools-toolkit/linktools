#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_frida.py
@time    : 2018/11/25
@site    :
@software: PyCharm

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
import re
from argparse import ArgumentParser, Namespace
from typing import Optional, Type, List

from linktools import utils, DownloadError
from linktools.cli import CommandError, IOSCommand, CommandMain
from linktools.cli.argparse import KeyValueAction, range_type, BooleanOptionalAction
from linktools.frida import FridaApplication, FridaShareScript, FridaScriptFile, FridaEvalCode, FridaIOSServer


class Command(IOSCommand):
    """
    Use Frida for dynamic analysis on jailbroken iOS devices
    """

    @property
    def main(self) -> CommandMain:
        return CommandMain(self, show_log_level=True, show_log_time=True)

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [DownloadError]

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("-b", "--bundle-id", action="store", default=None,
                            help="target bundle id (default: frontmost application)")
        parser.add_argument("--spawn", action="store_true", default=False,
                            help="inject after spawn (default: false)")

        parser.add_argument("-P", "--parameters",
                            action=KeyValueAction, nargs="+", dest="user_parameters",
                            help="user script parameters")

        parser.add_argument("-l", "--load", metavar="SCRIPT",
                            action="append", dest="user_scripts", default=[],
                            type=lambda o: FridaScriptFile(o),
                            help="load user script")
        parser.add_argument("-e", "--eval", metavar="CODE", action="append", dest="user_scripts",
                            type=lambda o: FridaEvalCode(o),
                            help="evaluate code")
        parser.add_argument("-c", "--codeshare", metavar="URL", action="append", dest="user_scripts",
                            type=lambda o: FridaShareScript(o, cached=False),
                            help="load share script url")

        parser.add_argument("-a", "--auto-start", action="store_true", default=False,
                            help="automatically start when all processes exits")

        parser.add_argument("--local-port", metavar="PORT", action="store", dest="local_port",
                            type=range_type(1, 65536), default=None,
                            help="local frida port (default: unused port)")
        parser.add_argument("--remote-port", metavar="PORT", action="store", dest="remote_port",
                            type=range_type(1, 65536), default=27042,
                            help="remote frida port (default: 27042)")

        parser.add_argument("--spawn-gating", action=BooleanOptionalAction, default=True,
                            help="enable spawn gating (default: true)")
        parser.add_argument("--child-gating", action=BooleanOptionalAction, default=False,
                            help="enable child gating (default: false)")

    def run(self, args: Namespace) -> Optional[int]:

        logger = self.logger
        user_parameters = args.user_parameters
        user_scripts = args.user_scripts

        device = args.device_picker.pick()
        bundle_id = args.bundle_id

        class Application(FridaApplication):

            def on_session_detached(self, session, reason, crash) -> None:
                logger.info(f"{session} detached, reason={reason}")
                if reason in ("connection-terminated", "device-lost"):
                    self.signal_stop()
                elif len(self.sessions) == 0:
                    if args.auto_start:
                        app.spawn(bundle_id)

        server = FridaIOSServer(
            device=device,
            local_port=args.local_port or utils.get_free_port(),
            remote_port=args.remote_port,
        )

        with server:
            # 如果没有填包名，则找到顶层应用
            if utils.is_empty(bundle_id):
                target_app = server.get_frontmost_application()
                if target_app is None:
                    raise CommandError("Unknown frontmost application")
                bundle_id = target_app.identifier

            logger.info(f"Frida inject target application: {bundle_id}")

            app = Application(
                server,
                target_identifiers=rf"^{re.escape(bundle_id)}$",
                user_parameters=user_parameters,
                user_scripts=user_scripts,
                enable_spawn_gating=args.spawn_gating,
                enable_child_gating=args.child_gating
            )

            if args.spawn:
                # 打开进程后注入
                app.spawn(bundle_id)

            elif app.inject_all():
                # 注入所有进程进程
                pass

            elif args.auto_start:
                # 进程不存在，打开进程后注入
                app.spawn(bundle_id)

            return app.run()


command = Command()
if __name__ == "__main__":
    command.main()
