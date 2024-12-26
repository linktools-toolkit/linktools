#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : objection.py 
@time    : 2022/11/20
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
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Type

from linktools import utils, environ, DownloadError
from linktools.cli import CommandError, AndroidCommand
from linktools.cli.argparse import range_type, BooleanOptionalAction
from linktools.frida import FridaAndroidServer


class Command(AndroidCommand):
    """
    Simplify security testing with Objection on rooted Android devices
    """

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [DownloadError]

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("-p", "--package", action="store", default=None,
                            help="target package (default: frontmost application)")
        parser.add_argument("-s", "--startup-command", action="append", default=[],
                            help="A command to run before the repl polls the device for information.")
        parser.add_argument("-c", "--file-commands", action="store",
                            help="A file containing objection commands, separated by a "
                                 "newline, that will run before the repl polls the device for information.")
        parser.add_argument("-S", "--startup-script", action="store",
                            help="A script to import and run before the repl polls the device for information.")
        parser.add_argument("-P", "--plugin-folder", action="store", default=environ.get_asset_path("objection"),
                            help="The folder to load plugins from.")

        parser.add_argument("--serve", action=BooleanOptionalAction, default=True,
                            help="serve frida server (default: true)")
        parser.add_argument("--local-port", metavar="PORT", action="store", dest="local_port",
                            type=range_type(1, 65536), default=None,
                            help="local frida port (default: unused port)")
        parser.add_argument("--remote-port", metavar="PORT", action="store", dest="remote_port",
                            type=range_type(1, 65536), default=27042,
                            help="remote frida port (default: 27042)")

        parser.add_argument("--redirect-address", metavar="ADDRESS", action="store", dest="redirect_address",
                            type=str,
                            help="redirect traffic to target address (default: localhost)")
        parser.add_argument("--redirect-port", metavar="PORT", action="store", dest="redirect_port",
                            type=range_type(1, 65536),
                            help="redirect traffic to target port (default: 8080)")

    def run(self, args: Namespace) -> Optional[int]:
        device = args.device_picker.pick()

        server = FridaAndroidServer(
            device=device,
            local_port=args.local_port or utils.get_free_port(),
            remote_port=args.remote_port,
            serve=args.serve
        )

        with server:
            objection_args = [utils.get_interpreter(), "-m", "objection.console.cli"]
            if environ.debug:
                objection_args += ["--debug"]
            objection_args += ["-N", "-p", server.local_port]

            package = args.package
            if utils.is_empty(package):
                target_app = server.get_frontmost_application()
                if target_app is None:
                    raise CommandError("Unknown frontmost application")
                package = target_app.identifier
            environ.logger.info(f"Frida inject target application: {package}")

            objection_args += ["-g", package]
            objection_args += ["explore"]

            for command in args.startup_command:
                objection_args += ["--startup-command", command]
            if args.file_commands:
                objection_args += ["--file-commands", args.file_commands]
            if args.startup_script:
                objection_args += ["--startup-script", args.startup_script]
            if args.plugin_folder:
                objection_args += ["--plugin-folder", args.plugin_folder]

            if args.redirect_address or args.redirect_port:
                # 如果需要重定向到本地端口
                address = args.redirect_address
                port = args.redirect_port or 8080
                uid = device.get_uid(package)
                with device.redirect(address, port, uid):
                    return utils.popen(*objection_args).call()
            else:
                return utils.popen(*objection_args).call()


command = Command()
if __name__ == "__main__":
    command.main()
