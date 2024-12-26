#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : stub.py
@time    : 2024/8/6 16:34 
@site    : https://github.com/ice-black-tea
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
from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    import pathlib
    from typing import Optional, Callable, List, Iterable

    from .. import BaseEnviron
    from .command import SubCommand


def get_commands(environ: "BaseEnviron") -> "Iterable[SubCommand]":
    import argparse
    import os

    from .. import utils
    from .command import SubCommand

    commands: "List[SubCommand]" = []

    def register_command(name: str, description: str):
        def wrapper(cls: "Callable[[str, str], SubCommand]"):
            commands.append(cls(name, description))
            return None

        return wrapper

    def get_stub_path(environ: "BaseEnviron") -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            f"env_v{environ.version}",
            utils.get_md5(utils.get_interpreter())
        )

    def get_alias_path(environ: "BaseEnviron") -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            f"alias_v{environ.version}",
            utils.get_md5(utils.get_interpreter())
        )

    def get_default_shell(environ: "BaseEnviron") -> "Optional[str]":
        return "bash" if environ.system != "windows" else "powershell"

    @register_command(name="shell", description="run shell command")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("-c", "--command", help="shell command", default=None)
            return parser

        def run(self, args: "argparse.Namespace"):
            shell = environ.tools["shell"]
            if not shell.exists:
                raise NotImplementedError(f"Not found shell path")

            paths = os.environ.get("PATH", "").split(os.pathsep)
            stub_path = str(get_stub_path(environ))
            if stub_path not in paths:
                paths.append(stub_path)
            stub_path = str(environ.tools.stub_path)
            if stub_path not in paths:
                paths.append(stub_path)

            env = dict(PATH=os.pathsep.join(paths))
            if args.command:
                return utils.popen(args.command, shell=True, append_env=env).call()

            return shell.popen(append_env=env).call()

    @register_command(name="alias", description="generate shell alias script")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"], default=None)
            parser.add_argument("--reload", action="store_true", help="reload alias script", default=False)
            return parser

        def run(self, args: "argparse.Namespace"):
            alias_path = get_alias_path(environ)
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.reload and os.path.exists(alias_path):
                environ.logger.info(f"Found alias script: {alias_path}")
                print(utils.read_file(alias_path, text=True), flush=True)
                return 0

            from .. import metadata
            from ..cli import commands
            from ..cli.argparse import auto_complete
            from ..cli.command import iter_module_commands, iter_entry_point_commands
            from .._tools import Tool

            stub_path = get_stub_path(environ)
            stub_path.mkdir(parents=True, exist_ok=True)
            utils.clear_directory(stub_path)

            executables = []
            command_infos = {
                command_info.id: command_info
                for command_info in (
                    *iter_module_commands(commands, onerror="warn"),
                    *iter_entry_point_commands(metadata.__ep_group__, onerror="warn")
                )
            }
            for command_info in command_infos.values():
                if command_info.command:
                    temp = command_info
                    names = [command_info.command_name]
                    while temp.parent_id in command_infos:
                        temp = command_infos[temp.parent_id]
                        names.append(temp.command_name)
                    executable = "-".join(reversed(names))
                    executables.append(executable)
                    Tool.create_stub_file(
                        stub_path / Tool.get_stub_name(executable),
                        utils.list2cmdline([utils.get_interpreter(), "-m", command_info.module]),
                        system=environ.system
                    )
                    environ.logger.info(f"Found alias: {executable} -> {command_info.module}")

            shell = args.shell or get_default_shell(environ)

            lines = []
            if auto_complete:
                try:
                    environ.logger.info("Generate completion script ...")
                    lines.append(auto_complete.shellcode(executables, shell=shell))
                except:
                    pass

            tools_path = environ.tools.stub_path
            if shell in ("bash", "zsh"):
                lines.append(f"export PATH=\"$PATH:{stub_path}:{tools_path}\"")
            elif shell in ("fish",):
                lines.append(f"set -x PATH \"$PATH\" \"{stub_path}\" \"{tools_path}\"")
            elif shell in ("tcsh",):
                lines.append(f"setenv PATH \"$PATH:{stub_path}:{tools_path}\"")
            elif shell in ("powershell",):
                lines.append(f"$env:PATH=\"$env:PATH;{stub_path}:{tools_path}\"")

            result = os.linesep.join(lines)
            utils.write_file(alias_path, result)
            print(result, flush=True)

    @register_command(name="completion", description="generate shell auto complete script (deprecated)")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"])
            return parser

        def run(self, args: "argparse.Namespace"):
            environ.logger.warning("Not support generate completion script, already integrated into alias subcommand")

    @register_command(name="java", description="generate java environment script")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"])
            parser.add_argument("version", metavar="VERSION", nargs="?",
                                help="java version, such as 11.0.23 / 17.0.11 / 22.0.1")
            return parser

        def run(self, args: "argparse.Namespace"):
            java = environ.tools["java"]
            if args.version:
                java = java.copy(version=args.version)

            cmdline = java.make_stub_cmdline("java")
            shell = args.shell or get_default_shell(environ)

            lines = []
            if shell in ("bash", "zsh"):
                lines.append(f"alias java='{cmdline}'")
                lines.append(f"export JAVA_VERSION='{java.get('version')}'")
                lines.append(f"export JAVA_HOME='{java.get('home_path')}'")
                lines.append(f"export PATH=\"$JAVA_HOME/bin:$PATH\"")
            elif shell in ("fish",):
                lines.append(f"alias java '{cmdline}'")
                lines.append(f"set -x JAVA_VERSION '{java.get('version')}'")
                lines.append(f"set -x JAVA_HOME '{java.get('home_path')}'")
                lines.append(f"set -x PATH \"$JAVA_HOME/bin\" \"$PATH\"")
            elif shell in ("tcsh",):
                lines.append(f"alias java '{cmdline}'")
                lines.append(f"setenv JAVA_VERSION '{java.get('version')}'")
                lines.append(f"setenv JAVA_HOME '{java.get('home_path')}'")
                lines.append(f"setenv PATH \"$JAVA_HOME/bin:$PATH\"")
            elif shell in ("powershell",):
                lines.append(f"function __tool_java__ {{ {cmdline} $args }}")
                lines.append(f"Set-Alias -Name java -Value __tool_java__")
                lines.append(f"$env:JAVA_VERSION='{java.get('version')}'")
                lines.append(f"$env:JAVA_HOME='{java.get('home_path')}'")
                lines.append(f"$env:PATH=\"$env:JAVA_HOME\\bin;$env:PATH\"")

            result = os.linesep.join(lines)
            print(result, flush=True)

    @register_command(name="update", description=f"update {environ.name} packages")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("dependencies", metavar="DEPENDENCY", nargs='*', default=None,)
            return parser

        def run(self, args: "argparse.Namespace"):
            try:
                package = environ.name
                if args.dependencies:
                    dependencies = f"[{','.join(args.dependencies)}]"
                    package = f"{package}{dependencies}"
                utils.popen(
                    utils.get_interpreter(), "-m", "pip", "install", "-U", package,
                ).check_call()
            except Exception as e:
                environ.logger.warning(f"Update {environ.name} packages failed: {e}")

    @register_command(name="clean", description="clean temporary files")
    class Command(SubCommand):

        def create_parser(self, type: "Callable[..., ArgumentParser]") -> "argparse.ArgumentParser":
            parser = super().create_parser(type)
            parser.add_argument("days", metavar="DAYS", nargs="?", type=int, default=7,
                                help="expire days")
            return parser

        def run(self, args: "argparse.Namespace"):
            environ.clean_temp_files(expire_days=args.days)

    return commands


if __name__ == '__main__':
    import argparse
    import logging
    from .. import environ

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

    command_parser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
    command_parser.required = True

    for command in get_commands(environ):
        env_parser = command.create_parser(command_parser.add_parser)
        env_parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
        env_parser.set_defaults(func=command.run)

    tool_parser = command_parser.add_parser("tool")
    tool_parser.add_argument("name", help="tool Name").required = True
    tool_parser.add_argument("args", nargs=argparse.REMAINDER, help="tool Args")
    tool_parser.set_defaults(func=lambda args: \
        environ.get_tool(args.name, cmdline=None) \
                             .popen(*args.args) \
                             .call())

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.CRITICAL
        if not args.verbose
        else logging.DEBUG
    )

    exit(args.func(args))
