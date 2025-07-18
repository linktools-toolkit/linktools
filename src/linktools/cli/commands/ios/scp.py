#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from argparse import Namespace
from typing import Optional, Type, List

import paramiko
from paramiko.ssh_exception import SSHException

from linktools import utils
from linktools.cli import IOSCommand, CommandParser
from linktools.mobile.ios import SibDevice
from linktools.ssh import SSHClient

_REMOTE_PATH_PREFIX = ":"


class SCPFile(os.PathLike):
    path: str = property(fget=lambda self: self._path)
    is_local: bool = property(fget=lambda self: not self._is_remote)
    is_remote: bool = property(fget=lambda self: self._is_remote)

    def __init__(self, path: str):
        if path.startswith(_REMOTE_PATH_PREFIX):
            self._is_remote = True
            self._path = path[len(_REMOTE_PATH_PREFIX):]
        else:
            self._is_remote = False
            self._path = os.path.abspath(os.path.expanduser(path))

    def __fspath__(self):
        return self._path


class Command(IOSCommand):
    """
    Securely copy files to/from a jailbroken iOS device using OpenSSH
    """

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [NotImplementedError, FileNotFoundError, SSHException]

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-u", "--username", action="store", default="root",
                            help="iOS ssh username (default: root)")
        parser.add_argument("-p", "--port", action="store", type=int, default=22,
                            help="iOS ssh port (default: 22)")
        parser.add_argument("--password", action="store",
                            help="iOS ssh password")

        parser.add_argument("source", action="store", type=SCPFile, default=None,
                            help=f"source file path, remote path needs to be prefixed with \"{_REMOTE_PATH_PREFIX}\"")
        parser.add_argument("target", action="store", type=SCPFile, default=None,
                            help=f"target file path, remote path needs to be prefixed with \"{_REMOTE_PATH_PREFIX}\"")

    def run(self, args: Namespace) -> Optional[int]:
        device: SibDevice = args.device_picker.pick()

        local_port = utils.get_free_port()
        with device.forward(local_port, args.port):
            with SSHClient() as client:
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect_with_pwd("localhost", port=local_port, username=args.username, password=args.password)
                if args.source.is_remote and args.target.is_local:
                    client.get_file(args.source.path, args.target.path)
                elif args.source.is_local and args.target.is_remote:
                    client.put_file(args.source.path, args.target.path)
                elif args.source.is_remote and args.target.is_remote:
                    raise NotImplementedError("It does not support copying files between remote files")
                elif args.source.is_local and args.target.is_local:
                    raise NotImplementedError("It does not support copying files between local files")

        return None


command = Command()
if __name__ == "__main__":
    command.main()
