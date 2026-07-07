#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""User/uid/gid/shell helpers (spec §14.1)."""

import os

from .platform import get_system, is_unix_like, is_windows


if is_windows():

    def get_user(uid=None):
        import getpass
        return getpass.getuser()

    def get_uid(user=None):
        return 0

    def get_gid(user=None):
        return 0

    def get_shell_path():
        import shutil
        shell_path = shutil.which("powershell") or shutil.which("cmd")
        if shell_path:
            return shell_path
        if "ComSpec" in os.environ:
            shell_path = os.environ["ComSpec"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        raise NotImplementedError("Unsupported system `%s`" % get_system())

elif is_unix_like():

    def get_user(uid=None):
        if uid is not None:
            import pwd
            return pwd.getpwuid(int(uid)).pw_name
        import getpass
        return getpass.getuser()

    def get_uid(user=None):
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_uid
        return os.getuid()

    def get_gid(user=None):
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_gid
        return os.getgid()

    def get_shell_path():
        if "SHELL" in os.environ:
            shell_path = os.environ["SHELL"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        try:
            import pwd
            return pwd.getpwnam(get_user()).pw_shell
        except Exception:
            import shutil
            return shutil.which("zsh") or shutil.which("bash") or shutil.which("sh")

else:

    def get_user(uid=None):
        raise NotImplementedError("Unsupported system `%s`" % get_system())

    def get_uid(user=None):
        raise NotImplementedError("Unsupported system `%s`" % get_system())

    def get_gid(user=None):
        raise NotImplementedError("Unsupported system `%s`" % get_system())

    def get_shell_path():
        raise NotImplementedError("Unsupported system `%s`" % get_system())
