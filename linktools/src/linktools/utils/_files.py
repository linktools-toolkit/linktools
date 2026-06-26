#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import overload, TYPE_CHECKING

from ..errors import Error
from ._utils import get_environ, get_logger

DEFAULT_ENCODING = "utf-8"


def is_sub_path(path, root_path) -> bool:
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        abs_root_path = os.path.abspath(os.path.expanduser(root_path))
        return os.path.commonpath([abs_path, abs_root_path]) == abs_root_path
    except ValueError:
        return False


def join_path(root_path, *paths: str) -> Path:
    target_path = str(root_path)
    for path in paths:
        parent_path = target_path
        target_path = os.path.abspath(os.path.join(target_path, path))
        try:
            if os.path.commonpath([target_path, parent_path]) != parent_path:
                raise Error(f"Unsafe path \"{path}\"")
        except ValueError:
            raise Error(f"Unsafe path \"{path}\"")
    return Path(target_path)


if TYPE_CHECKING:

    @overload
    def read_file(path):
        ...


    @overload
    def read_file(path, text: False):
        ...


    @overload
    def read_file(path, text: True, encoding: str = DEFAULT_ENCODING):
        ...


    @overload
    def read_file(path, text: bool, encoding: str = DEFAULT_ENCODING):
        ...


def read_file(path, text: bool = False, encoding: str = DEFAULT_ENCODING):
    if text:
        with open(path, "rt", encoding=encoding) as fd:
            return fd.read()
    with open(path, "rb") as fd:
        return fd.read()


def write_file(path, data, encoding: str = DEFAULT_ENCODING) -> None:
    if isinstance(data, str):
        with open(path, "wt", encoding=encoding) as fd:
            fd.write(data)
    else:
        with open(path, "wb") as fd:
            fd.write(data)


def remove_file(path) -> None:
    if not os.path.exists(path):
        return
    environ = get_environ()
    if environ.debug:
        get_logger().debug(f"Remove File: {path}")
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except:
            pass


def clear_directory(path) -> None:
    if not os.path.isdir(path):
        return
    if get_environ().debug:
        get_logger().debug(f"Clear Directory: {path}")
    for name in os.listdir(path):
        target_path = os.path.join(path, name)
        if os.path.isdir(target_path):
            import shutil
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            try:
                os.remove(target_path)
            except:
                pass
