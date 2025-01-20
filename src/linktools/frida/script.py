#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : script.py 
@time    : 2022/01/22
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
import abc
import json
import os
import threading
from typing import Union, Optional

from .. import utils, environ
from ..metadata import __missing__
from ..rich import confirm

_logger = environ.get_logger("frida.app")


class FridaUserScript(metaclass=abc.ABCMeta):

    def __init__(self):
        self._source: Union[str, object] = __missing__
        self._lock = threading.RLock()

    @property
    def source(self) -> Optional[str]:
        return self.load()

    def load(self) -> Optional[str]:
        with self._lock:
            if self._source is __missing__:
                self._source = self._load()
            return self._source

    def clear(self) -> None:
        with self._lock:
            self._source = __missing__

    @property
    @abc.abstractmethod
    def filename(self):
        pass

    @abc.abstractmethod
    def _load(self) -> Optional[str]:
        pass

    def as_dict(self) -> dict:
        return {"filename": str(self.filename), "source": self.source}

    def as_json(self) -> str:
        return json.dumps(self.as_dict())

    def __repr__(self):
        class_name = self.__class__.__name__
        if class_name.startswith("Frida"):
            class_name = class_name[len("Frida"):]
        return f"{class_name}(filename={self.filename})"


class FridaScriptFile(FridaUserScript):

    def __init__(self, file_path: str):
        super().__init__()
        self._path = file_path

    @property
    def path(self):
        return self._path

    @property
    def filename(self):
        return self._path

    def _load(self) -> Optional[str]:
        _logger.info(f"Load {self}")
        return utils.read_file(self._path, text=True)


class FridaEvalCode(FridaUserScript):

    def __init__(self, code):
        super().__init__()
        self._code = code

    @property
    def filename(self):
        return "<anonymous>"

    def _load(self):
        return self._code


class FridaShareScript(FridaUserScript):

    def __init__(self, url: str, cached: bool = False, trusted: bool = False):
        super().__init__()
        self._url = url
        self._cached = cached
        self._trusted = trusted

    @property
    def filename(self):
        return self._url

    def _load(self):
        with environ.get_url_file(self._url) as file:  # 文件锁，避免多进程同时操作

            # 如果是本地文件，直接就返回了
            if file.is_local:
                _logger.info(f"Load local {self}")
                source = utils.read_file(file.save(), text=True)
                return source

            # 判断是否需要情况缓存
            if not self._cached:
                file.clear()

            _logger.info(f"Download {self}")
            dest_path = file.save()

            source = utils.read_file(dest_path, text=True)
            if self._trusted:
                _logger.info(f"Load trusted {self}")
                return source

            cached_md5 = ""
            cached_md5_path = dest_path + ".md5"
            if os.path.exists(cached_md5_path):
                cached_md5 = utils.read_file(cached_md5_path, text=True)

            source_md5 = utils.get_md5(source)
            if cached_md5 == source_md5:
                _logger.info(f"Load trusted {self}")
                return source

            line_count = 20
            source_lines = source.splitlines(keepends=True)
            source_summary = "".join(source_lines[:line_count])
            if len(source_lines) > line_count:
                source_summary += "... ..."

            prompt = f"This is the first time you're running this particular snippet, " \
                     f"or the snippet's source code has changed.{os.linesep}" \
                     f"Url: {self._url}{os.linesep}" \
                     f"Original md5: {cached_md5}{os.linesep}" \
                     f"Current md5: {source_md5}{os.linesep}" \
                     f"Source: {os.linesep}{source_summary}{os.linesep}" \
                     f"Are you sure you'd like to trust it?"

            if confirm(prompt):
                utils.write_file(cached_md5_path, source_md5)
                _logger.info(f"Load trusted {self}")
                return source
            else:
                _logger.info(f"Ignore untrusted {self}")
                return None
