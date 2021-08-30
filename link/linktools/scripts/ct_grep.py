#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : TGrep.py
@time    : 2018/12/25
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
   /  oooooooooooooooo  .o.  oooo /,   \,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""

# !/usr/bin/python

# -*- coding: utf-8 -*-
import os
import re
import shutil
import zipfile

import lief
import magic
from colorama import Fore
from linktools.decorator import entry_point

from linktools import utils, logger
from linktools.android.argparser import ArgumentParser


class GrepHandler():
    _handlers = {}
    _filter_handlers = {}

    @staticmethod
    def match(*mimetypes, **kwargs):

        def decorator(fn):

            def wrapper(instance, filename: str, mimetype: str):
                try:
                    fn(instance, filename, mimetype)
                    return True
                except (KeyboardInterrupt, EOFError) as e:
                    raise e
                except:
                    return False

            for mimetype in mimetypes:
                if mimetype in GrepHandler._handlers:
                    raise Exception("redefine {} handler".format(mimetype))
                GrepHandler._handlers[mimetype] = wrapper

            filter = kwargs.get("filter")
            if filter is not None:
                if filter in GrepHandler._filter_handlers:
                    raise Exception("redefine {} handler".format(filter))
                GrepHandler._filter_handlers[filter] = wrapper

            return wrapper

        return decorator

    @staticmethod
    def handle(instance, filename: str, mimetype: str):
        if mimetype in GrepHandler._handlers:
            fn = GrepHandler._handlers[mimetype]
            if fn(instance, filename, mimetype):
                return True
        for key in GrepHandler._filter_handlers:
            if key(mimetype):
                fn = GrepHandler._filter_handlers[key]
                if fn(instance, filename, mimetype):
                    return True
        return False


class GrepMatcher:

    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, path: str):
        if not os.path.exists(path):
            return
        elif os.path.isfile(path):
            self.on_file(path)
            return
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                self.on_file(os.path.join(root, name))

    def on_file(self, filename: str):
        if os.path.exists(filename):
            mimetype = magic.from_file(filename, mime=True)
            if not GrepHandler.handle(self, filename, mimetype):
                self.on_binary(filename, mimetype)

    @GrepHandler.match(
        "application/xml",
        filter=lambda t: t.startswith("text/"),
    )
    def on_text(self, filename: str, mimetype: str):
        with open(filename, "rb") as fd:
            lines = fd.readlines()
            for i in range(0, len(lines)):
                out = self.match_content(lines[i].rstrip())
                if not utils.is_empty(out):
                    logger.message(Fore.CYAN, filename,
                                   Fore.RESET, ":", Fore.GREEN, i + 1,
                                   Fore.RESET, ": ", out)

    @GrepHandler.match(
        "application/zip",
        "application/x-gzip",
        "application/java-archive"
    )
    def on_zip(self, filename: str, mimetype: str):
        dirname = filename + ":"
        while os.path.exists(dirname):
            dirname = dirname + " "
        try:
            zip_file = zipfile.ZipFile(filename, "r")
            zip_file.extractall(dirname)
            self.match(dirname)
        finally:
            shutil.rmtree(dirname, ignore_errors=True)

    @GrepHandler.match(
        "application/x-executable",
        "application/x-sharedlib"
    )
    def on_elf(self, filename: str, mimetype: str):
        file = lief.parse(filename)
        for symbol in file.imported_symbols:
            out = self.match_content(symbol.name)
            if not utils.is_empty(out):
                logger.message(Fore.CYAN, filename,
                               Fore.RESET, ":", Fore.GREEN, "import_symbols",
                               Fore.RESET, ": ", out,
                               Fore.RESET, " match")

        for symbol in file.exported_symbols:
            out = self.match_content(symbol.name)
            if not utils.is_empty(out):
                logger.message(Fore.CYAN, filename,
                               Fore.RESET, ":", Fore.GREEN, "export_symbols",
                               Fore.RESET, ": ", out,
                               Fore.RESET, " match")

        self.on_binary(filename, mimetype=mimetype)

    @GrepHandler.match()
    def on_binary(self, filename: str, mimetype: str):
        with open(filename, "rb") as fd:
            for line in fd.readlines():
                if self.pattern.search(line) is not None:
                    logger.message(Fore.CYAN, filename,
                                   Fore.RESET, ": ", Fore.RED, mimetype,
                                   Fore.RESET, " match")
                    return

    def match_content(self, content):
        out, last = "", 0
        if type(content) == str:
            content = bytes(content, encoding="utf-8")
        for match in self.pattern.finditer(content):
            start, end = match.span()
            out = out + Fore.RESET + str(content[last:start], encoding="utf-8")
            out = out + Fore.RED + str(content[start:end], encoding="utf-8")
            last = end
        if not utils.is_empty(out):
            out = out + Fore.RESET + str(content[last:], encoding="utf-8")
        return out


@entry_point()
def main():
    parser = ArgumentParser(description='match files with regular expression')

    parser.add_argument('-i', '--ignore-case', action='store_true', default=False,
                        help='ignore case')
    parser.add_argument('pattern', action='store', default=None,
                        help='regular expression')
    parser.add_argument('files', metavar="file", action='store', nargs='*', default=None,
                        help='target files path')

    args = parser.parse_args()

    flags = 0
    if args.ignore_case:
        flags = flags | re.I
    pattern = re.compile(bytes(args.pattern, encoding="utf8"), flags=flags)

    if utils.is_empty(args.files):
        args.files = ["."]

    for file in args.files:
        GrepMatcher(pattern).match(file)


if __name__ == '__main__':
    main()
