#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .commands.root import Command

command = Command()

__all__ = [
    "Command",
    "command",
]

if __name__ == "__main__":
    raise SystemExit(command.main())
