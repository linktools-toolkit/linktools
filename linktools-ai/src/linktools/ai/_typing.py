#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_typing.py"""
from typing import Any

#: A JSON-serializable value. Not statically enforced beyond `Any` -- a fully
#: recursive alias is not worth the complexity under quoted 3.10-compatible syntax.
JSONValue = Any
