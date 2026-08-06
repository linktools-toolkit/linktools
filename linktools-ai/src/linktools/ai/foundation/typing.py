#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small JSON typing aliases shared by pure layers."""

from typing import TypeAlias

JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
