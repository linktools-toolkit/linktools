#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode declared constructor fields without materializing omitted defaults."""

from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields
from functools import lru_cache
from inspect import Parameter, signature


@lru_cache(maxsize=128)
def _defaulted_parameters(target: type) -> frozenset[str]:
    declared = {field.name: field for field in fields(target)}
    defaulted = {
        name for name, field in declared.items()
        if field.init and (field.default is not MISSING or field.default_factory is not MISSING)
    }
    # SDK constructor aliases need not have the same names as dataclass fields.
    defaulted.update(
        name for name, parameter in signature(target).parameters.items()
        if name not in declared and parameter.default is not Parameter.empty
    )
    return frozenset(defaulted)


def decode_fields(
    value: Mapping[str, object],
    target: type,
    decoders: Mapping[str, Callable[[object], object]],
    *,
    required: set[str] | frozenset[str] = frozenset(),
    aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    if not required.issubset(value) or set(value) - set(decoders) - required:
        raise ValueError("durable message shape is invalid")
    defaulted = _defaulted_parameters(target)
    names = {name: name if aliases is None else aliases.get(name, name) for name in decoders}
    for wire_name, name in names.items():
        if wire_name not in value:
            if name not in defaulted:
                raise ValueError("durable message required field is missing")
    return {
        names[name]: decode(value[name])
        for name, decode in decoders.items() if name in value
    }


__all__ = ["decode_fields"]
