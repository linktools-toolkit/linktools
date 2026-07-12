#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MappingProvider: a generic mapping-backed spec provider (spec §14.3).

Given ``{id: spec}``, it provides ``list_ids`` + ``get`` -- the two methods
every spec-provider Protocol requires. Used for built-in registries and test
fakes; downstream is NOT required to inherit (it satisfies the Protocol
structurally)."""

from typing import Generic, Mapping, TypeVar

T = TypeVar("T")


class MappingProvider(Generic[T]):
    """A simple in-memory spec provider backed by a ``{id: spec}`` mapping."""

    def __init__(self, specs: "Mapping[str, T]") -> None:
        self._specs = dict(specs)

    async def list_ids(self) -> "tuple[str, ...]":
        return tuple(self._specs.keys())

    async def get(self, spec_id: str) -> T:
        if spec_id not in self._specs:
            raise KeyError(spec_id)
        return self._specs[spec_id]
