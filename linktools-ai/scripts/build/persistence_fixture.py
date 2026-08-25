#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Importable durable capability fixture used by the persistence build gate."""

from pydantic_ai.capabilities import AbstractCapability


class PersistenceCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "runtime-persistence-fixture"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "PersistenceCapability":
        del kwargs
        return cls()


__all__ = ["PersistenceCapability"]
