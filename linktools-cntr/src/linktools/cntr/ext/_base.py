#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declarations and ordered composition."""
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union
from collections.abc import Mapping

if TYPE_CHECKING:
    from typing import Iterator


class Integration:
    """A consumer-specific declaration."""

    consumer: str = ""

    @classmethod
    def iter_declarations(cls, values: "Integrations") -> "Iterator[Integration]":
        """Flatten ordered list/tuple compositions without evaluating declarations."""
        def visit(value: "Declaration") -> "Iterator[Integration]":
            if isinstance(value, cls):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from visit(item)
            else:
                raise ValueError("Invalid integration: expected Integration, list or tuple")

        if isinstance(values, cls):
            yield values
            return
        if isinstance(values, (str, bytes, Mapping)):
            raise ValueError("Invalid integrations: mappings and strings are not declarations")
        try:
            entries = iter(values)
        except TypeError:
            raise ValueError("Invalid integrations") from None
        for entry in entries:
            yield from visit(entry)


Declaration = Union[Integration, List["Declaration"], Tuple["Declaration", ...]]
Integrations = Union[Integration, Iterable[Declaration]]
