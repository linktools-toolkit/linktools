#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata and ordered composition."""
from typing import TYPE_CHECKING, Iterable, List, Mapping, Tuple, Union

if TYPE_CHECKING:
    from typing import Iterator, Optional


class Integration:
    """A consumer-specific declaration."""

    consumer: str = ""

    @classmethod
    def iter_declarations(cls, values: "Integrations") -> "Iterator[Tuple[Optional[str], Integration]]":
        """Flatten ordered list/tuple compositions without evaluating declarations."""
        def visit(value: "Declaration") -> "Iterator[Tuple[Optional[str], Integration]]":
            if isinstance(value, cls):
                yield None, value
            elif isinstance(value, Mapping):
                for key, declaration in value.items():
                    if not isinstance(key, str) or not key:
                        raise ValueError("Invalid integration key")
                    if not isinstance(declaration, cls):
                        raise ValueError("Invalid integration: expected Integration")
                    yield key, declaration
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from visit(item)
            else:
                raise ValueError("Invalid integration: expected Integration, mapping, list or tuple")

        if isinstance(values, Mapping):
            yield from visit(values)
            return
        if isinstance(values, (str, bytes)):
            raise ValueError("Invalid integrations")
        try:
            entries = iter(values)
        except TypeError:
            raise ValueError("Invalid integrations") from None
        for entry in entries:
            yield from visit(entry)


Declaration = Union[Integration, Mapping[str, Integration], List["Declaration"], Tuple["Declaration", ...]]
Integrations = Union[Mapping[str, Integration], Iterable[Declaration]]
