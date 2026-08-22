#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable RuntimeCapability type identity."""

import pytest
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.capability import RuntimeCapability
from linktools.ai.errors import AIError, ErrorCode


class _OtherCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "other-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_OtherCapability":
        del kwargs
        return cls()


class _WrongFactoryCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "wrong-factory-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_OtherCapability":
        del kwargs
        return _OtherCapability()


class _RestorableCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "restorable-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_RestorableCapability":
        del kwargs
        return cls()


def test_from_spec_rejects_factory_returning_a_different_type() -> None:
    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec(
            "capability",
            _WrongFactoryCapability,
            config={},
        )
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_restore_rejects_factory_returning_a_different_type(monkeypatch: pytest.MonkeyPatch) -> None:
    value = RuntimeCapability.from_spec(
        "capability",
        _RestorableCapability,
        config={},
    )
    descriptor = value.descriptor
    assert descriptor is not None

    monkeypatch.setattr(
        _RestorableCapability,
        "from_spec",
        classmethod(lambda cls, **kwargs: _OtherCapability()),
    )

    with pytest.raises(AIError) as error:
        RuntimeCapability.restore(descriptor)
    assert error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
