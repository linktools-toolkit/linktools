#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable RuntimeCapability type identity."""

import pytest
from linktools.ai.capability import RuntimeCapability
from linktools.ai.errors import AIError, ErrorCode
from pydantic_ai.capabilities import AbstractCapability


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
    raise_serialization_error = False
    factory_calls = 0

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        if cls.raise_serialization_error:
            raise RuntimeError("serialization failure")
        return "restorable-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_RestorableCapability":
        del kwargs
        cls.factory_calls += 1
        return cls()


def test_from_spec_rejects_factory_returning_a_different_type() -> None:
    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec(
            "capability",
            _WrongFactoryCapability,
            config={},
        )
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_restore_rejects_factory_returning_a_different_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_invalid_identity_is_rejected_before_factory_execution() -> None:
    before = _RestorableCapability.factory_calls

    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec(
            "",
            _RestorableCapability,
            config={},
        )

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    assert _RestorableCapability.factory_calls == before


def test_restore_ignores_unknown_descriptor_field() -> None:
    value = RuntimeCapability.from_spec(
        "capability",
        _RestorableCapability,
        config={},
    )
    descriptor = value.descriptor
    assert descriptor is not None
    descriptor["future_metadata"] = {"$future_v2": ["must", "not", "decode"]}

    restored = RuntimeCapability.restore(descriptor)

    assert restored.id == value.id
    assert restored.revision == value.revision
    assert restored.fingerprint == value.fingerprint


def test_restore_rejects_missing_descriptor_field() -> None:
    value = RuntimeCapability.from_spec(
        "capability",
        _RestorableCapability,
        config={},
    )
    descriptor = value.descriptor
    assert descriptor is not None
    descriptor.pop("serialization_name")

    with pytest.raises(AIError) as error:
        RuntimeCapability.restore(descriptor)
    assert error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


@pytest.mark.parametrize("revision", [0, -1, True])
def test_invalid_revision_is_rejected_before_factory_execution(revision: object) -> None:
    before = _RestorableCapability.factory_calls

    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec(
            "capability",
            _RestorableCapability,
            config={},
            revision=revision,  # type: ignore[arg-type]
        )

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    assert _RestorableCapability.factory_calls == before


def test_from_spec_maps_serialization_extension_failure() -> None:
    _RestorableCapability.raise_serialization_error = True
    try:
        with pytest.raises(AIError) as error:
            RuntimeCapability.from_spec(
                "capability",
                _RestorableCapability,
                config={},
            )
    finally:
        _RestorableCapability.raise_serialization_error = False

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_restore_maps_serialization_extension_failure() -> None:
    value = RuntimeCapability.from_spec(
        "capability",
        _RestorableCapability,
        config={},
    )
    descriptor = value.descriptor
    assert descriptor is not None

    _RestorableCapability.raise_serialization_error = True
    try:
        with pytest.raises(AIError) as error:
            RuntimeCapability.restore(descriptor)
    finally:
        _RestorableCapability.raise_serialization_error = False

    assert error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
