#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native message roots retain scope and typed owner-slot semantics."""

import pytest

from linktools.ai.steps import native_store_key
from linktools.ai.core import JsonValue
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice
from linktools.ai.persistence.codec import (
    _CURRENT_CODEC, _decode_domain, _encode_persisted_domain, decode_domain, encode_domain,
)
from linktools.ai.persistence.contracts import ArchivedContextBlock, TranscriptChunk, TranscriptOrigin


def _root() -> NativeMessageRoot:
    return NativeMessageRoot("group", native_store_key("a", "tenant"), (NativeMessageSlice("b" * 64, 7, 1, 5),))


@pytest.mark.parametrize("persisted", (False, True))
def test_native_roots_round_trip_in_owner_slots(persisted: bool) -> None:
    root = _root()
    values = (root, ArchivedContextBlock(root, "c" * 64, 9),
              TranscriptChunk("owner", 0, 4, TranscriptOrigin.RAW, "raw", "c" * 64, 9, root))
    for value in values:
        wire = _encode_persisted_domain(value) if persisted else encode_domain(value)
        assert _decode_domain(wire, type(value), _CURRENT_CODEC, persisted=persisted) == value
    assert encode_domain(root)["$dataclass"] == "native_message_root"


def test_native_root_wire_is_explicit_and_validates_ranges() -> None:
    root = _root()
    assert NativeMessageRoot.from_json(root.to_json()) == root
    assert root.slice(1, 3).slices == (NativeMessageSlice("b" * 64, 7, 2, 4),)
    for value in ({**root.to_json(), "type": "unknown"},
                  {**root.to_json(), "store_key": "A" * 64},
                  {**root.to_json(), "slices": [{"first": "b" * 64, "count": 7, "start": True, "end": 4}]}):
        with pytest.raises((TypeError, ValueError)):
            NativeMessageRoot.from_json(value)


def test_root_shaped_business_json_remains_business_json() -> None:
    business = {"message": _root().to_json(), "typed": encode_domain(_root())}
    assert decode_domain(encode_domain(business), JsonValue) == business
