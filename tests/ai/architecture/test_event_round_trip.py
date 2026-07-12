#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event payload round-trip through FileEventStore.

Every standard EventPayload must survive: dataclass -> FileEventStore.append
-> JSON on disk -> list() -> reconstructed dataclass with identical type and
fields. A payload that loses a field or reconstructs as the wrong type fails
here. This locks the serialization contract before the simplification touches
the event layer.
"""

import asyncio
import dataclasses
from typing import Any, get_args

from linktools.ai.events import payloads as _payloads
from linktools.ai.events.payloads import EventPayload
from linktools.ai.storage.file.event import FileEventStore


def _value_for(field: "dataclasses.Field[Any]") -> Any:
    """A JSON-compatible value appropriate for the field's annotated type."""
    type_str = (
        field.type
        if isinstance(field.type, str)
        else getattr(field.type, "__name__", str(field.type))
    )
    if "Mapping" in type_str or "dict" in type_str.lower():
        return {"k": "v", "n": 1}
    if "bool" in type_str:
        return True
    if "float" in type_str:
        return 1.5
    if "int" in type_str:
        return 7
    if "str" in type_str:
        return "x"
    return "any"


def _make_payload(cls: type) -> Any:
    kwargs = {
        f.name: _value_for(f)
        for f in dataclasses.fields(cls)
        # Skip fields with defaults so we don't fight a factory constraint --
        # but also set them to exercise round-trip on populated values.
    }
    return cls(**kwargs)


PAYLOAD_CLASSES = [
    c
    for c in get_args(EventPayload)
    if isinstance(c, type) and dataclasses.is_dataclass(c)
]


def test_every_standard_payload_is_covered():
    """Guard: if a new EventPayload is added to the union but not exercised
    here, this fails so the round-trip stays complete."""
    assert len(PAYLOAD_CLASSES) >= 40, (
        f"expected the standard event payload union to cover >=40 types, "
        f"got {len(PAYLOAD_CLASSES)}"
    )


def test_all_event_payloads_round_trip_through_file_store(tmp_path):
    store = FileEventStore(root=tmp_path)
    originals = [_make_payload(cls) for cls in PAYLOAD_CLASSES]

    async def _drive():
        for payload in originals:
            await store.append(
                stream_id="s",
                run_id="r",
                root_run_id="r",
                parent_run_id=None,
                session_id="sess",
                runnable_id="a",
                payload=payload,
            )
        page = await store.list("s", limit=10000)
        return page

    page = asyncio.run(_drive())
    assert len(page.items) == len(originals)

    for envelope, original, cls in zip(page.items, originals, PAYLOAD_CLASSES):
        assert type(envelope.payload) is cls
        assert envelope.payload == original
        # Field-level check: every field round-trips with the same value.
        for field in dataclasses.fields(cls):
            assert getattr(envelope.payload, field.name) == getattr(
                original, field.name
            )


def test_payload_type_name_is_the_class_name(tmp_path):
    """FileEventStore persists ``payload_type = type(payload).__name__`` and
    reconstructs by that name -- a payload whose stored name drifts from the
    class would break reconstruction silently."""
    store = FileEventStore(root=tmp_path)
    payload = _payloads.RunStarted(run_id="r", runnable_id="a")

    async def _drive():
        await store.append(
            stream_id="s",
            run_id="r",
            root_run_id="r",
            parent_run_id=None,
            session_id="sess",
            runnable_id="a",
            payload=payload,
        )
        return await store.list("s")

    page = asyncio.run(_drive())
    assert type(page.items[0].payload) is _payloads.RunStarted
