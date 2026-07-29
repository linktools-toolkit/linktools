#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed event envelopes, payloads, classification, and wire codecs."""

from .context import EventStreamContext
from .envelope import EventEnvelope
from .models import EventCriticality
from .registry import (
    EventCodec,
    EventDescriptor,
    EventRegistry,
    EventSchemaError,
    UnknownEventPayload,
    build_default_registry,
    classify_event,
    default_codec,
)

__all__ = [
    "EventCodec",
    "EventCriticality",
    "EventDescriptor",
    "EventEnvelope",
    "EventRegistry",
    "EventSchemaError",
    "EventStreamContext",
    "UnknownEventPayload",
    "build_default_registry",
    "classify_event",
    "default_codec",
]
