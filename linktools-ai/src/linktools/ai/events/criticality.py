#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event criticality classification.

Criticality is owned by the :class:`EventRegistry` -- each
:class:`EventDescriptor` carries the level for its payload. ``classify_event``
here is a thin convenience that looks the payload up in the default registry;
it is no longer driven by the payload's Python class name, so renaming a
payload class cannot change its criticality.

Levels govern the persistence failure policy:

* ``STATE_CRITICAL`` -- bound to run/approval state; a persistence failure
  blocks the operation (fail closed).
* ``SECURITY_CRITICAL`` -- security decisions (deny/expose/pipeline); must be
  auditable, fail closed.
* ``OBSERVABILITY`` -- lifecycle markers, metrics, spans; best-effort (a
  persistence failure is logged but does not block).
"""

from enum import Enum
from typing import Any


class EventCriticality(str, Enum):
    STATE_CRITICAL = "state_critical"
    SECURITY_CRITICAL = "security_critical"
    OBSERVABILITY = "observability"


def classify_event(payload: Any) -> EventCriticality:
    """Map an event payload to its criticality via the default registry.

    The registry is the single source; the lookup is by the payload's
    registered ``event_type``, never by ``type(payload).__name__``. An
    unregistered payload defaults to observability (it cannot be state- or
    security-critical without an explicit descriptor).
    """
    from .registry import default_codec

    return default_codec.registry.criticality_of(payload)


__all__: "list[str]" = ["EventCriticality", "classify_event"]
