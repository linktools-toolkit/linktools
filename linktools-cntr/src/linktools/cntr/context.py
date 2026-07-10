#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .container import BaseContainer


@dataclass
class EventContext:
    """Lifecycle event context passed to container hooks.

    Plain (non-frozen, non-slots) dataclass so legacy code that sets attributes
    dynamically keeps working (refactor spec §12.2). Defaults match the previous
    hand-written ``__init__`` (``is_full_containers`` defaults to True as before;
    every caller sets these fields explicitly).
    """

    commands: "list[str] | None" = None
    containers: "list[BaseContainer] | None" = None
    target_containers: "list[BaseContainer] | None" = None
    is_full_containers: bool = True
    # Optional, opt-in extension surface for hooks; legacy code ignores it.
    metadata: "dict[str, Any]" = field(default_factory=dict)
