#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityBundle: the output of resolving one (or many) CapabilityRef(s).
A bundle contributes zero or more of: prompt sections (injected text),
tool_contributions (ToolContribution with explicit descriptors), middleware,
pipelines, and resources."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class CapabilityBundle:
    prompt_sections: "Mapping[str, str]" = field(default_factory=dict)
    toolsets: "tuple[Any, ...]" = ()
    tool_contributions: "tuple[Any, ...]" = ()
    middleware: "tuple[Any, ...]" = ()
    resources: "tuple[Any, ...]" = ()
    pipelines: "tuple[Any, ...]" = ()

    @classmethod
    def empty(cls) -> "CapabilityBundle":
        return cls()
