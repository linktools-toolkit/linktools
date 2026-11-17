#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ephemeral, explicitly selected structured output text; never a committed result."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartStartEvent, ToolCallPart, ToolCallPartDelta
from pydantic_core import from_json

from ..agent import OutputBinding
from ..capability import ExecutionCapability
from ..core import JsonValue


_MAX_PREVIEW_CHARS = 300


@dataclass
class OutputPreview:
    field_name: str
    max_chars: int
    parts: dict[int, ToolCallPart] = field(default_factory=dict)
    selected: int | None = None
    sequence: int = 0
    text: str = ""

    @classmethod
    def from_binding(cls, binding: OutputBinding, capability: ExecutionCapability) -> "OutputPreview | None":
        name = capability.output_preview_field
        if name is None:
            return None
        schema = binding.schema_definition
        properties = schema.get("properties")
        definition = properties.get(name) if isinstance(properties, Mapping) else None
        if binding.mode != "structured" or not isinstance(definition, Mapping) or definition.get("type") != "string":
            return None
        maximum = definition.get("maxLength", _MAX_PREVIEW_CHARS)
        if type(maximum) is not int or maximum < 1:
            return None
        return cls(name, min(maximum, _MAX_PREVIEW_CHARS))

    def new_stream(self) -> "OutputPreview":
        return type(self)(self.field_name, self.max_chars)

    def observe(self, event: object) -> "tuple[str, dict[str, JsonValue]] | None":
        reset = False
        if isinstance(event, PartStartEvent):
            if self.selected is not None and (self.selected == event.index or isinstance(event.part, ToolCallPart)):
                self.selected = None
                reset = True
            self.parts.pop(event.index, None)
            if isinstance(event.part, ToolCallPart):
                self.parts[event.index] = event.part
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
            part = self.parts.get(event.index)
            if part is not None:
                updated = event.delta.apply(part)
                if self.selected == event.index and (updated.tool_name != part.tool_name or updated.tool_call_id != part.tool_call_id):
                    self.selected = None
                    reset = True
                self.parts[event.index] = updated
        elif isinstance(event, FinalResultEvent):
            # None is also emitted for deferred tools. It is not proof of native JSON output.
            matches = [index for index, part in self.parts.items()
                       if event.tool_name is not None and event.tool_call_id is not None
                       and (part.tool_name, part.tool_call_id) == (event.tool_name, event.tool_call_id)]
            selected = matches[0] if len(matches) == 1 else None
            reset = selected != self.selected
            self.selected = selected
        else:
            return None
        text = ""
        if self.selected is not None:
            args = self.parts[self.selected].args
            if isinstance(args, str):
                # Bound parsing work as well as emitted text. Unsupported/oversized output falls back to final output.
                if len(args) <= 65536:
                    try:
                        args = from_json(args, allow_partial="trailing-strings")
                    except ValueError:
                        args = None
                else:
                    args = None
            value = args.get(self.field_name) if isinstance(args, dict) else None
            if isinstance(value, str):
                # A partial unicode escape must not create an invalid UTF-8 transport string.
                text = value[:self.max_chars].encode("utf-8", errors="ignore").decode("utf-8")
        if text == self.text and not reset:
            return None
        self.text = text
        self.sequence += 1
        return text, {"preview_field": self.field_name, "preview_seq": self.sequence,
                      "part_index": self.selected if self.selected is not None else 0, "part_start": True}


__all__ = ["OutputPreview"]
