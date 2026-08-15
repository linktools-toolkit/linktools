#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Pydantic AI agent from an already materialized definition."""

from typing import cast

from pydantic import BaseModel
from pydantic_ai import Agent, TextOutput
from pydantic_ai.models import Model

from ._definition import AgentDefinition
from ._output import AssistantTextOutput


def build_pydantic_agent(
    definition: AgentDefinition,
    *,
    model: Model,
) -> "Agent[None, object]":
    output_type: "type[BaseModel] | TextOutput" = definition.output_type
    if output_type is AssistantTextOutput:
        output_type = TextOutput(_assistant_text_output)
    return cast(
        "Agent[None, object]",
        Agent(
            model,
            name=definition.spec.id,
            system_prompt=definition.spec.system_prompt,
            instructions="\n".join(definition.spec.instructions),
            output_type=output_type,
        ),
    )


def _assistant_text_output(value: str) -> AssistantTextOutput:
    return AssistantTextOutput(text=value)


__all__ = ["build_pydantic_agent"]
