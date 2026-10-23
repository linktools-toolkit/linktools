#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit model-binding composition using the SDK's native fallback policy."""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel, FallbackOn

from ..core import JsonValue
from ._contract import ModelBinding
from ._input import normalize_input_mime_types


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FallbackBinding:
    """Compose ordered bindings without taking over SDK fallback or lifecycle.

    ``fallback_on`` is passed unchanged to the SDK, including annotated response
    handlers. The contract describes candidate model semantics, not callable
    implementation identity. Each candidate still owns its input constraints
    and thinking profile.
    """

    bindings: tuple[ModelBinding, ...]
    fallback_on: FallbackOn = field(repr=False)

    def __init__(
        self, *bindings: ModelBinding, fallback_on: FallbackOn = (ModelAPIError,),
    ) -> None:
        if not bindings:
            raise ValueError("fallback binding requires at least one candidate")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "fallback_on", fallback_on)

    @property
    def provider(self) -> str:
        return "fallback"

    @property
    def model_identity(self) -> str:
        return "fallback:" + ",".join(binding.model_identity for binding in self.bindings)

    @property
    def contract(self) -> dict[str, JsonValue]:
        candidates: list[JsonValue] = []
        input_mime_types: set[str] = set()
        for binding in self.bindings:
            contract = dict(binding.contract)
            candidates.append({
                "provider": binding.provider,
                "model_identity": binding.model_identity,
                "contract": contract,
            })
            input_mime_types.update(normalize_input_mime_types(contract.get("input_mime_types", [])))
        return {
            "provider": self.provider,
            "model_identity": self.model_identity,
            "candidates": candidates,
            "input_mime_types": sorted(input_mime_types),
        }

    def materialize(
        self, *, prepare: Callable[[ModelBinding], Model] | None = None,
    ) -> Model:
        """Build a fresh native composition; the caller owns its async context.

        Runtime can prepare each candidate before SDK composition, preserving
        candidate-specific capabilities without inspecting model wrappers.
        """
        models = [prepare(binding) if prepare is not None else binding.materialize()
                  for binding in self.bindings]
        return FallbackModel(models[0], *models[1:], fallback_on=self.fallback_on)


__all__ = ["FallbackBinding"]
