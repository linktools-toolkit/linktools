#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target resolution for the evaluation plane.

An :class:`EvalTarget` names what to evaluate (kind = agent / skill / subagent,
an id, an optional revision). Resolution maps that target to the concrete spec
the executor runs. The core does NOT scan the filesystem or the agent registry
-- a caller supplies the mapping (explicit registration), so the evaluation
plane stays business-neutral and deterministic across replays."""

from collections.abc import Mapping

from .models import EvalTarget


class MappingTargetResolver:
    """Resolve an EvalTarget to its spec by ``id`` from a caller-supplied
    mapping. ``revision`` is informational: the mapping owns the pinned spec, so
    a replay resolves the same bytes the original run used."""

    def __init__(self, mapping: "Mapping[str, object]") -> None:
        self._mapping = dict(mapping)

    async def resolve(self, target: EvalTarget) -> object:
        if target.id not in self._mapping:
            raise KeyError(f"target not found: {target.kind}:{target.id}")
        return self._mapping[target.id]


__all__: "list[str]" = ["MappingTargetResolver"]
