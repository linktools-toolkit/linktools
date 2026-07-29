"""Runtime dependency inputs for the composition root."""

from dataclasses import dataclass

from ..model.resolver import ModelResolver


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    model_resolver: ModelResolver


__all__ = ["RuntimeDependencies"]
