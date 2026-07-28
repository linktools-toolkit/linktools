"""Public runtime composition for the v4 execution storage path."""

from __future__ import annotations

from typing import Any

from ..model.resolver import ModelResolver
from .modern import ModernRuntime
from .storage import RuntimeStorage


Runtime = ModernRuntime


def build_runtime(*, storage: RuntimeStorage, model_resolver: ModelResolver | None = None, **_: Any) -> Runtime:
    """Build a runtime from already-created v4 storage dependencies."""
    if not isinstance(storage, RuntimeStorage):
        raise TypeError("build_runtime requires RuntimeStorage")
    return ModernRuntime(storage=storage, model_resolver=model_resolver or ModelResolver())


__all__ = ["Runtime", "build_runtime"]
