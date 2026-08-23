"""Model binding and resolution contracts."""

from typing import Protocol

from pydantic_ai.models import Model


class ModelBinding(Protocol):
    @property
    def route_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def connection_identity(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def materialize(self) -> Model: ...


class ModelResolver(Protocol):
    def resolve(self, route_id: str) -> ModelBinding: ...


__all__ = ["ModelBinding", "ModelResolver"]
