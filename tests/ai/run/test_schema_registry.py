import pytest
from pydantic import BaseModel

from linktools.ai.errors import ManifestDriftError
from linktools.ai.run.schema_registry import OutputSchemaRegistry


class Output(BaseModel):
    value: str


def test_registry_resolves_exact_revision_and_rejects_unknown():
    registry = OutputSchemaRegistry()
    registry.register("output", "1", Output)
    assert registry.resolve("output", "1") is Output
    with pytest.raises(ManifestDriftError):
        registry.resolve("output", "2")


def test_registry_rejects_fingerprint_drift():
    registry = OutputSchemaRegistry()
    registry.register("output", "1", Output)

    class Changed(BaseModel):
        other: int

    with pytest.raises(ManifestDriftError):
        registry.register("output", "1", Changed)
