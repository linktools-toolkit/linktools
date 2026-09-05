#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical metric persistence envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ..core import canonical_sha256, validate_observation_payload
from ..errors import AIError, ErrorCode
from ._model import (
    MetricAggregation,
    MetricDefinition,
    MetricMeasurement,
    MetricSource,
    MetricSourceKind,
    MetricType,
    Observation,
)


def _source_payload(source: MetricSource) -> dict[str, object]:
    return {
        "kind": source.kind.value,
        "measurement_name": source.measurement_name,
        "measurement_revision": source.measurement_revision,
        "indicator_field": source.indicator_field,
        "indicator_values": list(source.indicator_values),
    }


def _definition_payload(
    definition: MetricDefinition, *, include_description: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": definition.name,
        "revision": definition.revision,
        "observation_kind": definition.observation_kind,
        "source": _source_payload(definition.source),
        "metric_type": definition.metric_type.value,
        "unit": definition.unit,
        "default_aggregation": definition.default_aggregation.value,
        "query_fields": list(definition.query_fields),
    }
    if include_description:
        payload["description"] = definition.description
    return payload


def definition_semantic_digest(definition: MetricDefinition) -> str:
    return canonical_sha256(_definition_payload(definition, include_description=False))


def definition_envelope(
    namespace: str, definition: MetricDefinition
) -> dict[str, object]:
    return {
        "type": "MetricDefinitionEnvelope",
        "version": 1,
        "namespace": namespace,
        "definition": _definition_payload(definition, include_description=True),
    }


def definition_envelope_digest(namespace: str, definition: MetricDefinition) -> str:
    return canonical_sha256(definition_envelope(namespace, definition))


def observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "version": observation.version,
        "observation_id": observation.observation_id,
        "kind": observation.kind,
        "occurred_at": observation.occurred_at.isoformat(),
        "source_namespace": observation.source_namespace,
        "tenant_id": observation.tenant_id,
        "status": observation.status,
        "error_code": observation.error_code,
        "correlation": dict(observation.correlation),
        "dimensions": dict(observation.dimensions),
        "measurements": [
            {
                "name": item.name,
                "revision": item.revision,
                "value": item.value,
            }
            for item in observation.measurements
        ],
    }


def observation_envelope(namespace: str, observation: Observation) -> dict[str, object]:
    envelope: dict[str, Any] = {
        "type": "ObservationEnvelope",
        "version": 1,
        "namespace": namespace,
        "observation": observation_payload(observation),
    }
    validate_observation_payload(envelope)
    return envelope


def observation_digest(namespace: str, observation_id: str) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "namespace": namespace,
            "observation_id": observation_id,
        }
    )


def observation_payload_digest(namespace: str, observation: Observation) -> str:
    return canonical_sha256(observation_envelope(namespace, observation))


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            f"{field} must be a mapping",
        )
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _optional_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field=field)


def _text_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return tuple(_text(item, field=field) for item in value)


def decode_definition_envelope(
    payload: object, *, expected_namespace: str
) -> MetricDefinition:
    root = _mapping(payload, field="definition envelope")
    if root.get("type") != "MetricDefinitionEnvelope":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = root.get("version")
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if root.get("namespace") != expected_namespace:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    data = _mapping(root.get("definition"), field="definition")
    source_data = _mapping(data.get("source"), field="metric source")
    try:
        source = MetricSource(
            kind=MetricSourceKind(_text(source_data["kind"], field="source kind")),
            measurement_name=_optional_text(
                source_data.get("measurement_name"),
                field="measurement_name",
            ),
            measurement_revision=_optional_integer(
                source_data.get("measurement_revision"),
                field="measurement_revision",
            ),
            indicator_field=_optional_text(
                source_data.get("indicator_field"),
                field="indicator_field",
            ),
            indicator_values=_text_tuple(
                source_data.get("indicator_values", []),
                field="indicator_values",
            ),
        )
        return MetricDefinition(
            name=_text(data["name"], field="name"),
            revision=_integer(data["revision"], field="revision"),
            observation_kind=_text(
                data["observation_kind"], field="observation_kind"
            ),
            source=source,
            metric_type=MetricType(
                _text(data["metric_type"], field="metric_type")
            ),
            unit=_text(data["unit"], field="unit"),
            default_aggregation=MetricAggregation(
                _text(data["default_aggregation"], field="default_aggregation")
            ),
            query_fields=_text_tuple(
                data.get("query_fields", []), field="query_fields"
            ),
            description=_optional_text(
                data.get("description"), field="description"
            ),
        )
    except AIError as exc:
        if exc.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
            raise
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc


def decode_observation_envelope(
    payload: object, *, expected_namespace: str
) -> Observation:
    root = _mapping(payload, field="observation envelope")
    if root.get("type") != "ObservationEnvelope":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = root.get("version")
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if root.get("namespace") != expected_namespace:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    data = _mapping(root.get("observation"), field="observation")
    try:
        occurred_at = datetime.fromisoformat(
            _text(data["occurred_at"], field="occurred_at")
        )
        correlation_data = _mapping(data.get("correlation", {}), field="correlation")
        dimensions_data = _mapping(data.get("dimensions", {}), field="dimensions")
        raw_measurements = data.get("measurements", [])
        if not isinstance(raw_measurements, list):
            raise TypeError("measurements must be a list")
        measurements = []
        for raw in raw_measurements:
            item = _mapping(raw, field="measurement")
            measurements.append(
                MetricMeasurement(
                    name=_text(item["name"], field="measurement name"),
                    revision=_integer(
                        item["revision"], field="measurement revision"
                    ),
                    value=item["value"],
                )
            )
        correlation: dict[str, str | int] = {}
        for key, value in correlation_data.items():
            if not isinstance(key, str):
                raise TypeError("invalid correlation key")
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise TypeError("invalid correlation value")
            correlation[key] = value
        dimensions: dict[str, str] = {}
        for key, value in dimensions_data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("invalid dimension value")
            dimensions[key] = value
        return Observation(
            version=_integer(data["version"], field="version"),
            observation_id=_text(data["observation_id"], field="observation_id"),
            kind=_text(data["kind"], field="kind"),
            occurred_at=occurred_at,
            source_namespace=_optional_text(
                data.get("source_namespace"), field="source_namespace"
            ),
            tenant_id=_optional_text(data.get("tenant_id"), field="tenant_id"),
            status=_optional_text(data.get("status"), field="status"),
            error_code=_optional_text(data.get("error_code"), field="error_code"),
            correlation=correlation,
            dimensions=dimensions,
            measurements=tuple(measurements),
        )
    except AIError as exc:
        if exc.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
            raise
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc


__all__: list[str] = []
