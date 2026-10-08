#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoring declarations and immutable evaluation contracts."""

import math
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

from pydantic_ai.messages import UserContent

from ..agent import AgentInputCaptureRef
from ..asset import AssetVersionRef
from ..core import (
    ImmutableJsonMapping, JsonValue, Principal, WorkspaceFileInput,
    canonical_sha256, principal_identity_payload, validate_idempotency_key,
)
from ..task import (
    TaskGraphCaptureRef, TaskGraphLimits, TaskGraphTemplate,
    TaskGraphTemplateRef, TaskInvocationInputRef, TaskRef, TaskResultRef,
)


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name must be non-empty")


def _revision(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("revision must be a positive integer")


def _positive(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("value must be finite and positive")


@dataclass(frozen=True, slots=True)
class CaseRef:
    dataset_id: str
    case_id: str
    revision: int

    def __post_init__(self) -> None:
        _name(self.dataset_id)
        _name(self.case_id)
        _revision(self.revision)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"dataset_id": self.dataset_id, "case_id": self.case_id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class DatasetRef:
    id: str
    revision: int

    def __post_init__(self) -> None:
        _name(self.id)
        _revision(self.revision)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"id": self.id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class InlineValue:
    content: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if set(self.content) != {"value"}:
            raise ValueError("inline value requires exactly one value")
        object.__setattr__(self, "content", ImmutableJsonMapping(self.content))

    @classmethod
    def from_value(cls, value: JsonValue) -> "InlineValue":
        return cls({"value": value})

    @property
    def value(self) -> JsonValue:
        return self.content["value"]

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "json", "value": self.value}


EvaluationValue: TypeAlias = InlineValue | AssetVersionRef


def value_mapping(value: EvaluationValue | None) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, InlineValue):
        return value.to_mapping()
    return {"kind": "asset", "ref": value.to_payload()}


@dataclass(frozen=True, slots=True)
class LabelProvenance:
    source: str = "provided/unknown"
    actor: str | None = None
    procedure: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        _name(self.source)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"source": self.source, "actor": self.actor, "procedure": self.procedure, "revision": self.revision}


class _Unset:
    __slots__ = ()


_UNSET = _Unset()


@dataclass(frozen=True, slots=True)
class AgentCaseInput:
    prompt: str | Sequence[UserContent | WorkspaceFileInput] | _Unset = _UNSET
    capture: AgentInputCaptureRef | None = None
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.prompt is _UNSET) == (self.capture is None):
            raise ValueError("provide exactly one of prompt or capture")
        if self.capture is not None and self.files:
            raise ValueError("captured files cannot be replaced")
        if self.prompt is not _UNSET and not isinstance(self.prompt, str):
            object.__setattr__(self, "prompt", tuple(self.prompt))
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True, slots=True)
class TaskCaseInput:
    input: Mapping[str, JsonValue] | None = None
    capture: TaskInvocationInputRef | None = None
    input_refs: Mapping[str, TaskResultRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.input is None) == (self.capture is None):
            raise ValueError("provide exactly one of input or capture")
        if self.capture is not None and self.input_refs:
            raise ValueError("captured dependencies cannot be replaced")
        if self.input is not None:
            object.__setattr__(self, "input", ImmutableJsonMapping(self.input))
        object.__setattr__(self, "input_refs", MappingProxyType(dict(self.input_refs)))


@dataclass(frozen=True, slots=True)
class GraphCaseInput:
    inputs: Mapping[str, AgentCaseInput | AgentInputCaptureRef | TaskCaseInput]
    source_capture: TaskGraphCaptureRef | None = None
    node_mapping: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.inputs.items():
            _name(name)
            if not isinstance(value, (AgentCaseInput, AgentInputCaptureRef, TaskCaseInput)):
                raise TypeError("graph inputs must be typed invocation inputs")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "node_mapping", MappingProxyType(dict(self.node_mapping)))


@dataclass(frozen=True, slots=True)
class GraphInputContract:
    inputs: Mapping[str, AgentInputCaptureRef | TaskCaseInput]
    source_capture: TaskGraphCaptureRef | None = None
    node_mapping: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(value, (AgentInputCaptureRef, TaskCaseInput)) for value in self.inputs.values()):
            raise TypeError("durable graph input requires captured Agent or normalized Task inputs")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "node_mapping", MappingProxyType(dict(self.node_mapping)))


CaseInput: TypeAlias = AgentCaseInput | TaskCaseInput | GraphCaseInput


@dataclass(frozen=True, slots=True)
class CaseSpec:
    ref: CaseRef
    input: CaseInput
    expected: JsonValue | AssetVersionRef | _Unset = _UNSET
    label_provenance: LabelProvenance = field(default_factory=LabelProvenance)
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.weight)
        object.__setattr__(self, "weight", float(self.weight))
        if not isinstance(self.input, (AgentCaseInput, TaskCaseInput, GraphCaseInput)):
            raise TypeError("case requires a typed Agent, Task, or Graph input")
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def expected_present(self) -> bool:
        return self.expected is not _UNSET

    @classmethod
    def agent(cls, ref: CaseRef, *, prompt: str | Sequence[UserContent | WorkspaceFileInput],
              expected: JsonValue | AssetVersionRef | _Unset = _UNSET,
              files: tuple[str, ...] = (), tags: tuple[str, ...] = (), weight: float = 1.0,
              label_provenance: LabelProvenance | None = None) -> "CaseSpec":
        return cls(ref, AgentCaseInput(prompt=prompt, files=files), expected,
                   label_provenance or LabelProvenance(), tags, weight)

    @classmethod
    def task(cls, ref: CaseRef, *, input: Mapping[str, JsonValue],
             expected: JsonValue | AssetVersionRef | _Unset = _UNSET,
             input_refs: Mapping[str, TaskResultRef] | None = None,
             tags: tuple[str, ...] = (), weight: float = 1.0,
             label_provenance: LabelProvenance | None = None) -> "CaseSpec":
        return cls(ref, TaskCaseInput(input=input, input_refs=input_refs or {}), expected,
                   label_provenance or LabelProvenance(), tags, weight)

    @classmethod
    def graph(cls, ref: CaseRef, *, inputs: Mapping[str, AgentCaseInput | AgentInputCaptureRef | TaskCaseInput],
              expected: JsonValue | AssetVersionRef | _Unset = _UNSET,
              source_capture: TaskGraphCaptureRef | None = None,
              node_mapping: Mapping[str, str] | None = None,
              tags: tuple[str, ...] = (), weight: float = 1.0,
              label_provenance: LabelProvenance | None = None) -> "CaseSpec":
        return cls(ref, GraphCaseInput(inputs, source_capture, node_mapping or {}), expected,
                   label_provenance or LabelProvenance(), tags, weight)

    @classmethod
    def from_capture(cls, ref: CaseRef, *, capture: AgentInputCaptureRef | TaskInvocationInputRef,
                     expected: JsonValue | AssetVersionRef | _Unset = _UNSET,
                     tags: tuple[str, ...] = (), weight: float = 1.0,
                     label_provenance: LabelProvenance | None = None) -> "CaseSpec":
        if isinstance(capture, AgentInputCaptureRef):
            value: CaseInput = AgentCaseInput(capture=capture)
        elif isinstance(capture, TaskInvocationInputRef):
            value = TaskCaseInput(capture=capture)
        else:
            raise TypeError("expected an Agent or Task invocation capture")
        return cls(ref, value, expected, label_provenance or LabelProvenance(), tags, weight)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    ref: DatasetRef
    cases: tuple[CaseSpec | CaseRef, ...]
    selection: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        refs = tuple(case.ref if isinstance(case, CaseSpec) else case for case in self.cases)
        _validate_cases(self.ref, refs)
        object.__setattr__(self, "selection", ImmutableJsonMapping(self.selection))


def _validate_cases(dataset: DatasetRef, refs: tuple[CaseRef, ...]) -> None:
    if not refs or any(ref.dataset_id != dataset.id for ref in refs):
        raise ValueError("dataset requires cases from the same dataset identity")
    if len({ref.case_id for ref in refs}) != len(refs):
        raise ValueError("dataset contains duplicate case identities")


@dataclass(frozen=True, slots=True)
class GraphTargetSpec:
    template: TaskGraphTemplate | None = None
    capture: TaskGraphCaptureRef | None = None
    outputs: Mapping[str, str] = field(default_factory=dict)
    selector: Literal["terminal_sinks"] | None = None

    def __post_init__(self) -> None:
        if (self.template is None) == (self.capture is None):
            raise ValueError("provide exactly one graph template or capture")
        if bool(self.outputs) == (self.selector is not None):
            raise ValueError("provide outputs or terminal_sinks selector")
        if self.selector not in (None, "terminal_sinks"):
            raise ValueError("unknown graph output selector")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    slot_id: str
    task: TaskRef | None = None
    graph_template: GraphTargetSpec | None = None

    def __post_init__(self) -> None:
        _name(self.slot_id)
        if (self.task is None) == (self.graph_template is None):
            raise ValueError("candidate requires exactly one task or graph target")


@dataclass(frozen=True, slots=True)
class DimensionContract:
    name: str
    unit: str
    direction: Literal["higher", "lower"]
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        _name(self.name)
        _name(self.unit)
        if self.direction not in ("higher", "lower"):
            raise ValueError("unknown score direction")
        for value in (self.minimum, self.maximum):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("score limits must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("score limits are inverted")
        if self.minimum is not None:
            object.__setattr__(self, "minimum", float(self.minimum))
        if self.maximum is not None:
            object.__setattr__(self, "maximum", float(self.maximum))

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"name": self.name, "unit": self.unit, "direction": self.direction,
                "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    include_input: bool = True
    include_output: bool = True
    include_trace: bool = False
    include_attachments: bool = False

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"include_input": self.include_input, "include_output": self.include_output,
                "include_trace": self.include_trace, "include_attachments": self.include_attachments}


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    slot_id: str
    task: TaskRef
    dimensions: tuple[DimensionContract, ...]
    rubric: JsonValue | AssetVersionRef | _Unset = _UNSET
    config: Mapping[str, JsonValue] = field(default_factory=dict)
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    required: bool = True
    accepts_target_failure: bool = False
    accepts_target_kinds: tuple[Literal["execution", "graph"], ...] = ("execution", "graph")

    def __post_init__(self) -> None:
        _name(self.slot_id)
        _dimensions(self.dimensions)
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "config", ImmutableJsonMapping(self.config))
        if not self.accepts_target_kinds or any(kind not in ("execution", "graph") for kind in self.accepts_target_kinds):
            raise ValueError("unknown target evidence kind")
        object.__setattr__(self, "accepts_target_kinds", tuple(self.accepts_target_kinds))

    @property
    def rubric_present(self) -> bool:
        return self.rubric is not _UNSET


def _dimensions(values: tuple[DimensionContract, ...]) -> None:
    if not values or len({value.name for value in values}) != len(values):
        raise ValueError("scorer requires uniquely named dimensions")


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    model_mode: Literal["fixture_only", "live_model"] = "fixture_only"
    external_effects: Literal["deny", "read_only", "live"] = "deny"
    max_trials: int = 1000
    target_concurrency: int = 4
    scorer_concurrency: int = 2
    target_graph_limits: TaskGraphLimits = field(default_factory=TaskGraphLimits)
    scorer_graph_limits: TaskGraphLimits = field(default_factory=TaskGraphLimits)
    trial_timeout_seconds: float | None = 300.0
    scorer_timeout_seconds: float | None = 300.0
    human_timeout_seconds: float | None = 86400.0
    environment: Mapping[str, JsonValue] = field(default_factory=dict)
    allow_volatile: bool = False
    token_limit: int | None = None
    cost_limit: str | None = None
    currency: str | None = None
    price_table: AssetVersionRef | None = None
    unknown_usage: Literal["stop", "continue"] = "stop"
    model_fixtures: tuple[Mapping[str, JsonValue], ...] = ()
    content_retention_seconds: float | None = None
    metadata_retention_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.model_mode not in ("fixture_only", "live_model") or self.external_effects not in ("deny", "read_only", "live"):
            raise ValueError("unknown evaluation execution policy")
        for value in (self.max_trials, self.target_concurrency, self.scorer_concurrency):
            _revision(value)
        for name in ("trial_timeout_seconds", "scorer_timeout_seconds", "human_timeout_seconds"):
            value = getattr(self, name)
            if value is not None:
                _positive(value)
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "environment", ImmutableJsonMapping(self.environment))
        object.__setattr__(self, "model_fixtures", tuple(ImmutableJsonMapping(item) for item in self.model_fixtures))
        if self.token_limit is not None:
            if isinstance(self.token_limit, bool) or not isinstance(self.token_limit, int) or self.token_limit < 0:
                raise ValueError("token limit must be a nonnegative integer")
        if self.unknown_usage not in ("stop", "continue"):
            raise ValueError("unknown usage policy")
        if self.cost_limit is not None:
            try:
                amount = Decimal(self.cost_limit)
            except (InvalidOperation, TypeError, ValueError) as error:
                raise ValueError("cost limit must be a decimal string") from error
            if not isinstance(self.cost_limit, str) or not amount.is_finite() or amount < 0:
                raise ValueError("cost limit must be a nonnegative finite decimal string")
            if not self.currency or self.price_table is None:
                raise ValueError("cost limit requires currency and an immutable price table")

        for name in ("content_retention_seconds", "metadata_retention_seconds"):
            value = getattr(self, name)
            if value is not None:
                _positive(value)
                object.__setattr__(self, name, float(value))
        if (self.content_retention_seconds is not None and self.metadata_retention_seconds is not None
                and self.metadata_retention_seconds < self.content_retention_seconds):
            raise ValueError("metadata retention cannot be shorter than content retention")

    @classmethod
    def fixture_only(cls) -> "EvaluationPolicy":
        return cls()

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"model_mode": self.model_mode, "external_effects": self.external_effects,
                "max_trials": self.max_trials, "target_concurrency": self.target_concurrency,
                "scorer_concurrency": self.scorer_concurrency,
                "target_graph_limits": _limits_mapping(self.target_graph_limits),
                "scorer_graph_limits": _limits_mapping(self.scorer_graph_limits),
                "trial_timeout_seconds": self.trial_timeout_seconds,
                "scorer_timeout_seconds": self.scorer_timeout_seconds,
                "human_timeout_seconds": self.human_timeout_seconds,
                "environment": dict(self.environment), "allow_volatile": self.allow_volatile,
                "token_limit": self.token_limit, "cost_limit": self.cost_limit,
                "currency": self.currency,
                "price_table": None if self.price_table is None else self.price_table.to_payload(),
                "unknown_usage": self.unknown_usage,
                "model_fixtures": [dict(item) for item in self.model_fixtures],
                "content_retention_seconds": self.content_retention_seconds,
                "metadata_retention_seconds": self.metadata_retention_seconds}


def _limits_mapping(value: TaskGraphLimits) -> dict[str, JsonValue]:
    return {"max_concurrency": value.max_concurrency, "max_depth": value.max_depth,
            "max_nodes": value.max_nodes, "max_budget": value.max_budget}


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    dataset: DatasetRef
    candidates: tuple[CandidateSpec, ...]
    scorers: tuple[ScorerSpec, ...]
    repetitions: int = 1
    input_mode: Literal["fixed_input", "reproject_input"] = "fixed_input"
    policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)

    def __post_init__(self) -> None:
        _revision(self.repetitions)
        if self.input_mode not in ("fixed_input", "reproject_input"):
            raise ValueError("unknown target input mode")
        for values in (self.candidates, self.scorers):
            if not values or len({value.slot_id for value in values}) != len(values):
                raise ValueError("evaluation requires unique nonempty candidate and scorer slots")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "scorers", tuple(self.scorers))


@dataclass(frozen=True, slots=True)
class StartEvaluationRequest:
    spec: EvaluationSpec
    principal: Principal
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class RescoreRequest:
    scorers: tuple[ScorerSpec, ...]
    idempotency_key: str
    trial_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if not self.scorers or len({value.slot_id for value in self.scorers}) != len(self.scorers):
            raise ValueError("rescore requires unique scorer slots")
        object.__setattr__(self, "scorers", tuple(self.scorers))
        if self.trial_ids is not None:
            if not self.trial_ids or len(set(self.trial_ids)) != len(self.trial_ids):
                raise ValueError("rescore trial selection must be nonempty and unique")
            object.__setattr__(self, "trial_ids", tuple(self.trial_ids))


def capture_mapping(value: AgentInputCaptureRef | TaskInvocationInputRef | TaskGraphCaptureRef | TaskGraphTemplateRef) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"namespace": value.namespace, "tenant_id": value.tenant_id,
                                  "capture_id": value.capture_id, "digest": value.digest}
    if isinstance(value, AgentInputCaptureRef):
        result.update(kind="agent_input", source_execution_id=value.source_execution_id)
    elif isinstance(value, TaskInvocationInputRef):
        result.update(kind="task_input", source_execution_id=value.source_execution_id)
    elif isinstance(value, TaskGraphCaptureRef):
        result.update(kind="graph_input", source_graph_id=value.source_graph_id)
    else:
        result["kind"] = "graph_template"
    return result


def _task_input_mapping(value: TaskCaseInput) -> dict[str, JsonValue]:
    return {"kind": "task_input", "input": None if value.input is None else dict(value.input),
            "capture": None if value.capture is None else capture_mapping(value.capture),
            "input_refs": {name: {"namespace": ref.namespace, "tenant_id": ref.tenant_id,
                                  "graph_id": ref.graph_id, "node_id": ref.node_id,
                                  "result_digest": ref.result_digest}
                           for name, ref in value.input_refs.items()}}


def input_mapping(value: AgentInputCaptureRef | TaskCaseInput | GraphInputContract) -> dict[str, JsonValue]:
    if isinstance(value, AgentInputCaptureRef):
        return capture_mapping(value)
    if isinstance(value, TaskCaseInput):
        return _task_input_mapping(value)
    inputs: dict[str, JsonValue] = {}
    for name, item in value.inputs.items():
        if isinstance(item, TaskCaseInput):
            inputs[name] = _task_input_mapping(item)
        elif isinstance(item, AgentInputCaptureRef):
            inputs[name] = capture_mapping(item)
        else:
            raise ValueError("durable graph Agent input must be captured")
    return {"kind": "graph_input", "inputs": inputs,
            "source_capture": None if value.source_capture is None else capture_mapping(value.source_capture),
            "node_mapping": dict(value.node_mapping)}


@dataclass(frozen=True, slots=True)
class CaseContract:
    ref: CaseRef
    input: AgentInputCaptureRef | TaskCaseInput | GraphInputContract
    expected: EvaluationValue | None = None
    label_provenance: LabelProvenance = field(default_factory=LabelProvenance)
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.weight)
        object.__setattr__(self, "weight", float(self.weight))
        if not isinstance(self.input, (AgentInputCaptureRef, TaskCaseInput, GraphInputContract)):
            raise TypeError("case contract requires resolved input")
        input_mapping(self.input)
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def input_kind(self) -> Literal["agent_input", "task_input", "graph_input"]:
        if isinstance(self.input, AgentInputCaptureRef):
            return "agent_input"
        return "task_input" if isinstance(self.input, TaskCaseInput) else "graph_input"

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"ref": self.ref.to_mapping(), "input": input_mapping(self.input),
                "expected": value_mapping(self.expected), "label_provenance": self.label_provenance.to_mapping(),
                "tags": list(self.tags), "weight": self.weight}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class DatasetContract:
    ref: DatasetRef
    ordered_case_refs: tuple[CaseRef, ...]
    input_kind: Literal["agent_input", "task_input", "graph_input"]
    selection: Mapping[str, JsonValue] = field(default_factory=dict)
    asset_refs: tuple[AssetVersionRef, ...] = ()

    def __post_init__(self) -> None:
        _validate_cases(self.ref, self.ordered_case_refs)
        if self.input_kind not in ("agent_input", "task_input", "graph_input"):
            raise ValueError("unknown dataset input kind")
        object.__setattr__(self, "ordered_case_refs", tuple(self.ordered_case_refs))
        object.__setattr__(self, "asset_refs", tuple(self.asset_refs))
        object.__setattr__(self, "selection", ImmutableJsonMapping(self.selection))

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"ref": self.ref.to_mapping(), "ordered_case_refs": [ref.to_mapping() for ref in self.ordered_case_refs],
                "input_kind": self.input_kind, "selection": dict(self.selection),
                "asset_refs": [ref.to_payload() for ref in self.asset_refs]}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class GraphTargetContract:
    template: TaskGraphTemplate | None
    namespace: str
    tenant_id: str
    outputs: Mapping[str, str]
    selector: Literal["terminal_sinks"] | None
    limits: TaskGraphLimits

    def __post_init__(self) -> None:
        _name(self.namespace)
        _name(self.tenant_id)
        if self.template is not None and self.template.limits != self.limits:
            raise ValueError("graph template limits must match its execution contract")
        if bool(self.outputs) == (self.selector is not None) or self.selector not in (None, "terminal_sinks"):
            raise ValueError("graph contract requires exactly one output selection")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"template": None if self.template is None else self.template.to_mapping(),
                "namespace": self.namespace, "tenant_id": self.tenant_id, "outputs": dict(self.outputs),
                "selector": self.selector, "limits": _limits_mapping(self.limits)}


@dataclass(frozen=True, slots=True)
class CandidateContract:
    slot_id: str
    task: TaskRef | None
    graph_template: GraphTargetContract | None
    definition_contracts: tuple[Mapping[str, JsonValue], ...]

    def __post_init__(self) -> None:
        _name(self.slot_id)
        if (self.task is None) == (self.graph_template is None):
            raise ValueError("candidate requires exactly one target")
        object.__setattr__(self, "definition_contracts", tuple(ImmutableJsonMapping(item) for item in self.definition_contracts))

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"slot_id": self.slot_id,
                "task": None if self.task is None else {"id": self.task.id, "revision": self.task.revision},
                "graph_template": None if self.graph_template is None else self.graph_template.to_mapping(),
                "definition_contracts": [dict(item) for item in self.definition_contracts]}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class ScorerContract:
    slot_id: str
    task: TaskRef
    task_contract: Mapping[str, JsonValue]
    dimensions: tuple[DimensionContract, ...]
    output_contract: Mapping[str, JsonValue]
    rubric: EvaluationValue | None = None
    config: Mapping[str, JsonValue] = field(default_factory=dict)
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    required: bool = True
    accepts_target_failure: bool = False
    accepts_target_kinds: tuple[Literal["execution", "graph"], ...] = ("execution", "graph")
    input_projection: Mapping[str, JsonValue] = field(default_factory=lambda: {"kind": "task", "version": 1})

    def __post_init__(self) -> None:
        _name(self.slot_id)
        _dimensions(self.dimensions)
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "task_contract", ImmutableJsonMapping(self.task_contract))
        object.__setattr__(self, "output_contract", ImmutableJsonMapping(self.output_contract))
        object.__setattr__(self, "config", ImmutableJsonMapping(self.config))
        object.__setattr__(self, "accepts_target_kinds", tuple(self.accepts_target_kinds))
        if not self.accepts_target_kinds or any(kind not in ("execution", "graph") for kind in self.accepts_target_kinds):
            raise ValueError("unknown target evidence kind")
        projection = ImmutableJsonMapping(self.input_projection)
        kind = projection.get("kind")
        expected_keys = {
            "task": {"kind", "version"},
            "agent_projected": {"kind", "version", "builder_task"},
            "agent_literal": {"kind", "version", "instructions", "format"},
        }
        if not isinstance(kind, str) or kind not in expected_keys or set(projection) != expected_keys[kind]:
            raise ValueError("unknown scorer input projection contract")
        if type(projection["version"]) is not int or projection["version"] != 1:
            raise ValueError("unsupported scorer input projection version")
        if kind == "agent_projected" and projection["builder_task"] != {"id": self.task.id, "revision": self.task.revision}:
            raise ValueError("scorer builder must match its named Task")
        if kind == "agent_literal":
            if projection["format"] != "canonical-json" or not isinstance(projection["instructions"], str) or not projection["instructions"].strip():
                raise ValueError("literal scorer requires fixed instructions and canonical JSON")
        object.__setattr__(self, "input_projection", projection)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"slot_id": self.slot_id, "task": {"id": self.task.id, "revision": self.task.revision},
                "task_contract": dict(self.task_contract), "dimensions": [item.to_mapping() for item in self.dimensions],
                "output_contract": dict(self.output_contract), "rubric": value_mapping(self.rubric),
                "config": dict(self.config), "evidence_policy": self.evidence_policy.to_mapping(),
                "required": self.required, "accepts_target_failure": self.accepts_target_failure,
                "accepts_target_kinds": list(self.accepts_target_kinds), "input_projection": dict(self.input_projection)}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class TargetTrialRef:
    target_experiment_id: str
    trial_id: str

    def __post_init__(self) -> None:
        _name(self.target_experiment_id)
        _name(self.trial_id)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"target_experiment_id": self.target_experiment_id, "trial_id": self.trial_id}


@dataclass(frozen=True, slots=True)
class TrialPlan:
    trial_id: str
    case_ref: CaseRef
    candidate_slot_id: str
    repetition: int

    def __post_init__(self) -> None:
        _name(self.trial_id)
        _name(self.candidate_slot_id)
        _revision(self.repetition)

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"trial_id": self.trial_id, "case_ref": self.case_ref.to_mapping(),
                "candidate_slot_id": self.candidate_slot_id, "repetition": self.repetition}


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    experiment_id: str
    kind: Literal["experiment", "score_only"]
    source_experiment_id: str | None
    dataset: DatasetRef
    candidates: tuple[CandidateContract, ...]
    scorers: tuple[ScorerContract, ...]
    trials: tuple[TrialPlan, ...]
    source_trials: tuple[TargetTrialRef, ...]
    policy: EvaluationPolicy
    input_mode: Literal["fixed_input", "reproject_input"]
    principal: Principal

    def __post_init__(self) -> None:
        _name(self.experiment_id)
        if self.kind not in ("experiment", "score_only") or self.input_mode not in ("fixed_input", "reproject_input"):
            raise ValueError("unknown experiment contract kind")
        for name in ("candidates", "scorers", "trials", "source_trials"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for values in (self.candidates, self.scorers):
            if not values or len({value.slot_id for value in values}) != len(values):
                raise ValueError("manifest requires unique nonempty candidate and scorer slots")
        if self.kind == "score_only":
            if not self.source_experiment_id or self.trials or not self.source_trials:
                raise ValueError("score-only manifest requires original trial references")
            if any(ref.target_experiment_id != self.source_experiment_id for ref in self.source_trials):
                raise ValueError("score-only references must belong to one source experiment")
            if len(set(self.source_trials)) != len(self.source_trials):
                raise ValueError("duplicate source trial")
        elif not self.trials or self.source_trials:
            raise ValueError("target experiment requires its own trial plan")
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("duplicate planned trial")
        candidates = {candidate.slot_id for candidate in self.candidates}
        if any(trial.candidate_slot_id not in candidates or trial.case_ref.dataset_id != self.dataset.id
               for trial in self.trials):
            raise ValueError("trial plan refers to an unknown candidate or dataset")

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"experiment_id": self.experiment_id, "kind": self.kind,
                "source_experiment_id": self.source_experiment_id, "dataset": self.dataset.to_mapping(),
                "candidates": [item.to_mapping() for item in self.candidates],
                "scorers": [item.to_mapping() for item in self.scorers],
                "trials": [item.to_mapping() for item in self.trials],
                "source_trials": [item.to_mapping() for item in self.source_trials],
                "policy": self.policy.to_mapping(), "input_mode": self.input_mode,
                "principal": principal_identity_payload(self.principal)}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_mapping())


__all__ = [
    "AgentCaseInput", "CandidateContract", "CandidateSpec", "CaseContract", "CaseInput", "CaseRef", "CaseSpec",
    "DatasetContract", "DatasetRef", "DatasetSpec", "DimensionContract", "EvaluationManifest", "EvaluationPolicy",
    "EvaluationSpec", "EvaluationValue", "EvidencePolicy", "GraphCaseInput", "GraphInputContract", "GraphTargetContract", "GraphTargetSpec",
    "InlineValue", "LabelProvenance", "RescoreRequest", "ScorerContract", "ScorerSpec", "StartEvaluationRequest",
    "TargetTrialRef", "TaskCaseInput", "TrialPlan", "capture_mapping", "input_mapping", "value_mapping",
]
