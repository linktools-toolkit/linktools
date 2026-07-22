#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DirectEvalExecutor: run an EvalTarget inline through the existing Runtime.

Direct mode does NOT go through JobRuntime -- it resolves the target to a
spec, reads the case's input artifact as the prompt, calls ``Runtime.run``, and
wraps the result in an :class:`EvalExecution`. When the run/definition/event
stores are wired, it also captures a full :class:`RunSnapshot` (run record,
definition, and event stream sealed to artifacts) so the case is replayable and
evaluators can see the trajectory. The Evaluation core depends on the
:class:`EvalExecutor` Protocol, not on JobRuntime."""

import json
import uuid
from typing import Protocol

from ...artifact.models import ArtifactProvenance
from ...json import to_jsonable
from ..models import EvalCase, EvalExecution, EvalTarget, normalize_usage
from ..snapshot import RunSnapshot


class EvalTargetResolver(Protocol):
    """Resolve an :class:`EvalTarget` to a Runtime spec (AgentSpec / SwarmSpec)."""

    async def resolve(self, target: EvalTarget) -> object: ...


class DirectEvalExecutor:
    def __init__(
        self,
        runtime,
        resolver: EvalTargetResolver,
        artifact_store,
        *,
        tenant_id: str,
        user_id: "str | None" = None,
        run_store=None,
        run_definition_store=None,
        event_store=None,
    ) -> None:
        self._runtime = runtime
        self._resolver = resolver
        self._artifact_store = artifact_store
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._run_store = run_store
        self._run_definition_store = run_definition_store
        self._event_store = event_store

    async def execute(self, target: EvalTarget, case: EvalCase) -> EvalExecution:
        try:
            spec = await self._resolver.resolve(target)
            prompt = await self._read_prompt(case)
        except Exception as exc:  # noqa: BLE001 - resolution / input read failure
            return EvalExecution(
                case_id=case.id, run_id=None, output=None, error=type(exc).__name__
            )

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        try:
            result = await self._runtime.run(
                spec,
                prompt,
                run_id=run_id,
                user_id=self._user_id,
                tenant_id=self._tenant_id,
            )
        except Exception as exc:  # noqa: BLE001 - a target that raises is scored, not fatal
            # A PolicyError means the Runtime's security pipeline refused the
            # run -- record it as a safety refusal so the safety_refusal_rate
            # metric is populated in direct mode too. Duck-typed via the MRO so
            # no core/errors import is needed.
            usage = {"safety_refusal": 1.0} if _is_policy_error(exc) else {}
            return EvalExecution(
                case_id=case.id,
                run_id=run_id,
                output=None,
                error=type(exc).__name__,
                model_usage=usage,
            )

        output = getattr(result, "output", result)
        model_usage = normalize_usage(getattr(result, "token_usage", {}) or {})
        output_artifact_id = await self._seal_json(output)
        snapshot_artifact_id, snapshot = await self._capture_snapshot(
            run_id, case, output_artifact_id, model_usage
        )
        return EvalExecution(
            case_id=case.id,
            run_id=run_id,
            output=output,
            output_artifact_id=output_artifact_id,
            model_usage=model_usage,
            snapshot_artifact_id=snapshot_artifact_id,
            snapshot=snapshot,
        )

    async def _read_prompt(self, case: EvalCase) -> str:
        if not case.input_artifact_id:
            return ""
        content = await self._artifact_store.get(
            artifact_id=case.input_artifact_id, tenant_id=self._tenant_id
        )
        return content.decode("utf-8") if content else ""

    async def _seal_json(self, value) -> "str | None":
        """Seal ``value`` into a content-addressed artifact as JSON (best-effort:
        a serialization or store failure returns None). An exotic value the
        generic encoder cannot handle is coerced to its str form so sealing
        never crashes."""
        try:
            text = json.dumps(to_jsonable({"value": value}))
        except TypeError:
            text = json.dumps({"value": str(value), "_value_coerced": True})
        try:
            record = await self._artifact_store.put(
                content=text.encode("utf-8"),
                media_type="application/json",
                tenant_id=self._tenant_id,
                provenance=ArtifactProvenance(producer_kind="eval", producer_id=""),
            )
            return record.ref.id
        except Exception:  # noqa: BLE001 - sealing is best-effort
            return None

    async def _capture_snapshot(
        self, run_id, case, output_artifact_id, model_usage
    ) -> "tuple[str | None, RunSnapshot | None]":
        """Build a full RunSnapshot by sealing the run record, run definition,
        and event stream (stream_id == run_id) to artifacts. Best-effort: if the
        stores aren't wired or any required piece is missing, no snapshot is
        captured (returns None) rather than failing the case."""
        if not (self._run_store and self._run_definition_store and self._event_store):
            return None, None
        try:
            run_record = await self._run_store.get(run_id)
            run_def = await self._run_definition_store.get(run_id)
            if run_record is None or run_def is None:
                return None, None
            run_record_id = await self._seal_json(run_record)
            run_definition_id = await self._seal_json(run_def)
            if run_record_id is None or run_definition_id is None:
                return None, None
            page = await self._event_store.list(run_id, limit=1000)
            event_ids: "list[str]" = []
            for envelope in page.items:
                eid = await self._seal_json(envelope)
                if eid is not None:
                    event_ids.append(eid)
            snapshot = RunSnapshot(
                run_id=run_id,
                run_record_artifact_id=run_record_id,
                run_definition_artifact_id=run_definition_id,
                input_artifact_id=case.input_artifact_id,
                output_artifact_id=output_artifact_id,
                event_artifact_ids=tuple(event_ids),
                asset_snapshots=(),
                task_attempt_id=None,
                model_usage=dict(model_usage or {}),
                metadata={},
            )
            snapshot_id = await self._seal_json(snapshot)
            return snapshot_id, (snapshot if snapshot_id is not None else None)
        except Exception:  # noqa: BLE001 - snapshot capture is best-effort
            return None, None


def _is_policy_error(exc: BaseException) -> bool:
    """True if ``exc`` is a security/policy refusal from the Runtime pipeline
    (a PolicyError or any subclass). Duck-typed by walking the MRO for a class
    named ``PolicyError`` so this module imports no core/errors type."""
    return any(cls.__name__ == "PolicyError" for cls in type(exc).__mro__)


__all__: "list[str]" = ["DirectEvalExecutor", "EvalTargetResolver"]
