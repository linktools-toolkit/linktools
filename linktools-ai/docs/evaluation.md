# Evaluation

`runtime.evaluations` evaluates existing Tasks, Agent-backed Tasks, and native
TaskGraphs. Dataset, scoring, evidence, and report values live in
`linktools.ai.evaluation`; execution still uses `runtime.tasks`. There is no
separate candidate or judge runtime.

## An offline evaluation, end to end

This complete example needs no model credentials or network calls. It uses an
empty `ModelRegistry`, durable filesystem storage, two cases, and a rule scorer.
Run it in an environment with `linktools-ai` installed. The state directory is
retained so the same requests can be reopened or retried.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from pathlib import Path

from linktools.ai.core import JsonValue, service_principal
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec,
    DimensionContract, EvaluationSpec, ScorerSpec, ScoringInput,
    ScoreBundle, StartEvaluationRequest, export_report,
)
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeStorage
from linktools.ai.task import Task, TaskNodeContext


async def echo(context: TaskNodeContext[None]) -> JsonValue:
    return context.input["answer"]


async def exact(context: TaskNodeContext[None]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    return ScoreBundle(dimensions={
        "exact_match": float(
            sample.expected_present and sample.target_output == sample.expected
        ),
    }).to_mapping()


async def main() -> None:
    principal = service_principal("example", "evaluation-owner")
    target = Task("example.echo", echo, effect_policy="none")
    judge = Task("example.exact", exact, effect_policy="none")
    dimension = DimensionContract("exact_match", "boolean", "higher", 0, 1)
    scorer = ScorerSpec("exact", judge.ref, (dimension,))

    async with Runtime.open(
        "evaluation-guide",
        models=ModelRegistry(),
        storage=RuntimeStorage.filesystem(Path("evaluation-state")),
        context=RuntimeContext(None, tenant_id="example"),
    ) as runtime:
        engine = runtime.tasks.bind(target, judge)
        dataset = await runtime.evaluations.publish_dataset(
            DatasetSpec(DatasetRef("echo-cases", 1), cases=(
                CaseSpec.task(CaseRef("echo-cases", "text", 1),
                              input={"answer": "yes"}, expected="yes"),
                CaseSpec.task(CaseRef("echo-cases", "null", 1),
                              input={"answer": None}, expected=None),
            )),
            principal=principal,
            idempotency_key="publish-echo-cases-v1",
        )
        request = StartEvaluationRequest(
            EvaluationSpec(dataset, (CandidateSpec("current", task=target.ref),),
                           (scorer,)),
            principal,
            "evaluate-echo-cases-v1",
        )
        run = await runtime.evaluations.start(request, engine=engine)
        view = await run.wait(timeout_seconds=10)
        assert view.completion == "complete", view.needs_attention
        report = await run.report()
        assert report.scores[0].valid == report.scores[0].planned == 2
        assert report.scores[0].mean == report.scores[0].coverage == 1.0
        print(export_report(report, format="markdown"))
        assert await runtime.evaluations.get_report(
            report.report_id, principal=principal
        ) == report


if __name__ == "__main__":
    asyncio.run(main())
```

`completion == "complete"` means orchestration finished, not that quality passed.
Read the report's failures and coverage, or configure a comparison gate.

The same idempotency key and request return the same resource; changed request
semantics under that key raise `IDEMPOTENCY_CONFLICT`. Dataset/case identities
and named Task definitions are immutable at a revision. Bump the appropriate
revision when changing their meaning. Bind Python Tasks again after reopening;
persistence does not serialize Python functions.

Evaluation actions require an authorized principal and enforce tenant/owner
boundaries. The example explicitly uses a service principal; the default Runtime
principal is not authorized for evaluation by default. Production applications
should use their authorization policy and real caller identity.

The remaining snippets are focused extensions: run asynchronous calls inside an
open Runtime and reuse or replace the example's Tasks, principal, and dimension.

## Cases and candidate inputs

`DatasetSpec.cases` is the single ordered collection. Supply `CaseSpec` values or
references to already published `CaseRef` values. A dataset is nonempty, has
unique case IDs, and contains one input kind: Agent, Task, or graph. Do not also
provide a second list of case references.

- `CaseSpec.task(..., input={...})` supplies Task parameters
- `CaseSpec.agent(..., prompt=...)` supplies an Agent prompt, optionally including
  `BinaryContent` or workspace file inputs
- `CaseSpec.graph(..., inputs={node_id: TaskCaseInput(...)})` supplies per-node
  invocation data; Agent nodes use `AgentCaseInput`
- `CaseSpec.from_capture(..., capture=...)` uses immutable historical input

Omitting `expected` means unlabeled. Explicit `expected=None` means the expected
JSON value is null; scorers must check `ScoringInput.expected_present`. Small
expected values and rubrics can be JSON; `AssetVersionRef` pins larger existing
Asset content. Publishing does not transfer ownership of those Assets.

Every candidate has one `task` or `graph_template`. Slot names identify the
candidate within an experiment; they are not global definition identities.
`repetitions` defaults to 1. Planned trials are cases × candidates × repetitions.

### Native graph candidates

The following declarations replace the simple target in the complete example.
Bind all three Tasks to the engine and publish the graph case in its own dataset.
The scorer sees named output records, including their status.

```python
from linktools.ai.evaluation import GraphTargetSpec, TaskCaseInput
from linktools.ai.task import TaskGraphTemplate, TaskNode, TaskNodeResultRef


async def prepare(context: TaskNodeContext[None]) -> JsonValue:
    return context.input["question"].strip().lower()


async def finish(context: TaskNodeContext[None]) -> JsonValue:
    return await context.read_dependency("prepared")


async def score_graph(context: TaskNodeContext[None]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    output = sample.target_output["answer"]
    return ScoreBundle(dimensions={"exact_match": float(
        output["status"] == "succeeded" and output["value"] == sample.expected
    )}).to_mapping()


prepare_task = Task("example.prepare", prepare, effect_policy="none")
finish_task = Task("example.finish", finish, effect_policy="none")
graph_judge = Task("example.graph-exact", score_graph, effect_policy="none")
template = TaskGraphTemplate(nodes=(
    TaskNode("prepare", task=prepare_task),
    TaskNode("finish", ("prepare",), task=finish_task,
             input_refs={"prepared": TaskNodeResultRef("prepare")}),
))
candidate = CandidateSpec("workflow", graph_template=GraphTargetSpec(
    template=template, outputs={"answer": "finish"},
))
case = CaseSpec.graph(CaseRef("graph-cases", "hello", 1),
    inputs={"prepare": TaskCaseInput(input={"question": " HELLO "})},
    expected="hello")
```

Each trial gets a new native graph. Template input and case input may combine
when fields do not conflict; conflicting values are rejected. Scheduling
`dependencies` alone do not provide model-visible data: `input_refs` explicitly
select results. A successful null is `{"status": "succeeded", "value": None,
"reason": None}`; failure is a different status with a reason. Use explicit
`outputs`, or `selector="terminal_sinks"`, but not both.

## Agent candidates and model judges

Open the Runtime with your application's `ModelRegistry` and `CapabilityGroup`.
An existing Agent becomes a candidate or scorer through the same Task adapter:

```python
from linktools.ai.evaluation import EvaluationPolicy

agent_target = runtime.tasks.from_agent("example.agent", runtime.agents.get("answer"))
agent_judge = runtime.tasks.from_agent("example.judge", runtime.agents.get("judge"))
agent_candidate = CandidateSpec("answer", task=agent_target.ref)
model_scorer = ScorerSpec("judge", agent_judge.ref, (dimension,),
                         rubric={"criterion": "Answer matches the reference"})
engine = runtime.tasks.bind(agent_target, agent_judge)
```

The default `EvaluationPolicy` is `model_mode="fixture_only"`,
`external_effects="deny"`. Merely naming a model “test” does not make it a
fixture. An offline model binding must actually materialize an offline model,
and its full semantic contract must be listed explicitly:

```python
policy = EvaluationPolicy(model_fixtures=(fixture_binding.contract,))
```

Register that binding with the ModelRegistry used to open the Runtime. For
intended real provider calls, explicitly choose
`EvaluationPolicy(model_mode="live_model")` and supply an appropriate route.
This permits live model calls and their costs; it does not enable external tool
effects. `read_only` permits only declared read-only tool behavior; `live`
requires an isolated environment. These policies validate declared contracts,
not arbitrary Python side effects inside a falsely declared Task.

An Agent judge receives `ScoringInput` as data, with a fixed instruction to treat
the answer as untrusted, and returns structured `ScoreBundle` output. Do not
pass an unsupported `output_type` argument to `from_agent`; evaluation installs
the scorer node's output contract. For a custom prompt, use the existing
`build_input` callback and decode `ScoringInput.from_mapping(context.input)`
inside it. That callback still runs for each new scoring invocation.

Ordinary scorers return `ScoreBundle(...).to_mapping()`. Every declared dimension
must be present, finite, within its bounds, and use a numeric value or
`ScoreNotApplicable(reason="...")`. Invalid output is a scoring error, not zero.
By default, failed targets produce `not_attempted` scores. A scorer can opt in
with `accepts_target_failure=True` and inspect `sample.target_status`.

Scoring runs as a native `score` → `record` graph. The recorder validates the
score and its evidence references even when the scoring node fails; callers do
not write result records directly.

## Capture historical input or restore original behavior

For a retained Agent execution, capture its accepted input and create a dataset
case. To rerun exactly its original Agent binding, also restore a Task from the
same capture:

```python
from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.runtime import CaptureInputRequest

capture = await runtime.executions.capture_input(
    source_execution_id,
    CaptureInputRequest(principal, "capture-source-v1", context_policy="captured"),
)
assert isinstance(capture, AgentInputCaptureRef)
case = CaseSpec.from_capture(CaseRef("historical-cases", "source", 1),
                             capture=capture, expected="reference answer")
historical = await runtime.tasks.from_agent_capture(
    "example.historical", capture, principal=principal,
)
candidate = CandidateSpec("original", task=historical.ref)
```

Alternatively, use that case with an explicitly chosen current Agent-backed
Task to test changed behavior. Input capture and behavior selection are separate.
`from_agent_capture` restores the original binding, including structured output;
it does not fetch the latest Agent or make the sample input part of its Task
definition. Required historical model/capability contracts must still be
available and compatible. Mixed original bindings need explicit grouping.

- `context_policy="captured"` is the default. New Agent executions retain their
  actual pre-input framework context: session history/metadata, memory content
  and versions, and repository instructions. Reexecution imports that fixed
  context into a new root execution with isolated memory, not the original
  mutable session or production memory scope
- `context_policy="clean"` does not import historical framework context. Fresh
  authored Agent cases are clean. Capture again explicitly to change the policy
- `EvaluationSpec.input_mode="fixed_input"` is the default: reuse the prepared
  historical target input. `"reproject_input"` applies the candidate's input
  projection to retained original parameters. It requires the original input
  contract; unavailable raw file projection is rejected

Standalone Task captures preserve explicit dependency results and failure
states. Their implementation still comes from a Task bound to the engine.
For a graph capture, use:

```python
from linktools.ai.runtime import CaptureGraphRequest

source_graph = await runtime.tasks.capture_graph(
    source_graph_id,
    CaptureGraphRequest(principal, "capture-graph-v1", context_policy="captured"),
)
workflow = CandidateSpec("historical-workflow", graph_template=GraphTargetSpec(
    capture=source_graph, selector="terminal_sinks",
))
```

Graph capture defaults to `mode="declaration_graph"`; materialized graphs use
`mode="materialized_graph"`. Bind the required Task definitions and supply graph
cases. Case node mappings and named outputs are checked. A template's explicit
`TaskNodeResultRef` chooses the new run's dependency result instead of an old
captured result for that alias.

Historical records that lack the needed pre-input context fail with
`INPUT_CONTEXT_UNAVAILABLE`, rather than reading today's session/memory state.
Captures fix retained framework inputs and contracts; they do not snapshot the
whole external world or guarantee deterministic live model/tool responses.

## Retained evidence and attachments

`EvidencePolicy` defaults to input/output included, trace/attachments excluded.
Enable trace or attachments in the initial scorers if later scoring needs them.
A rescore cannot retroactively capture omitted source evidence.

For a scorer with `EvidencePolicy(include_attachments=True)`, retained raw
`BinaryContent` is exposed through typed `EvidenceAttachmentRef` entries. Read
bytes through the authorized evidence facade:

```python
from linktools.ai.evaluation import EvidenceAttachmentRef

sample = ScoringInput.from_mapping(context.input)
evidence = await runtime.evaluations.read_evidence(
    sample.evidence_ref, principal=context.principal,
)
for attachment in evidence.attachments:
    if isinstance(attachment, EvidenceAttachmentRef):
        body = await runtime.evaluations.read_evidence_attachment(
            evidence.ref, attachment.attachment_id, principal=context.principal,
        )
```

The read checks the evidence grant and attachment integrity. `evidence_ids` in a
`ScoreBundle` may cite only IDs available in that scorer's evidence projection.
Asset-backed attachments remain pinned Asset references with independent
ownership and retention.

## Rescore without rerunning targets

Start rescoring from a target-bearing run. Bind only the new scorer Tasks if
that is all they need:

```python
from linktools.ai.evaluation import RescoreRequest

rescored = await run.rescore(
    RescoreRequest((revised_scorer,), "rescore-v2"),
    engine=runtime.tasks.bind(revised_judge),
)
view = await rescored.wait(timeout_seconds=10)
assert view.kind == "score_only"
assert view.source_experiment_id == run.experiment_id
```

`trial_ids=(...)` optionally selects a nonempty set of original trial IDs.
The new run has no new target trials, reuses fixed source evidence, and leaves
initial scores and saved reports unchanged. Cancelling it cancels its scoring,
not the original targets. `RescoreRequest` has no implicit “latest” or round
selection; a score-only run cannot itself be the source of another rescore.

## Compare and gate with explicit score selections

For a run containing `baseline` and `candidate` slots, compare their initial
scores as follows. Declare only intended candidate changes:

```python
from linktools.ai.evaluation import (
    CandidateSlotRef, ComparisonSpec, DimensionBound, GatePolicy,
    ScoreComparisonSelection, ScoreSelection,
)

selection = ScoreComparisonSelection(
    ScoreSelection("exact", "exact_match"),
    ScoreSelection("exact", "exact_match"),
)
comparison = await runtime.evaluations.compare(
    ComparisonSpec(
        CandidateSlotRef(run.experiment_id, "baseline"),
        CandidateSlotRef(run.experiment_id, "candidate"),
        (selection,),
        allowed_changes=("task_definition",),
        gate_policy=GatePolicy(bounds=(DimensionBound(selection, max_regression=0),)),
    ),
    principal=principal,
)
print(comparison.compatibility, comparison.gate, comparison.gate_reasons)
```

Set `ScoreSelection(..., scoring_experiment_id=rescored.experiment_id)` explicitly
to use a rescore. Omission always selects initial scoring. Pairing uses the fixed
case/repetition plan; partial rescoring leaves unselected pairs missing in the
original denominator. Missing, invalid, unavailable, and not-applicable results
are never silently removed or converted to zero.

Reports expose typed candidate/score summaries and pair rows. A score summary's
`planned` equals `pending + valid + not_applicable + error + not_attempted`;
`coverage` is valid/planned. Means use valid values and case weights. Paired
summaries expose complete pairs, one-sided/both missing pairs, and incompatible
pairs. `mean_difference` is candidate minus baseline; gates interpret regression
using the dimension's direction.

Strict comparison validates dataset, inputs, candidate differences, scorer
contracts, and environment policy. Unallowed differences are incompatible.
Exploratory comparisons can diagnose differences but cannot pass a gate.
The default gate requires full paired coverage and at least one distinct case;
it has no quality bound until you supply one. Insufficient evidence, pending
work, or incompatibility makes a configured gate `inconclusive`.

`report()` and `compare()` publish immutable snapshots. Their typed cutoffs pin
manifest/evidence references and observed revisions, not a cross-store atomic
instant. Save `report_id` and retrieve it with `get_report(...)` while retained.
`export_report(..., format="json" | "csv" | "markdown")` exports that snapshot
without querying changing sources.

## Human scores, inspection, and recovery

Use the native deferred-input Task reference as a human scorer:

```python
from linktools.ai.evaluation import HumanScoreRequest
from linktools.ai.task import TaskRef

human = ScorerSpec("human", TaskRef.deferred_input(), (dimension,))
```

Start an evaluation with this scorer, then inspect `await run.scores()`. For the
pending slot, use `scorer_graph`, `scorer_node_id`, and `scorer_execution` to
inspect its native graph with `engine.get(...).state(...)`. Once the deferred
node is `TaskStatus.WAITING`, submit the decision against that slot's exact
evidence reference:

```python
await run.submit_human_score(HumanScoreRequest(
    pending.trial.trial_id,
    pending.scorer_slot_id,
    pending.evidence_ref,
    ScoreBundle(dimensions={"exact_match": 1.0}),
    "review-decision-v1",
))
```

A slot accepts one decision. Equivalent idempotent retries return that decision;
conflicting submissions fail. To revise a completed decision, create a rescore.
The default human wait deadline is 86,400 seconds; configure
`human_timeout_seconds` explicitly when that does not suit the workflow.

- `inspect()` returns progress and safe `needs_attention` issues
- `trials()` and `scores()` return paged typed views. Follow `next_cursor` with
  the same filters and principal; `scores()` includes planned pending and
  unattempted slots
- `wait(timeout_seconds=...)` only bounds the caller's wait. A timeout does not
  cancel the evaluation. Use `cancel(idempotency_key=...)` explicitly
- Reopen with `runtime.evaluations.get(experiment_id, principal=...)`; resume
  coordination with `reconcile(experiment_id, engine=..., principal=...,
  idempotency_key=...)`. Bind matching current definitions first; same-revision
  contract drift is rejected before work can resume

## Storage, limits, and retention

Durable retained storage is required by default. In-memory tests must explicitly
use `EvaluationPolicy(allow_volatile=True)`; transient evidence storage is
rejected. Defaults are 1,000 maximum target trials, target/scorer concurrency
4/2, and per-trial/per-scorer deadlines of 300 seconds. Set these limits for the
workload rather than relying on the example's caller wait timeout.

Optional token and cost limits govern admission of subsequent work from observed
usage; they are not a provider billing cap or a guarantee against in-flight
overshoot. Cost limits require a decimal-string amount, currency, and immutable
`PriceTable` Asset reference. Unknown usage defaults to stopping further work.

No retention duration is imposed by default: both evaluation retention settings
are `None`. An application can choose explicit durations, measured from run
creation, and a separate timezone-aware Dataset publication expiry:

```python
from datetime import datetime, timedelta, timezone
from linktools.ai.evaluation import EvaluationPolicy

policy = EvaluationPolicy(
    content_retention_seconds=7 * 24 * 3600,
    metadata_retention_seconds=7 * 24 * 3600,
)
dataset = await runtime.evaluations.publish_dataset(
    dataset_spec,
    principal=principal,
    idempotency_key="publish-expiring-dataset-v1",
    content_expires_at=datetime.now(timezone.utc) + timedelta(days=14),
)
```

These numbers are application choices, not defaults. Matching content and
metadata deadlines requests cleanup of both operational content and full
evaluation definitions together. If both durations are set, metadata retention
cannot be shorter than content retention. Keep the same absolute publication
expiry when retrying the same idempotency key.

Content expiry ends operational use of retained evaluation evidence, attachment
bytes, score rationale/diagnostics, and saved report bodies. Full definition
metadata, including manifests and their rubric/configuration content, has the
separate metadata lifetime. If you extend that deadline, those full definitions
and their raw values remain until then; content expiry alone is not deletion of
every raw value. Metadata purge removes the full evaluation record rather than keeping a
hidden raw manifest behind an unavailable flag. Minimal tombstones/idempotency
facts remain to prevent accidental recreation.

Dataset/case expiry removes those owned contracts and invalidates dependent
evaluation content. Evaluation expiry does not automatically erase independently
owned datasets, source executions, historical captures, or Assets. Set those
owners' lifetimes separately. Admitted target and scorer TaskGraphs and their
execution histories remain owned by the native Task/Execution domains; evaluation
purge does not shorten those owners' lifetimes. Shared Runtime objects are
retained while another owner still references them. A score-only run cannot
extend source evidence's lifetime.

Expiry checks deny use before physical cleanup. Cleanup is an explicit authorized
maintenance operation, not a background timer:

```python
result = await runtime.evaluations.purge_expired(
    principal=principal,
    now=datetime.now(timezone.utc),
    exclusive=application_guard,
    limit=100,
)
```

`application_guard` must implement the public
`linktools.ai.runtime.state.SnapshotExclusiveGuard` protocol. Its
`offline_exclusivity()` context must actually quiesce every related writer and
object-cleanup process during reachability checking and deletion; a no-op guard
or read-only storage handle is insufficient. Purge first closes admission and
settles native graph cancellation fences. It reports blocked work instead of
deleting beneath an active execution. Object deletion uses durable cleanup
receipts and can be retried after failure/restart. Inspect `blocked` and
`objects_blocked` and retry when their causes are resolved. `objects_retained`
counts shared objects that still have owners, not necessarily a cleanup error.
