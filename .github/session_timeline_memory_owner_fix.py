#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        raise RuntimeError(f"{path}: marker already exists")
    target.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


path = "linktools-ai/src/linktools/ai/runtime/state/_steps.py"
replace_once(
    path,
    '''class InMemoryStepArchive(StagingStepStore):
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        super().__init__()
        self._runtime_domain = runtime_domain
''',
    '''class InMemoryStepArchive(StagingStepStore):
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        super().__init__()
        self._runtime_domain = runtime_domain
        self._session_history_owners: dict[str, str] = {}
''',
)
replace_once(
    path,
    '''            for snapshot in snapshots:
                if snapshot not in snapshot_values:
                    snapshot_values.append(snapshot)

    async def materialize_snapshot(
''',
    '''            for snapshot in snapshots:
                if snapshot not in snapshot_values:
                    snapshot_values.append(snapshot)
            if self._runtime_domain is RuntimeDomain.CONVERSATION and any(
                snapshot.state == "complete" for snapshot in snapshots
            ):
                history_id = run.metadata.get("history_id")
                if history_id:
                    self._session_history_owners[history_id] = run.run_id

    async def materialize_snapshot(
''',
)
replace_once(
    path,
    '''    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
''',
    '''    def release_run_local(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        history_id = (
            None
            if run is None or self._runtime_domain is not RuntimeDomain.CONVERSATION
            else run.metadata.get("history_id")
        )
        super().release_run_local(run_id)
        if history_id is not None and self._session_history_owners.get(history_id) == run_id:
            self._session_history_owners.pop(history_id, None)

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
''',
)
replace_once(
    path,
    '''    def _session_snapshot(self, history_id: str) -> ContinuableSnapshot | None:
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        candidates: list[ContinuableSnapshot] = []
        for run in self._runs.values():
            if run.metadata.get("history_id") != history_id:
                continue
            snapshot = self.latest_snapshot_local(run.run_id)
            if snapshot is not None:
                candidates.append(snapshot)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda value: (len(value.messages), value.timestamp, value.run_id),
        )
''',
    '''    def _session_snapshot(self, history_id: str) -> ContinuableSnapshot | None:
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run_id = self._session_history_owners.get(history_id)
        return None if run_id is None else self.latest_snapshot_local(run_id)
''',
)

path = "tests/ai/test_session_timeline_storage.py"
replace_once(
    path,
    '''from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
''',
    '''from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
''',
)
replace_once(
    path,
    '''from linktools.ai.runtime.state import RuntimeState
''',
    '''from linktools.ai.runtime.state import RuntimeDomain, RuntimeState
''',
)
replace_once(
    path,
    '''from linktools.ai.runtime.state._store import StoredFact
''',
    '''from linktools.ai.runtime.state._step_contracts import ContinuableSnapshot, RunRecord
from linktools.ai.runtime.state._steps import InMemoryStepArchive
from linktools.ai.runtime.state._store import StoredFact
''',
)
append_once(
    path,
    "test_in_memory_session_history_uses_latest_materialized_run",
    '''@pytest.mark.asyncio
async def test_in_memory_session_history_uses_latest_materialized_run() -> None:
    archive = InMemoryStepArchive(RuntimeDomain.CONVERSATION)
    await archive.initialize()
    try:
        now = datetime.now(timezone.utc)
        older = RunRecord(
            run_id="older",
            metadata={"history_id": "history"},
            started_at=now,
        )
        latest = RunRecord(
            run_id="latest",
            metadata={"history_id": "history"},
            started_at=now,
        )
        await archive.materialize_snapshot(
            older,
            ContinuableSnapshot(
                run_id="older",
                step_index=1,
                messages=[
                    ModelResponse(parts=[TextPart(content="old-1")]),
                    ModelResponse(parts=[TextPart(content="old-2")]),
                ],
                timestamp=now,
                state="complete",
            ),
        )
        await archive.materialize_snapshot(
            latest,
            ContinuableSnapshot(
                run_id="latest",
                step_index=1,
                messages=[ModelResponse(parts=[TextPart(content="new")])],
                timestamp=now,
                state="complete",
            ),
        )

        assert await archive.session_message_count(
            "history", tenant_id="tenant"
        ) == 1
        messages = [
            message
            async for message in archive.iter_session_message_range(
                "history",
                tenant_id="tenant",
                start=0,
                end=1,
            )
        ]
        assert len(messages) == 1
        assert messages[0] == latest_snapshot_message = ModelResponse(
            parts=[TextPart(content="new")]
        )
        assert latest_snapshot_message.parts[0].content == "new"
    finally:
        await archive.close()
''',
)

print("in-memory timeline owner fix applied")
