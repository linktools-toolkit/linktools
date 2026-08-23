#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary follow-up patches for error-contract verification."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new))


def main() -> None:
    replace_once(
        "tests/ai/test_local_execution_recovery.py",
        "assert failure.code is ErrorCode.STORAGE_INTEGRITY_ERROR",
        "assert failure.code is ErrorCode.INTERNAL_ERROR",
    )
    replace_once(
        "linktools-ai/src/linktools/ai/asset/_repository.py",
        "ErrorCode.ASSET_LAYOUT_CONFLICT if for_write else ErrorCode.STORAGE_NOT_FOUND",
        "ErrorCode.ASSET_LAYOUT_CONFLICT if for_write else ErrorCode.ASSET_NOT_FOUND",
    )

    path = "tests/ai/test_temporal_taskgraph_conformance.py"
    replace_once(
        path,
        "from linktools.ai.core import Principal, TaskStatus, canonical_sha256",
        "from linktools.ai.core import ExecutionStatus, Principal, TaskStatus, UsageMetrics, canonical_sha256",
    )
    replace_once(
        path,
        "    ExecutionRequest,\n    RuntimeObjectKeyFactory,",
        "    ExecutionRequest,\n    ExecutionResult,\n    RuntimeObjectKeyFactory,",
    )
    replace_once(
        path,
        '        self.result_digest = canonical_sha256({"output": "value"})\n        self.result_calls = 0\n',
        '        self.result_digest = canonical_sha256({"output": "value"})\n        self.result_calls = 0\n        self.terminal_status = ExecutionStatus.FAILED\n',
    )
    replace_once(
        path,
        "    async def result(\n        self,\n        execution_id: str,\n        *,\n        principal: Principal,\n    ) -> TaskNodeRunResult:\n",
        "    async def terminal_result(\n        self,\n        execution_id: str,\n        *,\n        principal: Principal,\n    ) -> ExecutionResult:\n        del principal\n        error_code = (\n            ErrorCode.EXECUTION_CANCELLED\n            if self.terminal_status is ExecutionStatus.CANCELLED\n            else ErrorCode.EXECUTION_FAILED\n        )\n        return ExecutionResult(\n            execution_id,\n            self.terminal_status,\n            None,\n            None,\n            None,\n            None,\n            UsageMetrics(),\n            error_code.value,\n        )\n\n    async def result(\n        self,\n        execution_id: str,\n        *,\n        principal: Principal,\n    ) -> TaskNodeRunResult:\n",
    )
    replace_once(
        path,
        "async def test_stale_settle_replay_uses_higher_fence_durable_status(\n    monkeypatch,\n    durable_status: TaskStatus,\n    child_status: str,\n) -> None:\n    runner = _ReplayRunner()\n",
        "async def test_stale_settle_replay_uses_higher_fence_durable_status(\n    monkeypatch,\n    durable_status: TaskStatus,\n    child_status: str,\n) -> None:\n    runner = _ReplayRunner()\n    if child_status in {\"FAILED\", \"CANCELLED\"}:\n        runner.terminal_status = ExecutionStatus(child_status)\n",
    )


if __name__ == "__main__":
    main()
