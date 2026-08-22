#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary exact patch driver for the remaining local-runtime error closure."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new))


def main() -> None:
    path = "linktools-ai/src/linktools/ai/runtime/_local.py"
    replace_once(
        path,
        '''        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                {"phase": "local_execution_worker"},
            )
''',
        '''        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.INTERNAL_ERROR,
                {"phase": "local_execution_worker"},
            )
''',
    )
    replace_once(
        path,
        '''                if current is not None and current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    await self._commit_failure(current, error, run_id=run_id)
                persisted = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
''',
        '''                if current is not None and current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    try:
                        await self._commit_failure(current, error, run_id=run_id)
                    except BaseException as commit_error:
                        if isinstance(commit_error, asyncio.CancelledError):
                            raise
                        persisted = await self._execution.executions.get(
                            execution_id,
                            tenant_id=original.tenant_id,
                        )
                        if persisted is not None and persisted.status in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            operation_result = _execution_operation_result(persisted.status)
                            _logger.error(
                                "terminal finalization failed after durable execution terminal: execution=%s",
                                execution_id,
                                exc_info=True,
                            )
                            return
                        primary_code = _execution_error_code(error)
                        if isinstance(commit_error, AIError):
                            details = dict(commit_error.safe_details)
                            details["primary_error_code"] = primary_code.value
                            raise AIError(
                                commit_error.code,
                                safe_details=details,
                            ) from error
                        raise AIError(
                            ErrorCode.INTERNAL_ERROR,
                            safe_details={
                                "phase": "execution_terminal_commit",
                                "primary_error_code": primary_code.value,
                            },
                        ) from error
                persisted = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
''',
    )
    replace_once(
        path,
        '''    async def _commit_failure(self, execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        code = ErrorCode.OUTPUT_VALIDATION_FAILED if isinstance(error, ValidationError) else error.code if isinstance(error, AIError) else ErrorCode.EXECUTION_FAILED
        details = error.safe_details if isinstance(error, AIError) else {}
        await self._commit_terminal(
            execution,
            ExecutionStatus.FAILED,
            None,
            code.value,
            StopReason.ERROR,
            run_id=run_id,
            safe_error_details=details,
        )
''',
        '''    async def _commit_failure(self, execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        code = _execution_error_code(error)
        details = error.safe_details if isinstance(error, AIError) else {}
        cancelled = code is ErrorCode.EXECUTION_CANCELLED
        await self._commit_terminal(
            execution,
            ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
            None,
            code.value,
            StopReason.CANCELLED if cancelled else StopReason.ERROR,
            run_id=run_id,
            safe_error_details=details,
        )
''',
    )
    replace_once(
        path,
        '''        recovery_run = None
        recovery_snapshot = None
        if run_id is not None and status is not ExecutionStatus.SUCCEEDED:
            recovery_run = await self._steps.get_run(run_id=run_id)
            recovery_snapshot = await self._steps.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
''',
        '''        recovery_run = None
        recovery_snapshot = None
        if run_id is not None and status is not ExecutionStatus.SUCCEEDED:
            candidate_run = await self._steps.get_run(run_id=run_id)
            candidate_snapshot = await self._steps.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
            if candidate_snapshot is not None:
                if candidate_run is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_run = candidate_run
                recovery_snapshot = candidate_snapshot
''',
    )
    replace_once(
        path,
        '''def _is_infrastructure_error(error: Exception) -> bool:
''',
        '''def _execution_error_code(error: Exception) -> ErrorCode:
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED
    if isinstance(error, AIError):
        return error.code
    return ErrorCode.INTERNAL_ERROR


def _is_infrastructure_error(error: Exception) -> bool:
''',
    )


if __name__ == "__main__":
    main()
