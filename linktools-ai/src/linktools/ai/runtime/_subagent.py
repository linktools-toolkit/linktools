#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned one-level subagent dispatch and cancellation."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, cast

from linktools.core import environ

from ..agent import AgentBinding, AgentBindingContract, AgentCatalog, AgentCompiler
from ..capability import SubagentCapability, SubagentDelegate
from ..core import (
    ExecutionLineageKind,
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Principal,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ..inputs import stored_user_input_view
from ..model import normalize_input_files
from ..persistence.contracts import ToolOperationRecord
from ..persistence.turns import SessionTurnKey, SqlSessionTurns
from ..persistence.repositories import ExecutionRepositoryImpl, ToolRepositoryImpl
from ..spec import SubagentRef
from ._execution import DefaultExecutionService
from ._handoff import execution_handoff_scope
from ._native_subagent import NativeSubagentAdmission
from ._tool_boundary import _ClaimedLeaf
from ..execution import CancelExecutionRequest, ExecutionRecord, ExecutionRequest, ExecutionResult

_logger = environ.get_logger("ai.runtime.subagent")


class _ChildExecutionObserver(Protocol):
    def publish(self, parent_execution_id: str, child_execution_id: str) -> None: ...


class SubagentDispatcher:
    """Compile and run one-level child executions for a root execution."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        execution: DefaultExecutionService,
        child_observer: _ChildExecutionObserver | None = None,
        restore_binding: Callable[[AgentBindingContract], Awaitable[AgentBinding]] | None = None,
    ) -> None:
        self._catalog = catalog
        self._compiler = compiler
        self._restore_binding = restore_binding
        self._execution = execution
        self._child_observer = child_observer
        self._detached_tasks: set[asyncio.Task[object]] = set()
        self._background_failures: dict[str, AIError] = {}

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return tuple(task for task in self._detached_tasks if not task.done())

    @property
    def background_failure(self) -> "AIError | None":
        if not self._background_failures:
            return None
        failure = next(iter(self._background_failures.values()))
        return AIError(
            failure.code,
            category=failure.category,
            retryable=failure.retryable,
            operation_id=failure.operation_id,
            safe_details=dict(failure.safe_details),
            diagnostics=failure.diagnostics,
        )

    def descriptions_for(
        self,
        refs: "tuple[SubagentRef, ...]",
    ) -> "dict[str, str | None]":
        descriptions: dict[str, str | None] = {}
        for ref in refs:
            if ref.description is not None:
                descriptions[ref.id] = ref.description
        return descriptions

    def delegate_for(
        self,
        *,
        parent_execution_id: str,
        parent_producer_generation: int,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        refs: "tuple[SubagentRef, ...]",
        binding: AgentBindingContract,
        mode: ExecutionMode,
        require_frozen_bindings: bool = False,
    ) -> SubagentDelegate:
        if not isinstance(binding, AgentBindingContract):
            raise TypeError("binding must be AgentBindingContract")
        allowed = {ref.id: ref for ref in refs}
        if len(allowed) != len(refs):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        frozen_children = dict(binding.subagent_binding_map)
        if require_frozen_bindings and set(frozen_children) != set(allowed):
            missing = tuple(sorted(set(allowed).difference(frozen_children)))
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "subagent_binding",
                    "agent_ids": list(missing),
                },
            )
        if frozen_children and set(frozen_children) != set(allowed):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def dispatch(
            ref: SubagentRef,
            task: str,
            *,
            files: tuple[str, ...],
            invocation_id: str,
        ) -> "dict[str, JsonValue]":
            expected = allowed.get(ref.id)
            if expected is None or expected != ref:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            return await self.dispatch(
                parent_execution_id=parent_execution_id,
                parent_producer_generation=parent_producer_generation,
                root_execution_id=root_execution_id,
                memory_scope=memory_scope,
                principal=principal,
                ref=ref,
                frozen_binding=frozen_children.get(ref.id),
                mode=mode,
                user_prompt=task,
                files=files,
                invocation_id=invocation_id,
            )

        return dispatch

    async def recover_native_result(
        self, *, key: SessionTurnKey, original: ToolOperationRecord, args: Mapping[str, JsonValue],
        principal: Principal, binding: AgentBindingContract, mode: ExecutionMode, timeout_seconds: float,
    ) -> tuple[ExecutionRecord, dict[str, JsonValue] | str]:
        """Project a previously established terminal child into its original tool result."""
        agent_id = args.get("subagent_id")
        task = args.get("task")
        if not isinstance(agent_id, str) or not isinstance(task, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        frozen = binding.subagent_binding_map.get(agent_id)
        if frozen is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        files = normalize_input_files(args.get("files", ()))
        invocation_id = original.tool_call_id
        child = await self._execution._load_native_child(
            key=_child_request_key(key.turn_id, invocation_id), principal=principal,
        )
        view = stored_user_input_view(child.stored_user_input)
        captured_files = view.get("files")
        if (child.lineage_kind is not ExecutionLineageKind.SUBAGENT
                or (child.parent_execution_id, child.root_execution_id, child.parent_invocation_id)
                   != (key.turn_id, key.turn_id, invocation_id)
                or (child.principal_id, child.principal_kind) != (principal.principal_id, principal.kind)
                or child.binding_digest != frozen.binding_digest
                or child.mode != ("plan" if mode == "plan" else "run")
                or view.get("prompt") != task.strip()
                or not isinstance(captured_files, list)
                or any(not isinstance(item, Mapping) for item in captured_files)
                or tuple(item.get("path") for item in captured_files) != files):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        child, result = await self._execution._wait_native_child_result(
            child, key=key, principal=principal, original=original, timeout_seconds=timeout_seconds,
        )
        try:
            return child, self._format_child_result(result, agent_id)
        except AIError as error:
            if error.code is not ErrorCode.TOOL_EXECUTION_FAILED:
                raise
            return child, SubagentCapability.format_failure(error, subagent_id=agent_id)

    def claimed_native_leaf(
        self, *, turns: SqlSessionTurns, key: SessionTurnKey, generation: int,
        tools: ToolRepositoryImpl, principal: Principal, binding: AgentBindingContract,
        capability: SubagentCapability, mode: ExecutionMode,
    ) -> _ClaimedLeaf:
        """Bind only the configured native delegate to fresh leaf authority."""
        frozen = binding.subagent_binding_map

        async def invoke(decision, call, args, context):
            if (call.tool_name != "delegate_task" or call.tool_call_id != context.tool_call_id
                    or context.deps.execution_id != key.turn_id or context.deps.principal != principal):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            proof = NativeSubagentAdmission(
                turns, key, generation, tools, decision, call.tool_call_id, principal,
            )

            async def launch(ref: SubagentRef, task: str, *, files: tuple[str, ...], invocation_id: str):
                child_binding = frozen.get(ref.id)
                if child_binding is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self.dispatch(
                    parent_execution_id=key.turn_id, parent_producer_generation=generation,
                    root_execution_id=key.turn_id, memory_scope=context.deps.memory_scope,
                    principal=principal, ref=ref, frozen_binding=child_binding,
                    mode=mode, user_prompt=task, files=files, invocation_id=invocation_id,
                    _native_parent=proof,
                )

            return await capability.invoke_delegate_tool(
                args["subagent_id"], args["task"], files=args.get("files", ()),
                invocation_id=call.tool_call_id, delegate=launch,
            )

        return invoke

    async def dispatch(
        self,
        *,
        parent_execution_id: str,
        parent_producer_generation: int,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        ref: SubagentRef,
        frozen_binding: "AgentBindingContract | None" = None,
        mode: ExecutionMode,
        user_prompt: str,
        files: Sequence[str] = (),
        invocation_id: str,
        _native_parent: NativeSubagentAdmission | None = None,
    ) -> "dict[str, JsonValue]":
        if not isinstance(ref, SubagentRef):
            raise TypeError("ref must be SubagentRef")
        if not isinstance(invocation_id, str) or not invocation_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(files, Sequence) or isinstance(
            files,
            (str, bytes, bytearray),
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        files = tuple(files)
        if any(
            not isinstance(path, str) or not path for path in files
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        child_mode: ExecutionMode = "plan" if mode == "plan" else "run"
        idempotency_key = _child_request_key(parent_execution_id, invocation_id)
        child = await self._dispatch(
            parent_execution_id=parent_execution_id,
            parent_producer_generation=parent_producer_generation,
            root_execution_id=root_execution_id,
            memory_scope=memory_scope,
            principal=principal,
            ref=ref,
            frozen_binding=frozen_binding,
            child_mode=child_mode,
            user_prompt=user_prompt,
            files=files,
            idempotency_key=idempotency_key,
            invocation_id=invocation_id,
            _native_parent=_native_parent,
        )
        if self._child_observer is not None:
            self._child_observer.publish(parent_execution_id, child.execution_id)
        return await self._wait_child(
            child.execution_id,
            parent_execution_id=parent_execution_id,
            principal=principal,
            ref=ref,
            native=_native_parent is not None,
        )

    async def _dispatch(
        self,
        *,
        parent_execution_id: str,
        parent_producer_generation: int,
        root_execution_id: str,
        memory_scope: "str | None",
        principal: Principal,
        ref: SubagentRef,
        frozen_binding: "AgentBindingContract | None",
        child_mode: ExecutionMode,
        user_prompt: str,
        files: tuple[str, ...],
        idempotency_key: str,
        invocation_id: str,
        _native_parent: NativeSubagentAdmission | None = None,
    ):
        if _native_parent is None:
            child = await self._execution.replay_subagent(
                agent_id=ref.id,
                user_prompt=user_prompt,
                files=files,
                principal=principal,
                idempotency_key=idempotency_key,
                memory_scope=memory_scope,
                mode=child_mode,
                parent_execution_id=parent_execution_id,
                parent_producer_generation=parent_producer_generation,
                root_execution_id=root_execution_id,
                parent_invocation_id=invocation_id,
            )
            if child is not None:
                return child
        if frozen_binding is None:
            compiled_agent = self._catalog.root_agent(ref.id)
            child_binding = self._compiler.bind_subagent(compiled_agent)
        else:
            if frozen_binding.agent_spec.id != ref.id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child_binding = (self._compiler.restore(frozen_binding) if self._restore_binding is None
                             else await self._restore_binding(frozen_binding))
            compiled_agent = child_binding.compiled_agent
        child_planning = (
            True
            if child_mode == "plan"
            else compiled_agent.spec.planning
        )
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode=child_mode,
            planning=child_planning,
            thinking=compiled_agent.spec.thinking,
            files=files,
        )
        try:
            return await self._execution.start_subagent(
                child_binding.binding_digest,
                request,
                parent_execution_id=parent_execution_id,
                parent_producer_generation=parent_producer_generation,
                root_execution_id=root_execution_id,
                parent_invocation_id=invocation_id,
                binding_contract=child_binding.binding_contract,
                resolved_binding=child_binding,
                _native_parent=_native_parent,
            )
        except AIError as error:
            if _native_parent is not None or error.code is not ErrorCode.IDEMPOTENCY_CONFLICT:
                raise
        replay = await self._execution.replay_subagent(
            agent_id=ref.id,
            user_prompt=user_prompt,
            files=files,
            principal=principal,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            mode=child_mode,
            parent_execution_id=parent_execution_id,
            parent_producer_generation=parent_producer_generation,
            root_execution_id=root_execution_id,
            parent_invocation_id=invocation_id,
        )
        if replay is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replay

    async def _wait_child(
        self,
        execution_id: str,
        *,
        parent_execution_id: str,
        principal: Principal,
        ref: SubagentRef,
        native: bool = False,
    ) -> "dict[str, JsonValue]":
        scope = execution_handoff_scope.get()
        waiter = asyncio.current_task()
        if scope is not None and waiter is not None:
            scope.register_child_wait(waiter)
        try:
            result = await self._execution.wait(
                execution_id,
                principal=principal,
            )
        except BaseException as primary:  # noqa: BLE001
            if native:
                # The native lifecycle handles explicit parent cancellation.
                # Dropping a wait (including during handoff) cannot cancel a child.
                raise
            if (isinstance(primary, AIError)
                    and primary.safe_details.get("phase") == "turn_owner_lost"):
                # Losing an observer/worker is not authorization to cancel the
                # logical child, which a new parent may explicitly resume.
                raise
            if (
                isinstance(primary, asyncio.CancelledError)
                and scope is not None
                and scope.stopping
                and not scope.cancelled
            ):
                # Worker handoff abandons this observer, not the logical child.
                raise
            cleanup = asyncio.create_task(
                self.cancel_child(
                    execution_id,
                    parent_execution_id=parent_execution_id,
                    principal=principal,
                ),
                name=f"ai-subagent-cleanup-{execution_id}",
            )
            if isinstance(primary, asyncio.CancelledError):
                self._detach(
                    cast("asyncio.Task[object]", cleanup),
                    "subagent child cleanup",
                )
                raise
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                self._detach(
                    cast("asyncio.Task[object]", cleanup),
                    "subagent child cleanup",
                )
                raise
            except BaseException:  # noqa: BLE001
                _logger.exception(
                    "subagent child cleanup failed: execution=%s",
                    execution_id,
                )
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from primary
            raise
        finally:
            if scope is not None and waiter is not None:
                scope.unregister_child_wait(waiter)
        self._background_failures.pop(execution_id, None)
        return self._format_child_result(result, ref.id)

    @staticmethod
    def _format_child_result(result: ExecutionResult, agent_id: str) -> dict[str, JsonValue]:
        if result.status is ExecutionStatus.SUCCEEDED:
            return _subagent_result(result)
        if result.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            details: dict[str, JsonValue] = {
                "phase": "subagent_execution",
                "subagent_id": agent_id,
                "execution_id": result.execution_id,
                "status": result.status.value,
                "safe_error_details": dict(result.safe_error_details),
            }
            if result.error_code is not None:
                details["error_code"] = result.error_code
            raise AIError(
                ErrorCode.TOOL_EXECUTION_FAILED,
                safe_details=details,
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def cancel_native_children(
        self, repository: ExecutionRepositoryImpl, key: SessionTurnKey, admitted: Principal, actor: Principal,
    ) -> None:
        """Drain durable children after the authorized native turn enters cancelling."""
        if admitted.tenant_id != actor.tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        children = await repository.list_children(key.turn_id, tenant_id=admitted.tenant_id)
        if any(
            child.lineage_kind is not ExecutionLineageKind.SUBAGENT
            or child.parent_execution_id != key.turn_id
            or child.root_execution_id != key.turn_id
            or child.principal_id != admitted.principal_id
            or child.principal_kind != admitted.kind
            for child in children
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        for child in sorted(children, key=lambda value: value.execution_id):
            if child.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                self._background_failures.pop(child.execution_id, None)
            else:
                await self.cancel_child(child.execution_id, parent_execution_id=key.turn_id, principal=actor)

    async def cancel_children(
        self,
        parent_execution_id: str,
        principal: Principal,
    ) -> None:
        children = await self._execution.list_children(
            parent_execution_id,
            principal=principal,
        )
        for child in sorted(children, key=lambda value: value.execution_id):
            if child.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                self._background_failures.pop(child.execution_id, None)
                continue
            await self.cancel_child(
                child.execution_id,
                parent_execution_id=parent_execution_id,
                principal=principal,
            )

    async def cancel_child(
        self,
        execution_id: str,
        *,
        parent_execution_id: str,
        principal: Principal,
    ) -> None:
        try:
            current = await self._execution.inspect(execution_id, principal=principal)
        except AIError as error:
            if error.safe_details.get("phase") != "turn_owner_lost":
                raise
            # A failed worker observer must not block explicit source cancellation.
        else:
            if current.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                self._background_failures.pop(execution_id, None)
                return
        request = CancelExecutionRequest(
            principal=principal,
            idempotency_key="subagent-cancel:"
            + canonical_sha256(
                {
                    "version": 1,
                    "parent_execution_id": parent_execution_id,
                    "child_execution_id": execution_id,
                }
            ),
            force=True,
        )
        task = asyncio.create_task(
            self._cancel_child_operation(
                execution_id,
                request,
                principal,
            ),
            name=f"ai-subagent-cancel-{execution_id}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:  # noqa: BLE001
                    _logger.warning(
                        "subagent cancellation failed after caller cancellation: "
                        "execution=%s error=%s",
                        execution_id,
                        type(error).__name__,
                    )
            else:
                self._detach(
                    cast("asyncio.Task[object]", task),
                    "subagent cancellation",
                )
            raise
        except AIError:
            raise
        except BaseException as error:  # noqa: BLE001
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error

    async def _cancel_child_operation(
        self,
        execution_id: str,
        request: CancelExecutionRequest,
        principal: Principal,
    ) -> None:
        try:
            try:
                result = await self._execution.cancel(execution_id, request)
            except asyncio.CancelledError:
                raise
            except AIError:
                raise
            except BaseException as error:  # noqa: BLE001
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            if not result.cancelled:
                current = await self._execution.inspect(
                    execution_id,
                    principal=principal,
                )
                if current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            current = await self._execution.inspect(
                execution_id,
                principal=principal,
            )
            if current.status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            failure = self._record_background_failure(
                execution_id,
                error,
                phase="subagent_cancel_cleanup",
            )
            raise failure from error
        self._background_failures.pop(execution_id, None)

    def _record_background_failure(
        self,
        execution_id: str,
        error: BaseException,
        *,
        phase: str,
    ) -> AIError:
        details = dict(error.safe_details) if isinstance(error, AIError) else {}
        details.setdefault("phase", phase)
        details.setdefault("execution_id", execution_id)
        if isinstance(error, AIError):
            failure = AIError(
                error.code,
                category=error.category,
                retryable=error.retryable,
                operation_id=error.operation_id,
                safe_details=details,
                diagnostics=error.diagnostics,
            )
        else:
            failure = AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details=details,
            )
        self._background_failures[execution_id] = failure
        return failure

    def _detach(
        self,
        task: "asyncio.Task[object]",
        label: str,
    ) -> None:
        if task.done():
            self._consume_done(task, label)
            return
        if task in self._detached_tasks:
            return
        self._detached_tasks.add(task)

        def consume(done: "asyncio.Task[object]") -> None:
            try:
                self._consume_done(done, label)
            finally:
                self._detached_tasks.discard(done)

        task.add_done_callback(consume)

    @staticmethod
    def _consume_done(
        task: "asyncio.Task[object]",
        label: str,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)


def _child_request_key(parent_execution_id: str, invocation_id: str) -> str:
    return canonical_sha256({
        "kind": "subagent", "parent_execution_id": parent_execution_id, "invocation_id": invocation_id,
    })


def _subagent_result(result: ExecutionResult) -> "dict[str, JsonValue]":
    return {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output,
        "error_code": result.error_code,
        "safe_error_details": dict(result.safe_error_details),
    }


__all__ = ["SubagentDispatcher"]
