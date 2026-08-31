from pathlib import Path
import re


def literal(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: literal anchor count={count}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1))


def regex(path: str, pattern: str, replacement: str) -> None:
    file = Path(path)
    text = file.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.M | re.S)
    if count != 1:
        raise SystemExit(f"{path}: regex anchor count={count}: {pattern[:100]!r}")
    file.write_text(updated)


def class_method(
    path: str,
    class_name: str,
    method: str,
    next_method: str,
    replacement: str,
) -> None:
    file = Path(path)
    text = file.read_text()
    class_start = text.index(f"class {class_name}")
    method_match = re.search(
        rf"^    async def {re.escape(method)}\(",
        text[class_start:],
        flags=re.M,
    )
    if method_match is None:
        raise SystemExit(f"{path}: method not found: {class_name}.{method}")
    left = class_start + method_match.start()
    next_match = re.search(
        rf"^    (?:async )?def {re.escape(next_method)}\(",
        text[left + 1 :],
        flags=re.M,
    )
    if next_match is None:
        raise SystemExit(f"{path}: next method not found: {class_name}.{next_method}")
    right = left + 1 + next_match.start()
    file.write_text(text[:left] + replacement + text[right:])


codec = "linktools-ai/src/linktools/ai/runtime/state/_codec.py"
literal(
    codec,
    "def _decode_v1_task_result(\n",
    '''def _encode_v1_task_result(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskResultRecord):
        raise TypeError("V1 task_result encoder received the wrong type")
    return {
        "graph_id": _encode_domain(value.graph_id, codec, persisted=persisted),
        "node_id": _encode_domain(value.node_id, codec, persisted=persisted),
        "result_digest": _encode_domain(value.result_digest, codec, persisted=persisted),
        "payload": _encode_domain(value.payload, codec, persisted=persisted),
    }


def _decode_v1_task_result(
''',
)
literal(
    codec,
    '''_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {"task_node": _encode_v1_task_node}
)
''',
    '''_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {
        "task_node": _encode_v1_task_node,
        "task_result": _encode_v1_task_result,
    }
)
''',
)
literal(
    codec,
    '    custom_dataclasses = {"task_node"}\n',
    '    custom_dataclasses = {"task_node", "task_result"}\n',
)

aggregate = '''def _aggregate_graph_status(nodes: "tuple[TaskNodeView, ...]") -> TaskStatus:
    statuses = {node.status for node in nodes}
    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if TaskStatus.BLOCKED in statuses:
        return TaskStatus.BLOCKED
    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
        return TaskStatus.CANCELLED
    if TaskStatus.RUNNING in statuses:
        return TaskStatus.RUNNING
    return TaskStatus.PENDING


'''
regex(
    "linktools-ai/src/linktools/ai/task/_graph.py",
    r'^def _aggregate_graph_status\([^\n]*\) -> TaskStatus:\n.*?(?=^@dataclass\(frozen=True, slots=True\)\nclass CancelGraphRequest:)',
    aggregate,
)
regex(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    r'^def _graph_status\([^\n]*\) -> TaskStatus:\n.*?(?=^def _record_cursor\()',
    aggregate.replace("_aggregate_graph_status", "_graph_status"),
)

local = "linktools-ai/src/linktools/ai/task/_local.py"
cancel = '''    async def cancel(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        key = (tenant_id, graph_id)
        view = await self._repository.cancel_graph(graph_id, tenant_id=tenant_id)
        states = await self._repository.list_nodes(graph_id, tenant_id=tenant_id)
        static = {node.node_id: node for node in view.nodes}
        cleanup_error: BaseException | None = None
        for state in states:
            if state.status is not TaskStatus.CANCELLED or state.fence < 1:
                continue
            node = static.get(state.node_id)
            if node is None:
                if cleanup_error is None:
                    cleanup_error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                continue
            try:
                await self._runner.cancel(
                    node,
                    graph_id=graph_id,
                    principal=request.principal,
                    dependency_results=await self._dependency_results(
                        graph_id, node, tenant_id=tenant_id
                    ),
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                if cleanup_error is None:
                    cleanup_error = error
        async with self._lock:
            run = self._runs.get(key)
            if run is not None:
                run.closed = True
                task = run.task
            else:
                task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if run is not None:
            await self._notify(run)
        if cleanup_error is not None:
            if isinstance(cleanup_error, AIError):
                raise cleanup_error
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={"phase": "task_graph_cancel_cleanup", "graph_id": graph_id},
            ) from cleanup_error
        if view.status not in _TERMINAL:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return view

'''
class_method(local, "LocalTaskGraphLauncher", "cancel", "shutdown", cancel)
literal(
    local,
    '''                states = await self._repository.list_nodes(
                    request.graph.graph_id,
                    tenant_id=tenant_id,
                )
                await self._notify(run)
                if view.status in _TERMINAL:
''',
    '''                states = await self._repository.list_nodes(
                    request.graph.graph_id,
                    tenant_id=tenant_id,
                )
                if view.status in _TERMINAL:
''',
)

run_node = '''    async def _run_node(
        self,
        run: _GraphRun,
        node: TaskNode,
        lease_state: _LeaseState,
    ) -> None:
        request = run.request
        graph_id = request.graph.graph_id
        tenant_id = request.principal.tenant_id
        dependency_results = await self._dependency_results(
            graph_id, node, tenant_id=tenant_id
        )
        control = _TaskNodeRunControlImpl(
            self._repository,
            lease_state,
            tenant_id=tenant_id,
            on_activity=lambda: self._notify(run),
        )
        heartbeat_stop = asyncio.Event()
        runner_task = asyncio.create_task(
            self._runner.run(
                node,
                graph_id=graph_id,
                principal=request.principal,
                dependency_results=dependency_results,
                control=control,
            ),
            name=f"task-runner-{graph_id}-{node.node_id}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(lease_state, tenant_id=tenant_id, stop=heartbeat_stop),
            name=f"task-heartbeat-{graph_id}-{node.node_id}",
        )
        try:
            done, _ = await asyncio.wait(
                (runner_task, heartbeat), return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                try:
                    heartbeat.result()
                except asyncio.CancelledError:
                    heartbeat_error: BaseException = AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED
                    )
                except BaseException as error:  # noqa: BLE001
                    heartbeat_error = error
                else:
                    heartbeat_error = AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if not runner_task.done():
                    runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
                if isinstance(heartbeat_error, AIError):
                    if heartbeat_error.code in _RECOVERY_UNKNOWN_CODES:
                        await self._defer_recovery(
                            run, node, cause_code=heartbeat_error.code
                        )
                        return
                    if heartbeat_error.code in {
                        ErrorCode.TASK_FENCE_STALE,
                        ErrorCode.TASK_OWNER_CONFLICT,
                        ErrorCode.TASK_NOT_READY,
                    }:
                        return
                raise heartbeat_error
            try:
                completion = runner_task.result()
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                await _stop_heartbeat(heartbeat_stop, heartbeat)
                if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:
                    await self._defer_recovery(run, node, cause_code=error.code)
                    return
                code = (
                    error.code.value
                    if isinstance(error, AIError)
                    else ErrorCode.TASK_NODE_FAILED.value
                )
                execution_id = (
                    error.execution_id if isinstance(error, TaskNodeRunError) else None
                )
                digest = canonical_sha256(
                    {"graph_id": graph_id, "node_id": node.node_id, "code": code}
                )
                async with lease_state.lock:
                    try:
                        await self._repository.fail(
                            lease_state.lease,
                            tenant_id=tenant_id,
                            error_code=code,
                            error_digest=digest,
                            execution_id=execution_id,
                        )
                    except AIError as terminal_error:
                        if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                            raise
                return
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            async with lease_state.lock:
                try:
                    await self._repository.complete(
                        lease_state.lease,
                        tenant_id=tenant_id,
                        execution_id=completion.execution_id,
                        result_digest=completion.result_digest,
                        result_payload=completion.result_payload,
                    )
                    return
                except AIError as error:
                    if error.code not in _RECOVERY_UNKNOWN_CODES:
                        raise
                    if await self._completion_committed(
                        graph_id,
                        node.node_id,
                        completion,
                        tenant_id=tenant_id,
                    ):
                        return
                    await self._defer_recovery(run, node, cause_code=error.code)
        except asyncio.CancelledError:
            if not runner_task.done():
                runner_task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            raise
        finally:
            await self._notify(run)

'''
class_method(local, "LocalTaskGraphLauncher", "_run_node", "_completion_committed", run_node)

drain = '''    async def _drain_runner_background(self) -> None:
        if not isinstance(self._runner, _RunnerBackgroundOwner):
            return
        while True:
            pending = {
                task
                for task in (
                    *self._runner.pending_background_tasks,
                    *self._runner.pending_cancelled_tasks,
                )
                if not task.done()
            }
            if not pending:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )
        failure = self._runner.background_failure
        if failure is not None:
            raise AIError(failure.code, safe_details=dict(failure.safe_details))

'''
class_method(local, "LocalTaskGraphLauncher", "_drain_runner_background", "_consume_run", drain)

durable = "linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py"
claim = '''    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        before = await self._node(graph_id, node_id, tenant_id)
        expected_fence = before.fence + 1
        try:
            return await super().claim(
                graph_id,
                node_id,
                tenant_id=tenant_id,
                owner=owner,
                lease_seconds=lease_seconds,
            )
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(graph_id, node_id, tenant_id)
            if current.status is TaskStatus.RUNNING:
                if current.owner != owner:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
                if current.fence != expected_fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE) from error
                if current.lease_expires_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return TaskLease(
                    graph_id, node_id, tenant_id, owner, current.fence, current.lease_expires_at
                )
            if current.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                raise AIError(ErrorCode.TASK_NOT_READY) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_claim",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

'''
class_method(durable, "DurableTaskRepositoryImpl", "claim", "complete", claim)

planner = "linktools-ai/src/linktools/ai/runtime/_planner.py"
literal(
    planner,
    "    ExecutionStatus,\n    JsonValue,\n    Principal,\n",
    "    ExecutionMode,\n    ExecutionStatus,\n    JsonValue,\n    Principal,\n    ThinkingValue,\n",
)
literal(
    planner,
    '            mode=cast(str, normalized["mode"]),\n',
    '            mode=cast(ExecutionMode, normalized["mode"]),\n',
)
literal(
    planner,
    '            thinking=normalized["thinking"],\n',
    '            thinking=cast(ThinkingValue, normalized["thinking"]),\n',
)

for name in (
    ".github/workflows/taskgraph-review-p0-fixes.yml",
    ".github/workflows/taskgraph-review-p0-fixes-retry.yml",
    ".github/workflows/taskgraph-final-review.yml",
    ".github/taskgraph_patch.py",
):
    Path(name).unlink(missing_ok=True)
