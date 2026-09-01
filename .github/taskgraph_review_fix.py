#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "linktools-ai/src/linktools/ai/task/_handler.py",
    "from ..core import JsonValue, Principal, canonical_sha256, normalize_json_value\n",
    "from ..core import (\n"
    "    ImmutableJsonMapping,\n"
    "    JsonValue,\n"
    "    Principal,\n"
    "    canonical_sha256,\n"
    "    normalize_json_value,\n"
    ")\n",
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_handler.py",
    '        object.__setattr__(self, "input", MappingProxyType(normalized_input))\n',
    '        object.__setattr__(self, "input", ImmutableJsonMapping(normalized_input))\n',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_handler.py",
    "@dataclass(frozen=True)\nclass TaskFunction(Generic[AppT]):\n",
    "@dataclass(frozen=True, slots=True)\nclass TaskFunction(Generic[AppT]):\n",
)

replace_once(
    "linktools-ai/src/linktools/ai/task/_graph.py",
    '''        if _aggregate_graph_status(states) is not self.status:\n            raise ValueError("task graph snapshot aggregate status is invalid")\n''',
    '''        aggregate = _aggregate_graph_status(states)\n        if aggregate is not self.status:\n            terminal = {\n                TaskStatus.SUCCEEDED,\n                TaskStatus.FAILED,\n                TaskStatus.BLOCKED,\n                TaskStatus.CANCELLED,\n            }\n            explicit_cancelled = (\n                self.status is TaskStatus.CANCELLED\n                and bool(states)\n                and all(state.status in terminal for state in states)\n                and any(state.status is TaskStatus.CANCELLED for state in states)\n            )\n            if not explicit_cancelled:\n                raise ValueError("task graph snapshot aggregate status is invalid")\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_graph.py",
    '''def _aggregate_graph_status(nodes: "tuple[TaskNodeView, ...]") -> TaskStatus:\n    statuses = {node.status for node in nodes}\n    if not statuses:\n        return TaskStatus.SUCCEEDED\n    terminal = {\n        TaskStatus.SUCCEEDED,\n        TaskStatus.FAILED,\n        TaskStatus.BLOCKED,\n        TaskStatus.CANCELLED,\n    }\n    if any(status not in terminal for status in statuses):\n        if TaskStatus.RUNNING in statuses:\n            return TaskStatus.RUNNING\n        return TaskStatus.PENDING\n    if TaskStatus.CANCELLED in statuses:\n        return TaskStatus.CANCELLED\n    if TaskStatus.FAILED in statuses:\n        return TaskStatus.FAILED\n    if TaskStatus.BLOCKED in statuses:\n        return TaskStatus.BLOCKED\n    return TaskStatus.SUCCEEDED\n''',
    '''def _aggregate_graph_status(nodes: "tuple[TaskNodeView, ...]") -> TaskStatus:\n    statuses = {node.status for node in nodes}\n    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:\n        return TaskStatus.SUCCEEDED\n    if TaskStatus.FAILED in statuses:\n        return TaskStatus.FAILED\n    if TaskStatus.BLOCKED in statuses:\n        return TaskStatus.BLOCKED\n    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:\n        return TaskStatus.CANCELLED\n    if TaskStatus.RUNNING in statuses:\n        return TaskStatus.RUNNING\n    return TaskStatus.PENDING\n''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''        graph = await self._decode(record, TaskGraphView)\n        nodes = await self.list_nodes(graph_id, tenant_id=tenant_id)\n        return TaskGraphView(graph_id, _graph_status(nodes), graph.nodes)\n''',
    '''        graph = await self._decode(record, TaskGraphView)\n        nodes = await self.list_nodes(graph_id, tenant_id=tenant_id)\n        status = (\n            TaskStatus.CANCELLED\n            if graph.status is TaskStatus.CANCELLED\n            else _graph_status(nodes)\n        )\n        return TaskGraphView(graph_id, status, graph.nodes)\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''            return TaskGraphSnapshot(\n                graph.graph_id,\n                _graph_status(ordered),\n                graph.nodes,\n                ordered,\n            )\n''',
    '''            status = (\n                TaskStatus.CANCELLED\n                if graph.status is TaskStatus.CANCELLED\n                else _graph_status(ordered)\n            )\n            return TaskGraphSnapshot(\n                graph.graph_id,\n                status,\n                graph.nodes,\n                ordered,\n            )\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    "            next_status = _graph_status(tuple(next_nodes.values()))\n",
    '''            next_status = (\n                TaskStatus.CANCELLED\n                if graph.status is TaskStatus.CANCELLED\n                else _graph_status(tuple(next_nodes.values()))\n            )\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    '''                if node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}\n''',
    '''                if node.status not in {\n                    TaskStatus.SUCCEEDED,\n                    TaskStatus.FAILED,\n                    TaskStatus.BLOCKED,\n                    TaskStatus.CANCELLED,\n                }\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_repositories.py",
    "            next_status = _graph_status(next_nodes)\n",
    '''            next_status = (\n                TaskStatus.CANCELLED\n                if graph.status is TaskStatus.CANCELLED or changed\n                else _graph_status(next_nodes)\n            )\n''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py",
    "            status_before_cancel = _effective_graph_status(graph, nodes)\n",
    '''            had_nonterminal = any(\n                node.status not in _TERMINAL_TASK_STATUSES\n                for node in nodes\n            )\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py",
    "            if status_before_cancel in _TERMINAL_TASK_STATUSES:\n",
    "            if not had_nonterminal:\n",
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/state/_task_recovery_repository.py",
    '''def _isolated_graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:\n    statuses = {node.status for node in nodes}\n    if not statuses:\n        return TaskStatus.SUCCEEDED\n    if any(status not in _TERMINAL_TASK_STATUSES for status in statuses):\n        if TaskStatus.RUNNING in statuses:\n            return TaskStatus.RUNNING\n        return TaskStatus.PENDING\n    if TaskStatus.FAILED in statuses:\n        return TaskStatus.FAILED\n    if TaskStatus.BLOCKED in statuses:\n        return TaskStatus.BLOCKED\n    if TaskStatus.CANCELLED in statuses:\n        return TaskStatus.CANCELLED\n    return TaskStatus.SUCCEEDED\n''',
    '''def _isolated_graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:\n    statuses = {node.status for node in nodes}\n    if not statuses or statuses <= {TaskStatus.SUCCEEDED}:\n        return TaskStatus.SUCCEEDED\n    if TaskStatus.FAILED in statuses:\n        return TaskStatus.FAILED\n    if TaskStatus.BLOCKED in statuses:\n        return TaskStatus.BLOCKED\n    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:\n        return TaskStatus.CANCELLED\n    if TaskStatus.RUNNING in statuses:\n        return TaskStatus.RUNNING\n    return TaskStatus.PENDING\n''',
)

replace_once(
    "linktools-ai/src/linktools/ai/task/_local.py",
    '''        async with self._lease_state.lock:\n            await self._repository.bind_execution(\n                self._lease_state.lease,\n                tenant_id=self._tenant_id,\n                execution_id=execution_id,\n            )\n            lease = self._lease_state.lease\n''',
    '''        async with self._lease_state.lock:\n            try:\n                await self._repository.bind_execution(\n                    self._lease_state.lease,\n                    tenant_id=self._tenant_id,\n                    execution_id=execution_id,\n                )\n            except AIError as error:\n                if error.code is not ErrorCode.STORAGE_CONFLICT:\n                    raise\n                lease = self._lease_state.lease\n                raise AIError(\n                    ErrorCode.STORAGE_RECOVERY_REQUIRED,\n                    safe_details={\n                        "phase": "task_execution_bind",\n                        "graph_id": lease.graph_id,\n                        "node_id": lease.node_id,\n                    },\n                ) from error\n            lease = self._lease_state.lease\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_local.py",
    '''                if view.status in _TERMINAL:\n                    return\n                _reap_inflight(inflight)\n                now = datetime.now(timezone.utc)\n                static = {node.node_id: node for node in request.graph.nodes}\n                for state in states:\n                    if len(inflight) >= request.limits.max_concurrency:\n                        break\n                    if state.node_id in inflight or not _runnable(state, now):\n                        continue\n''',
    '''                if view.status in _TERMINAL:\n                    if view.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}:\n                        await self._cancel_terminal_effects(run, states)\n                    return\n                _reap_inflight(inflight)\n                now = datetime.now(timezone.utc)\n                persisted = {\n                    state.node_id\n                    for state in states\n                    if state.status is TaskStatus.RUNNING\n                    and state.lease_expires_at is not None\n                    and state.lease_expires_at > now\n                }\n                used = persisted | set(inflight)\n                capacity = max(\n                    0,\n                    request.limits.max_concurrency - len(used),\n                )\n                static = {node.node_id: node for node in request.graph.nodes}\n                for state in states:\n                    if capacity <= 0:\n                        break\n                    if state.node_id in used or not _runnable(state, now):\n                        continue\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_local.py",
    '''                    inflight[node.node_id] = _InflightNode(task, lease_state)\n                    await self._notify(run)\n''',
    '''                    inflight[node.node_id] = _InflightNode(task, lease_state)\n                    used.add(node.node_id)\n                    capacity -= 1\n                    await self._notify(run)\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_local.py",
    '''    async def _wait_scheduler(\n        self,\n        run: _GraphRun,\n        states: "tuple[TaskNodeView, ...]",\n    ) -> None:\n''',
    '''    async def _cancel_terminal_effects(\n        self,\n        run: _GraphRun,\n        states: "tuple[TaskNodeView, ...]",\n    ) -> None:\n        request = run.request\n        static = {node.node_id: node for node in request.graph.nodes}\n        tenant_id = request.principal.tenant_id\n        for state in states:\n            if state.status is not TaskStatus.RUNNING or state.fence < 1:\n                continue\n            node = static.get(state.node_id)\n            if node is None:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            await self._runner.cancel(\n                node,\n                graph_id=request.graph.graph_id,\n                principal=request.principal,\n                dependency_results=await self._dependency_results(\n                    request.graph.graph_id,\n                    node,\n                    tenant_id=tenant_id,\n                ),\n            )\n\n    async def _wait_scheduler(\n        self,\n        run: _GraphRun,\n        states: "tuple[TaskNodeView, ...]",\n    ) -> None:\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_local.py",
    '''                if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:\n                    await self._defer_recovery(run, node, cause_code=error.code)\n                    return\n                code = (\n''',
    '''                if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:\n                    await self._defer_recovery(run, node, cause_code=error.code)\n                    return\n                if isinstance(error, AIError) and error.code in {\n                    ErrorCode.TASK_FENCE_STALE,\n                    ErrorCode.TASK_OWNER_CONFLICT,\n                    ErrorCode.TASK_NOT_READY,\n                }:\n                    return\n                code = (\n''',
)

replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''            if _terminal(snapshot.status):\n                await self._request_graph_release(graph_id, tenant_id)\n        fingerprint = _observation_fingerprint(snapshot)\n''',
    '''            if _terminal(snapshot.status) and (\n                self._local_waiter is None\n                or not self._local_waiter.owns_graph(graph_id, tenant_id=tenant_id)\n            ):\n                await self._request_graph_release(graph_id, tenant_id)\n        snapshot, generation = await self._stabilize_local_terminal(\n            snapshot,\n            tenant_id=tenant_id,\n            generation=generation,\n        )\n        fingerprint = _observation_fingerprint(snapshot)\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''            latest = await self._snapshot_with_terminal_release(\n                graph_id,\n                tenant_id=tenant_id,\n            )\n            latest_fingerprint = _observation_fingerprint(latest)\n''',
    '''            latest = await self._snapshot_with_terminal_release(\n                graph_id,\n                tenant_id=tenant_id,\n            )\n            latest, generation = await self._stabilize_local_terminal(\n                latest,\n                tenant_id=tenant_id,\n                generation=generation,\n            )\n            latest_fingerprint = _observation_fingerprint(latest)\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''            if _terminal(snapshot.status):\n                await self._request_graph_release(graph_id, tenant_id)\n            return snapshot\n\n    def _local_activity_generation(\n''',
    '''            if _terminal(snapshot.status) and (\n                self._local_waiter is None\n                or not self._local_waiter.owns_graph(graph_id, tenant_id=tenant_id)\n            ):\n                await self._request_graph_release(graph_id, tenant_id)\n            return snapshot\n\n    async def _stabilize_local_terminal(\n        self,\n        snapshot: TaskGraphSnapshot,\n        *,\n        tenant_id: str,\n        generation: int | None,\n    ) -> tuple[TaskGraphSnapshot, int | None]:\n        waiter = self._local_waiter\n        while (\n            _terminal(snapshot.status)\n            and waiter is not None\n            and waiter.owns_graph(snapshot.graph_id, tenant_id=tenant_id)\n        ):\n            try:\n                await self._wait_observation_opportunity(\n                    snapshot,\n                    tenant_id=tenant_id,\n                    after_generation=generation,\n                )\n            except AIError:\n                latest = await self._snapshot_with_terminal_release(\n                    snapshot.graph_id,\n                    tenant_id=tenant_id,\n                )\n                if _terminal(latest.status):\n                    return (\n                        latest,\n                        self._local_activity_generation(\n                            snapshot.graph_id,\n                            tenant_id=tenant_id,\n                        ),\n                    )\n                raise\n            generation = self._local_activity_generation(\n                snapshot.graph_id,\n                tenant_id=tenant_id,\n            )\n            snapshot = await self._snapshot_with_terminal_release(\n                snapshot.graph_id,\n                tenant_id=tenant_id,\n            )\n        return snapshot, generation\n\n    def _local_activity_generation(\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''        ) and view.status is TaskStatus.CANCELLED:\n            await self._cleanup_cancelled_graph(graph_id, request)\n        if claimed or operation.status in {\n''',
    '''        ) and view.status is TaskStatus.CANCELLED:\n            try:\n                await self._cleanup_cancelled_graph(graph_id, request)\n            except BaseException as error:  # noqa: BLE001\n                await self._raise_cancel_cleanup_error(\n                    operation,\n                    tenant_id,\n                    graph_id,\n                    error,\n                )\n        if claimed or operation.status in {\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''        if _terminal(view.status):\n            if view.status is TaskStatus.CANCELLED:\n                await self._cleanup_cancelled_graph(graph_id, request)\n            current = await self._record_success(\n''',
    '''        if _terminal(view.status):\n            if view.status is TaskStatus.CANCELLED:\n                try:\n                    await self._cleanup_cancelled_graph(graph_id, request)\n                except BaseException as cleanup_error:  # noqa: BLE001\n                    await self._raise_cancel_cleanup_error(\n                        operation,\n                        tenant_id,\n                        graph_id,\n                        cleanup_error,\n                    )\n            current = await self._record_success(\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/task/_service_impl.py",
    '''    async def _cleanup_cancelled_graph(\n        self,\n        graph_id: str,\n        request: CancelGraphRequest,\n    ) -> None:\n''',
    '''    async def _raise_cancel_cleanup_error(\n        self,\n        operation: OperationLedgerRecord,\n        tenant_id: str,\n        graph_id: str,\n        error: BaseException,\n    ) -> None:\n        await self._record_effect_unknown(operation, tenant_id)\n        if isinstance(error, asyncio.CancelledError):\n            raise error\n        if isinstance(error, AIError):\n            raise error\n        raise AIError(\n            ErrorCode.STORAGE_RECOVERY_REQUIRED,\n            safe_details={"phase": "task_cancel_cleanup", "graph_id": graph_id},\n        ) from error\n\n    async def _cleanup_cancelled_graph(\n        self,\n        graph_id: str,\n        request: CancelGraphRequest,\n    ) -> None:\n''',
)

replace_once(
    "linktools-ai/src/linktools/ai/runtime/_planner.py",
    '''        except asyncio.CancelledError:\n            self._detach(\n                cast("asyncio.Task[object]", task),\n                f"task execution bind graph={key[1]} task={key[2]}",\n            )\n            raise\n\n    async def _bind_after_launch(\n''',
    '''        except asyncio.CancelledError:\n            continuation = asyncio.create_task(\n                self._settle_detached_bind(task, key),\n                name=f"task-execution-bind-settle-{key[1]}-{key[2]}",\n            )\n            self._detach(\n                cast("asyncio.Task[object]", continuation),\n                f"task execution bind graph={key[1]} task={key[2]}",\n            )\n            raise\n\n    async def _settle_detached_bind(\n        self,\n        task: asyncio.Task[None],\n        key: tuple[str, str, str],\n    ) -> None:\n        try:\n            await task\n        except asyncio.CancelledError:\n            raise\n        except AIError as error:\n            if error.code in {\n                ErrorCode.TASK_FENCE_STALE,\n                ErrorCode.TASK_OWNER_CONFLICT,\n                ErrorCode.TASK_NOT_READY,\n            }:\n                return\n            raise self._record_background_failure(\n                key,\n                error,\n                phase="task_execution_bind",\n            ) from error\n        except BaseException as error:  # noqa: BLE001\n            raise self._record_background_failure(\n                key,\n                error,\n                phase="task_execution_bind",\n            ) from error\n\n    async def _bind_after_launch(\n''',
)
replace_once(
    "linktools-ai/src/linktools/ai/runtime/_planner.py",
    '''        except asyncio.CancelledError:\n            raise\n        except BaseException as error:  # noqa: BLE001\n            raise self._record_background_failure(\n                key,\n                error,\n                phase="task_execution_bind_after_launch",\n            ) from error\n''',
    '''        except asyncio.CancelledError:\n            raise\n        except AIError as error:\n            if error.code in {\n                ErrorCode.TASK_FENCE_STALE,\n                ErrorCode.TASK_OWNER_CONFLICT,\n                ErrorCode.TASK_NOT_READY,\n            }:\n                return\n            raise self._record_background_failure(\n                key,\n                error,\n                phase="task_execution_bind_after_launch",\n            ) from error\n        except BaseException as error:  # noqa: BLE001\n            raise self._record_background_failure(\n                key,\n                error,\n                phase="task_execution_bind_after_launch",\n            ) from error\n''',
)

replace_once(
    "tests/ai/test_task_reliable_review_regressions.py",
    "        assert before.status is TaskStatus.PENDING\n",
    "        assert before.status is TaskStatus.FAILED\n",
)

print("TaskGraph review fixes applied")
