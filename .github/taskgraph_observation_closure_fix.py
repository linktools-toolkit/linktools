#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "linktools-ai/src/linktools/ai/task/_service_impl.py"

OLD = '''    async def _observe_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[TaskGraphSnapshot]:
        tenant_id = principal.tenant_id
        generation = self._local_activity_generation(graph_id, tenant_id=tenant_id)
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            snapshot = await self._persistence.tasks.snapshot_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if _terminal(snapshot.status):
                await self._request_graph_release(graph_id, tenant_id)
        fingerprint = _observation_fingerprint(snapshot)
        yield snapshot
        if _terminal(snapshot.status):
            return
        while True:
            try:
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )
            except AIError:
                latest = await self._snapshot_with_terminal_release(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if _terminal(latest.status):
                    latest_fingerprint = _observation_fingerprint(latest)
                    if latest_fingerprint != fingerprint or not _terminal(snapshot.status):
                        yield latest
                    return
                raise
            generation = self._local_activity_generation(
                graph_id,
                tenant_id=tenant_id,
            )
            latest = await self._snapshot_with_terminal_release(
                graph_id,
                tenant_id=tenant_id,
            )
            latest_fingerprint = _observation_fingerprint(latest)
            if latest_fingerprint != fingerprint or _terminal(latest.status):
                yield latest
                fingerprint = latest_fingerprint
            if _terminal(latest.status):
                return
            snapshot = latest

'''

NEW = '''    async def _observe_graph(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> AsyncIterator[TaskGraphSnapshot]:
        tenant_id = principal.tenant_id
        generation = self._local_activity_generation(graph_id, tenant_id=tenant_id)
        async with self._graph_consumer(graph_id, tenant_id):
            header = await self._persistence.tasks.get_header(
                graph_id,
                tenant_id=tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(
                principal,
                AuthorizationAction.TASK_READ,
                header,
            )
            snapshot = await self._persistence.tasks.snapshot_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if _terminal(snapshot.status):
                await self._request_graph_release(graph_id, tenant_id)
        snapshot, generation = await self._stabilize_local_terminal(
            snapshot,
            tenant_id=tenant_id,
            generation=generation,
        )
        fingerprint = _observation_fingerprint(snapshot)
        yield snapshot
        if _terminal(snapshot.status):
            return
        while True:
            try:
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )
            except AIError:
                latest = await self._snapshot_with_terminal_release(
                    graph_id,
                    tenant_id=tenant_id,
                )
                if _terminal(latest.status):
                    latest_fingerprint = _observation_fingerprint(latest)
                    if latest_fingerprint != fingerprint or not _terminal(snapshot.status):
                        yield latest
                    return
                raise
            generation = self._local_activity_generation(
                graph_id,
                tenant_id=tenant_id,
            )
            latest = await self._snapshot_with_terminal_release(
                graph_id,
                tenant_id=tenant_id,
            )
            latest, generation = await self._stabilize_local_terminal(
                latest,
                tenant_id=tenant_id,
                generation=generation,
            )
            latest_fingerprint = _observation_fingerprint(latest)
            if latest_fingerprint != fingerprint or _terminal(latest.status):
                yield latest
                fingerprint = latest_fingerprint
            if _terminal(latest.status):
                return
            snapshot = latest

    async def _stabilize_local_terminal(
        self,
        snapshot: TaskGraphSnapshot,
        *,
        tenant_id: str,
        generation: "int | None",
    ) -> "tuple[TaskGraphSnapshot, int | None]":
        while _terminal(snapshot.status):
            waiter = self._local_waiter
            if waiter is None or not waiter.owns_graph(
                snapshot.graph_id,
                tenant_id=tenant_id,
            ):
                return snapshot, generation
            try:
                await self._wait_observation_opportunity(
                    snapshot,
                    tenant_id=tenant_id,
                    after_generation=generation,
                )
            except AIError:
                latest = await self._snapshot_with_terminal_release(
                    snapshot.graph_id,
                    tenant_id=tenant_id,
                )
                if _terminal(latest.status):
                    return latest, self._local_activity_generation(
                        snapshot.graph_id,
                        tenant_id=tenant_id,
                    )
                raise
            generation = self._local_activity_generation(
                snapshot.graph_id,
                tenant_id=tenant_id,
            )
            snapshot = await self._snapshot_with_terminal_release(
                snapshot.graph_id,
                tenant_id=tenant_id,
            )
        return snapshot, generation

'''


def run(*command: str) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    text = SERVICE_PATH.read_text()
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"observation source anchor count={count}")
    SERVICE_PATH.write_text(text.replace(OLD, NEW, 1))

    run("git", "diff", "--check")
    run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
    run(sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt")
    run(sys.executable, "manage.py", "install", "--editable")
    run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/ai/test_task_runtime_recovery.py::test_sqlite_runtime_open_recovers_expired_task_lease",
        "tests/ai/test_task_sqlite_cas_convergence.py::test_sqlite_public_runtime_task_failure_blocks_dependency",
    )
    run(sys.executable, "manage.py", "check", "linktools-ai")
    run(sys.executable, "manage.py", "build", "linktools-ai")
    run(sys.executable, "manage.py", "verify", "linktools-ai")

    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    for path in (
        ".github/taskgraph_observation_closure_fix.py",
        ".github/workflows/taskgraph-observation-closure-fix.yml",
        ".github/workflows/taskgraph-final-verification.yml",
    ):
        if (REPO_ROOT / path).exists():
            run("git", "rm", path)
    run("git", "add", "linktools-ai/src/linktools/ai/task/_service_impl.py")
    run("git", "diff", "--cached", "--check")
    run("git", "commit", "-m", "fix(ai-task): stabilize terminal graph observation")
    run("git", "push", "origin", "HEAD:feat/ai-taskgraph-reliable-mixed-nodes")


if __name__ == "__main__":
    main()
