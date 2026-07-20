"""Docker-backed execution boundary for untrusted process tools."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

from ...errors import UnsafeExecutionBackendError
from ..protocols import ExecutionIsolationLevel
from ..local import LocalExecutionBackend


class ContainerExecutionBackend:
    """Execute shell commands in a short-lived, constrained Docker container.

    File operations stay in the explicitly mounted workspace; process
    execution never falls back to ``LocalExecutionBackend``.
    """

    isolation_level = ExecutionIsolationLevel.CONTAINER

    def __init__(self, *, runtime_dir: Path, image: str = "python:3.12-slim", timeout_seconds: float = 60.0):
        self.runtime_dir = Path(runtime_dir)
        self.image = image
        self.timeout_seconds = timeout_seconds
        self._files = LocalExecutionBackend(runtime_dir=self.runtime_dir)
        self._processes: set[asyncio.subprocess.Process] = set()

    async def list_dir(self, path=".", recursive=False):
        return await self._files.list_dir(path, recursive)

    async def read_file(self, path, selectors=None, max_chars=6000):
        return await self._files.read_file(path, selectors, max_chars)

    async def write_file(self, path, content=None, updates=None):
        return await self._files.write_file(path, content, updates)

    async def batch_files(self, operations):
        return await self._files.batch_files(operations)

    async def apply_patch(self, diff):
        return await self._files.apply_patch(diff)

    async def run_bash(self, command: str, timeout_ms: int | None = None):
        if shutil.which("docker") is None:
            raise UnsafeExecutionBackendError("Docker is required for ContainerExecutionBackend")
        timeout = (timeout_ms / 1000) if timeout_ms is not None else self.timeout_seconds
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--user", "65532:65532", "--cpus", "1", "--memory", "512m",
            "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,src={self.runtime_dir.resolve()},dst=/workspace,rw",
            "--workdir", "/workspace", self.image, "sh", "-lc", command,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self._processes.add(proc)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": f"timeout after {timeout}s"}
        finally:
            self._processes.discard(proc)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[-16000:],
            "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
        }

    async def fork(self, branch_dir: Path):
        await self._files.fork(branch_dir)
        return ContainerExecutionBackend(runtime_dir=branch_dir, image=self.image, timeout_seconds=self.timeout_seconds)

    async def terminate(self) -> None:
        for proc in tuple(self._processes):
            proc.kill()
        await asyncio.gather(*(proc.wait() for proc in tuple(self._processes)), return_exceptions=True)
        self._processes.clear()
