#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocalExecutionBackend: subprocess + direct file IO on the local filesystem.

Path-escape confinement (_resolve_read_path/_resolve_runtime_path) is an
invariant of this backend, not gated by any feature toggle.
"""

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocalExecutionBackend:
    runtime_dir: Path
    base_dirs: "list[Path]" = field(default_factory=list)
    # Live run_bash subprocesses, keyed by pid. Populated by run_bash_tool as
    # each proc is spawned and removed again once it is awaited (success or
    # timeout), so this only ever contains processes still in flight.
    _subprocesses: "dict[str, asyncio.subprocess.Process]" = field(default_factory=dict)

    async def list_dir(
        self, path: str = ".", recursive: bool = False
    ) -> "dict[str, Any]":
        return await run_file_tool(
            "list_dir",
            {"path": path, "recursive": recursive},
            self.runtime_dir,
            self.base_dirs,
        )

    async def read_file(
        self,
        path: str,
        selectors: "list[str] | None" = None,
        max_chars: int = 6000,
    ) -> "dict[str, Any]":
        return await run_file_tool(
            "read_file",
            {"path": path, "selectors": selectors or [], "max_chars": max_chars},
            self.runtime_dir,
            self.base_dirs,
        )

    async def write_file(
        self,
        path: str,
        content: Any = None,
        updates: "list[dict[str, Any]] | None" = None,
    ) -> "dict[str, Any]":
        return await run_file_tool(
            "write_file",
            {"path": path, "content": content, "updates": updates or []},
            self.runtime_dir,
            self.base_dirs,
        )

    async def batch_files(self, operations: "list[dict[str, Any]]") -> "dict[str, Any]":
        return await run_file_tool(
            "batch_files", {"operations": operations}, self.runtime_dir, self.base_dirs
        )

    async def run_bash(
        self, command: str, timeout_ms: "int | None" = None
    ) -> "dict[str, Any]":
        resolved = command
        if self.base_dirs:
            resolved = await asyncio.to_thread(
                resolve_base_file_paths, command, self.base_dirs
            )
        args: "dict[str, Any]" = {"command": resolved}
        if timeout_ms is not None:
            args["timeout_ms"] = timeout_ms
        return await run_bash_tool(
            args, self.runtime_dir, timeout_s=60.0, registry=self._subprocesses
        )

    async def apply_patch(self, diff: str) -> "dict[str, Any]":
        raw_paths = _extract_patch_target_paths(diff)
        if not raw_paths:
            return {"error": "no file headers found in patch"}
        for raw_path in raw_paths:
            try:
                _resolve_runtime_path(_strip_patch_prefix(raw_path), self.runtime_dir)
            except ValueError as exc:
                return {"error": str(exc)}
        await asyncio.to_thread(self.runtime_dir.mkdir, parents=True, exist_ok=True)
        return await run_patch_tool(
            diff, self.runtime_dir, timeout_s=APPLY_PATCH_TIMEOUT_S
        )

    async def fork(self, branch_dir: Path) -> "LocalExecutionBackend":
        await asyncio.to_thread(branch_dir.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree, self.runtime_dir, branch_dir, dirs_exist_ok=True
        )
        return LocalExecutionBackend(runtime_dir=branch_dir, base_dirs=self.base_dirs)

    async def terminate(self) -> None:
        # SIGKILL every tracked subprocess still in flight, then reap it. A
        # process that already exited between the run_bash hot path and here
        # surfaces as ProcessLookupError on kill(), which we swallow.
        for proc in list(self._subprocesses.values()):
            try:
                proc.kill()
            except ProcessLookupError:
                continue
            try:
                await proc.wait()
            except Exception:
                pass
        self._subprocesses.clear()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

# apply_patch has no caller-configurable timeout_ms (unlike run_bash), so the
# protection is a fixed 60s cap matching run_bash's default.
APPLY_PATCH_TIMEOUT_S = 60.0


async def run_patch_tool(
    diff: str, runtime_dir: Path, timeout_s: float
) -> "dict[str, Any]":
    proc = await asyncio.create_subprocess_exec(
        "patch",
        "-p1",
        "--no-backup-if-mismatch",
        cwd=str(runtime_dir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(diff.encode("utf-8")), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"error": f"timeout after {timeout_s}s"}
    if proc.returncode != 0:
        return {
            "error": "patch failed",
            "detail": (stdout + stderr).decode("utf-8", errors="replace"),
        }
    return {"ok": True, "output": stdout.decode("utf-8", errors="replace")}


async def run_bash_tool(
    args: "dict[str, Any]",
    runtime_dir: Path,
    timeout_s: float,
    registry: "dict[str, asyncio.subprocess.Process] | None" = None,
) -> "dict[str, Any]":
    command = str(args.get("command", ""))
    if not command:
        return {"error": "missing 'command'"}
    try:
        timeout = float(args.get("timeout_ms", timeout_s * 1000)) / 1000
    except (TypeError, ValueError):
        timeout = timeout_s
    env = {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "LC_ALL"}
    }
    await asyncio.to_thread(runtime_dir.mkdir, parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(runtime_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    # Register the live proc so LocalExecutionBackend.terminate() can reap it
    # while communicate() is still awaiting. Removed in finally on both the
    # success and timeout paths so the registry only holds in-flight procs.
    key = str(proc.pid)
    if registry is not None:
        registry[key] = proc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"error": f"timeout after {timeout}s"}
    finally:
        if registry is not None:
            registry.pop(key, None)
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace")[-16000:],
        "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
    }


async def run_file_tool(
    tool_name: str,
    args: "dict[str, Any]",
    runtime_dir: Path,
    base_dirs: "list[Path]",
) -> "dict[str, Any]":
    return await asyncio.to_thread(
        _run_file_tool_sync, tool_name, args, runtime_dir, base_dirs
    )


def _run_file_tool_sync(
    tool_name: str,
    args: "dict[str, Any]",
    runtime_dir: Path,
    base_dirs: "list[Path]",
) -> "dict[str, Any]":
    rel_path = str(args.get("path") or ("." if tool_name == "list_dir" else ""))
    if (
        tool_name not in {"read_files", "write_files", "read_jsons", "batch_files"}
        and not rel_path
    ):
        return {"error": "missing 'path'"}

    if tool_name == "list_dir":
        try:
            target, _, display_path = _resolve_read_path(
                rel_path, runtime_dir, base_dirs
            )
        except ValueError as exc:
            return {"error": str(exc)}
        if not target.exists():
            return {"error": f"directory not found: {rel_path}"}
        if not target.is_dir():
            return {"error": f"not a directory: {rel_path}", "path": display_path}
        recursive = bool(args.get("recursive"))
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries = []
        for item in sorted(iterator, key=lambda p: p.relative_to(target).as_posix()):
            entries.append(
                {
                    "path": item.relative_to(target).as_posix(),
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(entries) >= 500:
                break
        return {
            "entries": entries,
            "truncated": len(entries) >= 500,
            "path": display_path,
            "absolute_path": str(target),
        }

    if tool_name == "read_file":
        selectors = (
            args.get("selectors") if isinstance(args.get("selectors"), list) else []
        )
        if selectors:
            return _run_file_tool_sync(
                "read_json",
                {"path": rel_path, "selectors": selectors},
                runtime_dir,
                base_dirs,
            )
        try:
            target, _, display_path = _resolve_read_path(
                rel_path, runtime_dir, base_dirs
            )
        except ValueError as exc:
            return {"error": str(exc)}
        if not target.exists():
            return {"error": f"file not found: {rel_path}"}
        if not target.is_file():
            return {"error": f"not a file: {rel_path}", "path": display_path}
        content = target.read_text(encoding="utf-8")
        max_chars = _positive_int(args.get("max_chars"), default=6000)
        truncated = bool(max_chars and len(content) > max_chars)
        return {
            "content": content[:max_chars] if truncated else content,
            "truncated": truncated,
            "original_chars": len(content),
            "path": display_path,
            "absolute_path": str(target),
        }

    if tool_name == "read_files":
        raw_paths = args.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            return {"error": "missing 'paths'"}
        max_chars_each = _positive_int(args.get("max_chars_each"), default=4000)
        files: "list[dict[str, Any]]" = []
        for raw_path in raw_paths[:50]:
            try:
                target, _, display_path = _resolve_read_path(
                    str(raw_path), runtime_dir, base_dirs
                )
            except ValueError as exc:
                files.append({"path": str(raw_path), "error": str(exc)})
                continue
            if not target.exists():
                files.append(
                    {"path": str(raw_path), "error": f"file not found: {raw_path}"}
                )
                continue
            if not target.is_file():
                files.append(
                    {
                        "path": display_path,
                        "error": f"not a file: {raw_path}",
                        "absolute_path": str(target),
                    }
                )
                continue
            content = target.read_text(encoding="utf-8")
            truncated = bool(max_chars_each and len(content) > max_chars_each)
            files.append(
                {
                    "path": display_path,
                    "absolute_path": str(target),
                    "content": content[:max_chars_each] if truncated else content,
                    "truncated": truncated,
                    "original_chars": len(content),
                }
            )
        return {
            "files": files,
            "requested_count": len(raw_paths),
            "returned_count": len(files),
        }

    if tool_name == "read_json":
        try:
            target, _, display_path = _resolve_read_path(
                rel_path, runtime_dir, base_dirs
            )
        except ValueError as exc:
            return {"error": str(exc)}
        if not target.exists():
            return {"error": f"file not found: {rel_path}"}
        if not target.is_file():
            return {"error": f"not a file: {rel_path}", "path": display_path}
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "error": f"invalid json: {display_path}",
                "detail": str(exc),
                "path": display_path,
            }
        selectors = args.get("selectors") or []
        if selectors and not isinstance(selectors, list):
            return {"error": "selectors must be a list"}
        if selectors:
            selected: "dict[str, Any]" = {}
            missing: "list[str]" = []
            for raw_selector in selectors[:50]:
                selector = str(raw_selector).strip()
                if not selector:
                    continue
                found, value = _json_select(payload, selector)
                if found:
                    selected[selector] = value
                else:
                    missing.append(selector)
            return {
                "path": display_path,
                "absolute_path": str(target),
                "selected": selected,
                "missing": missing,
                "selector_count": len(selectors),
            }
        return {
            "path": display_path,
            "absolute_path": str(target),
            "content": payload,
        }

    if tool_name == "read_jsons":
        raw_files = args.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            return {"error": "missing 'files'"}
        results: "list[dict[str, Any]]" = []
        for item in raw_files[:50]:
            if not isinstance(item, dict):
                results.append({"error": "file item must be an object"})
                continue
            path_value = str(item.get("path") or "").strip()
            if not path_value:
                results.append({"error": "file item missing 'path'"})
                continue
            selectors = item.get("selectors")
            result = _run_file_tool_sync(
                "read_json",
                {
                    "path": path_value,
                    "selectors": selectors if isinstance(selectors, list) else [],
                },
                runtime_dir,
                base_dirs,
            )
            if "path" not in result:
                result["path"] = path_value
            results.append(result)
        return {
            "files": results,
            "requested_count": len(raw_files),
            "returned_count": len(results),
        }

    if tool_name == "write_file":
        raw_updates = args.get("updates")
        if isinstance(raw_updates, list) and raw_updates:
            return _run_file_tool_sync(
                "update_json",
                {"path": rel_path, "updates": raw_updates},
                runtime_dir,
                base_dirs,
            )
        content = args.get("content")
        if not isinstance(content, str):
            return _run_file_tool_sync(
                "write_json",
                {"path": rel_path, "content": content},
                runtime_dir,
                base_dirs,
            )
        try:
            target = _resolve_runtime_path(rel_path, runtime_dir)
        except ValueError as exc:
            return {"error": str(exc)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content or ""), encoding="utf-8")
        return {"ok": True, "path": rel_path, "absolute_path": str(target)}

    if tool_name == "write_files":
        raw_files = args.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            return {"error": "missing 'files'"}
        written: "list[dict[str, Any]]" = []
        for item in raw_files[:50]:
            if not isinstance(item, dict):
                written.append({"error": "file item must be an object"})
                continue
            path_value = str(item.get("path") or "")
            if not path_value:
                written.append({"error": "file item missing 'path'"})
                continue
            try:
                target = _resolve_runtime_path(path_value, runtime_dir)
            except ValueError as exc:
                written.append({"path": path_value, "error": str(exc)})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(item.get("content", "")), encoding="utf-8")
            written.append(
                {"ok": True, "path": path_value, "absolute_path": str(target)}
            )
        return {
            "files": written,
            "requested_count": len(raw_files),
            "written_count": sum(1 for item in written if item.get("ok")),
        }

    if tool_name == "write_json":
        try:
            target = _resolve_runtime_path(rel_path, runtime_dir)
        except ValueError as exc:
            return {"error": str(exc)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json_dump(args.get("content")), encoding="utf-8")
        return {"ok": True, "path": rel_path, "absolute_path": str(target)}

    if tool_name == "update_json":
        raw_updates = args.get("updates")
        if not isinstance(raw_updates, list) or not raw_updates:
            return {"error": "missing 'updates'"}
        try:
            target = _resolve_runtime_path(rel_path, runtime_dir)
        except ValueError as exc:
            return {"error": str(exc)}
        current: Any = {}
        if target.exists():
            try:
                current = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return {"error": f"invalid json: {rel_path}", "detail": str(exc)}
        elif any(
            str(item.get("path") or "").strip() == ""
            for item in raw_updates
            if isinstance(item, dict)
        ):
            current = {}
        else:
            current = {}
        for item in raw_updates[:100]:
            if not isinstance(item, dict):
                return {"error": "update item must be an object"}
            selector = str(item.get("path") or "").strip()
            if not selector:
                return {"error": "update item missing 'path'"}
            current = _json_assign(current, selector, item.get("value"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_json_dump(current), encoding="utf-8")
        return {
            "ok": True,
            "path": rel_path,
            "absolute_path": str(target),
            "updated_count": min(len(raw_updates), 100),
        }

    if tool_name == "batch_files":
        raw_ops = args.get("operations")
        if not isinstance(raw_ops, list) or not raw_ops:
            return {"error": "missing 'operations'"}
        results: "list[dict[str, Any]]" = []
        for item in raw_ops[:50]:
            if not isinstance(item, dict):
                results.append({"error": "operation must be an object"})
                continue
            action = str(item.get("action") or "").strip().lower()
            if action == "read":
                op_path = str(item.get("path") or "").strip()
                op_selectors = (
                    item.get("selectors")
                    if isinstance(item.get("selectors"), list)
                    else []
                )
                op_max_chars = item.get("max_chars", 4000)
                result = _run_file_tool_sync(
                    "read_file",
                    {
                        "path": op_path,
                        "selectors": op_selectors,
                        "max_chars": op_max_chars,
                    },
                    runtime_dir,
                    base_dirs,
                )
            elif action == "write":
                result = _run_file_tool_sync(
                    "write_file",
                    {"path": item.get("path"), "content": item.get("content")},
                    runtime_dir,
                    base_dirs,
                )
            elif action == "update":
                result = _run_file_tool_sync(
                    "write_file",
                    {"path": item.get("path"), "updates": item.get("updates")},
                    runtime_dir,
                    base_dirs,
                )
            else:
                result = {"error": f"unknown action: {action or '<empty>'}"}
            results.append(result)
        return {
            "operations": results,
            "requested_count": len(raw_ops),
            "completed_count": len(results),
        }

    return {"error": f"unknown file tool: {tool_name}"}


def resolve_base_file_paths(command: str, base_dirs: "list[Path]") -> str:
    for base_dir in base_dirs:
        base_dir = base_dir.resolve()
        files = (
            f
            for f in base_dir.rglob("*")
            if f.is_file()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in f.relative_to(base_dir).parts
            )
        )
        for file_path in sorted(files, key=lambda p: len(p.parts), reverse=True):
            # Keep the trace materialized path instead of resolving the symlink target
            # back to the source capability tree; otherwise relative_to(base_dir) fails
            # for builtin files materialized as symlinks under workspace/TRC.../capabilities.
            abs_path = str(file_path)
            rel_to_base = file_path.relative_to(base_dir).as_posix()
            for candidate in sorted(
                {rel_to_base, f"./{rel_to_base}", file_path.name}, key=len, reverse=True
            ):
                command = re.sub(
                    rf"(?<![/\w.$]){re.escape(candidate)}(?![\w/])",
                    abs_path,
                    command,
                )
    return command


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _extract_patch_target_paths(diff: str) -> "list[str]":
    paths = []
    for line in diff.splitlines():
        match = re.match(r"^(?:---|\+\+\+)\s+(\S+)", line)
        if match and match.group(1) != "/dev/null":
            paths.append(match.group(1))
    return paths


def _strip_patch_prefix(path: str) -> str:
    # `-p1` semantics: `patch` unconditionally strips the first `/`-separated
    # path component of every header before use -- regardless of what that
    # component is named (it does NOT special-case git's "a/"/"b/" convention).
    # The escape check must therefore strip unconditionally too, so it
    # validates what `patch` will actually touch, not the raw header text.
    # Verified empirically: a header like "zzz/capabilities/x.txt" is written
    # by real `patch -p1` to "capabilities/x.txt", stripping "zzz/" just like
    # it would strip "a/". A header with no "/" at all (e.g. "bare.txt") has
    # nothing to strip; `patch -p1` itself refuses to apply such a hunk
    # ("can't find file to patch") and exits non-zero without writing, so
    # returning it unstripped here is safe -- it will never reach disk.
    head, sep, rest = path.partition("/")
    if not sep:
        return path
    return rest


def _resolve_runtime_path(rel_path: str, runtime_dir: Path) -> Path:
    base = runtime_dir.resolve()
    path_text = rel_path.strip()
    if Path(path_text).is_absolute():
        raise ValueError(f"write_file path must be relative to runtime: {rel_path}")
    for forbidden in ("base/", "capabilities/"):
        if path_text == forbidden.rstrip("/") or path_text.startswith(forbidden):
            raise ValueError(
                f"write_file cannot write to capability/base paths: {rel_path}"
            )
    if path_text == "runtime":
        path_text = "."
    elif path_text.startswith("runtime/"):
        path_text = path_text[len("runtime/") :]
    target = (base / path_text).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"Path escape denied: {rel_path}")
    return target


def _is_within_root(target: Path, root: Path) -> bool:
    target = target.resolve()
    root = root.resolve()
    return target == root or root in target.parents


def _resolve_existing_confined_path(
    candidate: Path,
    *,
    allowed_roots: "tuple[Path, ...]",
    raw_path: str,
) -> Path:
    """Resolve ``candidate`` following symlinks, then require the real target to
    lie within one of ``allowed_roots``. A symlink whose target escapes every
    allowed root is denied here -- the lexical position of the link itself is
    irrelevant. ``allowed_roots`` members must already be resolved."""
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"path not found: {raw_path}") from exc
    if not any(_is_within_root(resolved, root) for root in allowed_roots):
        raise ValueError(f"path escape denied: {raw_path}")
    return resolved


def _resolve_read_path(
    raw_path: str,
    runtime_dir: Path,
    base_dirs: "list[Path]",
) -> "tuple[Path, str, str]":
    roots: "list[tuple[str, Path]]" = [("runtime", runtime_dir.resolve())]
    for i, base_dir in enumerate(base_dirs):
        roots.append(
            ("agent_dir" if i == 0 else f"agent_dir{i + 1}", base_dir.resolve())
        )
    allowed_roots = tuple(root for _, root in roots)

    path_text = raw_path.strip()
    prefixed_root = ""
    for name, _ in roots:
        prefix = f"{name}/"
        if path_text == name:
            prefixed_root, path_text = name, "."
            break
        if path_text.startswith(prefix):
            prefixed_root, path_text = name, path_text[len(prefix) :]
            break

    candidates = (
        [(n, b) for n, b in roots if n == prefixed_root] if prefixed_root else roots
    )
    if prefixed_root and not candidates:
        raise ValueError(f"read root unavailable: {prefixed_root}")

    raw_candidate = Path(path_text).expanduser()

    def _logical_target(base: Path) -> Path:
        # Normalize "."/".." lexically without resolving the final component; the
        # strict resolve + confinement below follows symlinks and rejects any
        # whose real target leaves the allowed roots.
        joined = raw_candidate if raw_candidate.is_absolute() else base / raw_candidate
        return Path(os.path.normpath(joined))

    last_error: "ValueError | None" = None
    for name, base in candidates:
        logical = _logical_target(base)
        # Cheap lexical guard before touching disk: a ".."-escape from `base`
        # cannot be confined even before resolving symlinks.
        if logical != base and base not in logical.parents:
            continue
        try:
            resolved = _resolve_existing_confined_path(
                logical, allowed_roots=allowed_roots, raw_path=raw_path
            )
        except ValueError as exc:
            # A symlink whose real target escapes is a hard deny; a not-found
            # here just means the path is absent in this root, so try the next.
            if "path escape denied" in str(exc):
                raise
            last_error = exc
            continue
        display = str(logical.relative_to(base)) if logical != base else "."
        return resolved, name, display

    raise (
        last_error
        if last_error is not None
        else ValueError(f"Path escape denied: {raw_path}")
    )


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _parse_selector(selector: str) -> "list[str]":
    text = selector.strip()
    if not text:
        return []
    if text.startswith("/"):
        return [part for part in text.split("/") if part]
    return [part for part in text.split(".") if part]


def _json_select(payload: Any, selector: str) -> "tuple[bool, Any]":
    current = payload
    for part in _parse_selector(selector):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def _json_assign(payload: Any, selector: str, value: Any) -> Any:
    parts = _parse_selector(selector)
    if not parts:
        return value
    root = payload if isinstance(payload, (dict, list)) else {}
    if not isinstance(root, dict):
        raise ValueError(
            "root json value must be an object when applying nested updates"
        )
    current: Any = root
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        if not isinstance(current, dict):
            raise ValueError(
                f"selector requires object container at: {'.'.join(parts[:index])}"
            )
        if is_last:
            current[part] = value
            return root
        next_part = parts[index + 1]
        child = current.get(part)
        if not isinstance(child, dict):
            child = {} if not next_part.isdigit() else []
            current[part] = child
        current = child
        if isinstance(current, list):
            try:
                next_index = int(next_part)
            except ValueError as exc:
                raise ValueError(
                    f"selector requires list index at: {'.'.join(parts[: index + 2])}"
                ) from exc
            while len(current) <= next_index:
                current.append({})
            current = current[next_index]
    return root
