#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-backed repository instruction discovery."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

from ..errors import AIError, ErrorCode
from ..spec import RepositoryInstructionDocument, RepositoryInstructions
from ._root import WorkspacePolicy


class LocalRepositoryInstructionResolver:
    def __init__(
        self,
        root: Path,
        policy: WorkspacePolicy,
        rules: RepositoryInstructions,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be Path")
        if not isinstance(policy, WorkspacePolicy):
            raise TypeError("policy must be WorkspacePolicy")
        if not isinstance(rules, RepositoryInstructions) or any(
            not document.source.startswith("rule:")
            for document in rules.documents
        ):
            raise TypeError("rules must contain only Rule instruction documents")
        self._root = root
        self._policy = policy
        self._rules = rules

    def _resolve_agents_blocking(
        self,
        relative_target: Path,
        exclude_sources: frozenset[str],
    ) -> tuple[RepositoryInstructionDocument, ...]:
        resolved_workspace_root = _resolve_existing_path(self._root)
        parts = relative_target.parts
        documents: list[RepositoryInstructionDocument] = []
        for depth in range(len(parts) + 1):
            prefix_parts = parts[:depth]
            lexical_path = Path(*prefix_parts, "AGENTS.md")
            logical_path = lexical_path.as_posix()
            source = f"agents:{logical_path}"
            if source in exclude_sources:
                continue
            scope = "." if not prefix_parts else Path(*prefix_parts).as_posix()
            content = _read_verified_instruction_file(
                self._root / lexical_path,
                containment_roots=(resolved_workspace_root,),
                max_bytes=self._policy.max_repository_instruction_bytes,
                missing_ok=True,
                allow_lexical_symlink=True,
            )
            if content is not None:
                documents.append(
                    RepositoryInstructionDocument(source, scope, content)
                )
        return tuple(documents)

    async def resolve(
        self,
        path: str | Path = ".",
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        relative_target = _normalize_target_path(self._root, path)
        target_scope = "." if not relative_target.parts else relative_target.as_posix()
        rules = self._rules.for_path(
            target_scope,
            exclude_sources=exclude_sources,
        )
        agents = await asyncio.to_thread(
            self._resolve_agents_blocking,
            relative_target,
            exclude_sources,
        )
        bundle = RepositoryInstructions((*rules.documents, *agents))
        bundle.validate_limits(
            max_documents=self._policy.max_repository_instruction_documents,
            max_bytes=self._policy.max_repository_instruction_bytes,
        )
        return bundle


def _normalize_target_path(root: Path, value: str | Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    root_value = os.fspath(root)
    normalized = Path(
        os.path.abspath(
            os.path.normpath(
                raw if os.path.isabs(raw) else os.path.join(root_value, raw)
            )
        )
    )
    try:
        return normalized.relative_to(root)
    except (ValueError, OSError) as error:
        raise AIError(ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT) from error


def _read_verified_instruction_file(
    candidate: Path,
    *,
    containment_roots: tuple[Path, ...],
    max_bytes: int,
    missing_ok: bool,
    allow_lexical_symlink: bool,
) -> str | None:
    if (
        not isinstance(containment_roots, tuple)
        or not containment_roots
        or any(
            not isinstance(root, Path) or not root.is_absolute()
            for root in containment_roots
        )
        or not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
        or not isinstance(missing_ok, bool)
        or not isinstance(allow_lexical_symlink, bool)
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        lexical_stat = candidate.lstat()
    except (FileNotFoundError, NotADirectoryError) as error:
        if missing_ok:
            return None
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    lexical_is_symlink = stat.S_ISLNK(lexical_stat.st_mode)
    if lexical_is_symlink and not allow_lexical_symlink:
        raise AIError(ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT)

    resolved_source = _resolve_existing_path(candidate)
    _require_contained(resolved_source, containment_roots)
    try:
        resolved_stat = resolved_source.stat()
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if not lexical_is_symlink and not os.path.samestat(lexical_stat, resolved_stat):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    flags = os.O_RDONLY
    if os.name == "nt":
        flags |= os.O_BINARY
    else:
        flags |= os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        fd = os.open(resolved_source, flags)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    with os.fdopen(fd, "rb", closefd=True) as stream:
        opened_before = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_before.st_mode):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)

        revalidated_source = _resolve_existing_path(candidate)
        _require_contained(revalidated_source, containment_roots)
        if revalidated_source != resolved_source:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        try:
            revalidated_stat = revalidated_source.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if (
            not stat.S_ISREG(revalidated_stat.st_mode)
            or not os.path.samestat(opened_before, revalidated_stat)
        ):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        data = stream.read(max_bytes + 1)
        opened_after = os.fstat(stream.fileno())
        if (
            not os.path.samestat(opened_before, opened_after)
            or opened_before.st_size != opened_after.st_size
            or opened_before.st_mtime_ns != opened_after.st_mtime_ns
            or opened_before.st_ctime_ns != opened_after.st_ctime_ns
        ):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        final_source = _resolve_existing_path(candidate)
        _require_contained(final_source, containment_roots)
        if final_source != resolved_source:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        try:
            final_stat = final_source.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or not os.path.samestat(opened_after, final_stat)
        ):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    if len(data) > max_bytes:
        raise AIError(ErrorCode.PROMPT_TOO_LARGE)
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _resolve_existing_path(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _require_contained(path: Path, roots: tuple[Path, ...]) -> None:
    for root in roots:
        try:
            path.relative_to(root)
            return
        except (ValueError, OSError):
            continue
    raise AIError(ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT)


__all__ = ["LocalRepositoryInstructionResolver"]
