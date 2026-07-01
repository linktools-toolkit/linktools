#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manage workspace slots, trace files, findings, and report storage."""

import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class _EventLike(Protocol):
    """Structural shape `init_trace` needs — avoids an infra→secops import."""

    def to_dict(self) -> dict[str, Any]: ...


class _FindingLike(Protocol):
    """Structural shape `write_finding` needs — avoids an infra→secops import."""

    trace_id: str
    agent_id: str

    def to_dict(self) -> dict[str, Any]: ...


class SlotMode(str, Enum):
    WARM = "warm"                      # warm-load only, no write-back (operator-maintained content)
    WARM_WRITEBACK = "warm_writeback"  # warm-load + write-back (long-term memory)
    WRITEBACK = "writeback"            # no warm-load, write-back only (runtime output)


_DEFAULT_SLOTS: dict[str, SlotMode] = {
    "agent":      SlotMode.WARM,
    "runtime":    SlotMode.WRITEBACK,
    "findings":   SlotMode.WRITEBACK,
    "entities":   SlotMode.WARM_WRITEBACK,
    "feedback":   SlotMode.WARM_WRITEBACK,
    "candidates": SlotMode.WRITEBACK,
    "trace":      SlotMode.WRITEBACK,
    "session":    SlotMode.WRITEBACK,
    "cache":      SlotMode.WRITEBACK,
    "state":      SlotMode.WRITEBACK,
    "governance": SlotMode.WRITEBACK,
}

_SLOT_PATHS: dict[str, Path] = {
    "agent":      Path("agent"),
    "runtime":    Path("runtime"),
    "findings":   Path("runtime/findings"),
    "entities":   Path("runtime/entities"),
    "feedback":   Path("feedback"),
    "candidates": Path("candidates"),
    "trace":      Path("trace"),
    "session":    Path("session"),
    "cache":      Path("cache"),
    "state":      Path("state"),
    "governance": Path("state/governance"),
}


class WorkspaceStore(ABC):
    @abstractmethod
    def write_text(self, path: Path, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_text(self, path: Path, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, path: Path) -> str:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: Path) -> bool:
        raise NotImplementedError


class LocalWorkspaceStore(WorkspaceStore):
    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def exists(self, path: Path) -> bool:
        return path.exists()


class TraceStore:
    """Unified workspace storage with per-slot warm-load and write-back modes.

    Slot modes:
      WARM           → warm-load only, no write-back (e.g. agent overrides, operator-maintained)
      WARM_WRITEBACK → warm-load + write-back (e.g. long-term memory under memory/)
      WRITEBACK      → no warm-load, write-back only (e.g. trace, cache, state)

    Plugin interface:
      warm_load(slot, loader)        → read data from slot directory and cache it
      write_back(slot, rel, data)    → write a file to the slot directory
      register_slot(name, mode)      → register a custom slot
    """

    def __init__(
        self,
        root: Path,
        trace_id: str | None = None,
        trace_scoped: bool = False,
        extra_slots: dict[str, SlotMode] | None = None,
        store: WorkspaceStore | None = None,
    ):
        self.root = root
        self.trace_id = trace_id
        self.trace_scoped = trace_scoped
        self._slots: dict[str, SlotMode] = {**_DEFAULT_SLOTS, **(extra_slots or {})}
        self._warm_cache: dict[str, Any] = {}
        self.store = store or LocalWorkspaceStore()

    # ── Slot Management ──────────────────────────────────────────────────

    def register_slot(self, name: str, mode: SlotMode) -> None:
        """Register a custom slot from a plugin."""
        self._slots[name] = mode

    def slot_path(self, slot: str) -> Path:
        if slot == "trace":
            return self.root
        return self.root / _SLOT_PATHS.get(slot, Path(slot))

    def for_trace(self, trace_id: str) -> "TraceStore":
        """Return a workspace view bound to the specified trace."""
        if self.trace_scoped and self.trace_id == trace_id:
            scoped = TraceStore(self.root, trace_id=trace_id, trace_scoped=True, extra_slots=self._slots, store=self.store)
        else:
            scoped = TraceStore(self.root / trace_id, trace_id=trace_id, trace_scoped=True, extra_slots=self._slots, store=self.store)
        scoped._warm_cache = self._warm_cache
        return scoped

    def slot_mode(self, slot: str) -> SlotMode:
        return self._slots.get(slot, SlotMode.WRITEBACK)

    def warm_slots(self) -> list[str]:
        return [k for k, v in self._slots.items() if v in (SlotMode.WARM, SlotMode.WARM_WRITEBACK)]

    def writeback_slots(self) -> list[str]:
        return [k for k, v in self._slots.items() if v in (SlotMode.WARM_WRITEBACK, SlotMode.WRITEBACK)]

    # ── Warm-Load Interface (for plugins) ────────────────────────────────

    def warm_load(self, slot: str, loader: Callable[[Path], Any]) -> Any:
        """Read and cache warm-load data from the specified slot. loader receives the slot directory Path."""
        mode = self.slot_mode(slot)
        if mode not in (SlotMode.WARM, SlotMode.WARM_WRITEBACK):
            raise ValueError(f"Slot '{slot}' (mode={mode.value}) does not support warm-load")
        path = self.slot_path(slot)
        data = loader(path) if path.exists() else {}
        self._warm_cache[slot] = data
        return data

    def get_warm(self, slot: str) -> Any | None:
        """Return cached warm-load data for the slot."""
        return self._warm_cache.get(slot)

    # ── Write-Back Interface (for plugins) ───────────────────────────────

    def write_back(self, slot: str, rel_path: str | Path, data: Any) -> Path:
        """Write a JSON file to the specified slot. rel_path is relative to the slot directory."""
        mode = self.slot_mode(slot)
        if mode not in (SlotMode.WARM_WRITEBACK, SlotMode.WRITEBACK):
            raise ValueError(f"Slot '{slot}' (mode={mode.value}) does not support write-back")
        path = self.slot_path(slot) / rel_path
        if isinstance(data, (dict, list)):
            self.store.write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
        else:
            self.store.write_text(path, str(data))
        return path

    # ── Engine Internal Interface ────────────────────────────────────────

    def init_trace(self, trace_id: str, event: _EventLike) -> Path:
        scoped = self.for_trace(trace_id)
        trace_dir = scoped.trace_dir()
        scoped.slot_path("runtime").mkdir(parents=True, exist_ok=True)
        scoped.slot_path("findings").mkdir(parents=True, exist_ok=True)
        scoped.slot_path("session").mkdir(parents=True, exist_ok=True)
        self._write_json(trace_dir / "event.json", event.to_dict())
        return trace_dir

    def trace_dir(self, trace_id: str | None = None) -> Path:
        if self.trace_scoped and (trace_id is None or trace_id == self.trace_id):
            return self.slot_path("trace")
        if trace_id:
            return self.for_trace(trace_id).trace_dir()
        raise ValueError("trace_id is required for an unscoped TraceStore")

    def write_finding(self, finding: _FindingLike) -> None:
        self._write_json(
            self.for_trace(finding.trace_id).slot_path("findings") / f"{finding.agent_id}.json",
            finding.to_dict(),
        )

    def read_finding(self, trace_id: str, agent_id: str) -> dict[str, Any] | None:
        path = self.for_trace(trace_id).slot_path("findings") / f"{agent_id}.json"
        return json.loads(self.store.read_text(path)) if self.store.exists(path) else None

    def write_report(self, trace_id: str, report: dict[str, Any]) -> Path:
        path = self.trace_dir(trace_id) / "report.json"
        self._write_json(path, report)
        return path

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write a JSON object to an absolute path under the workspace."""
        self._ensure_store_path(path)
        self._write_json(path, payload)

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        """Append one JSON object to an absolute JSONL path under the workspace."""
        self._ensure_store_path(path)
        self.store.append_text(path, json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_store_path(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"workspace write path must be under workspace root: {path}") from exc

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self.store.write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
