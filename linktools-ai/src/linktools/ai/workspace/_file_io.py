#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded file facts, text ranges, and path-anchored glob matching."""

import base64
import codecs
import fnmatch
import hashlib
import hmac
import json
import mimetypes
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO, Protocol

from ..core import JsonValue
from ..errors import AIError, ErrorCode

_BLOCK_BYTES = 64 * 1024
_MAX_PREVIEW_BYTES = 50_000
_MAX_SCAN_BYTES = 8 * 1024 * 1024
_MAX_METADATA_BYTES = 256 * 1024 * 1024
_MAX_SCAN_SECONDS = 5.0
_MAGIC_BYTES = 8192
_LINE_ENDINGS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
_TEXT_TYPES = frozenset({
    "application/json", "application/xml", "application/javascript",
    "application/x-ndjson", "application/x-empty", "inode/x-empty",
    "application/csv", "application/yaml", "application/x-yaml", "application/toml",
})


class _ContentDetector(Protocol):
    def from_buffer(self, sample: bytes) -> str: ...


class _MagicThreadState(threading.local):
    def __init__(self) -> None:
        self.detector: _ContentDetector | None = None


_magic_available: bool | None = None
_magic_factory: Callable[..., _ContentDetector] | None = None
_magic_threads = _MagicThreadState()
_magic_lock = threading.Lock()


def _content_detector() -> _ContentDetector | None:
    global _magic_available, _magic_factory
    if _magic_available is False:
        return None
    if _magic_threads.detector is not None:
        return _magic_threads.detector
    with _magic_lock:
        if _magic_available is False:
            return None
        try:
            if _magic_factory is None:
                from magic import Magic
                _magic_factory = Magic
            detector = _magic_factory(mime=True)
        except Exception:
            # A missing binding, library, or database disables detection until restart.
            _magic_available = False
            _magic_factory = None
            return None
        _magic_threads.detector = detector
        _magic_available = True
        return detector


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _state(path: str, info: os.stat_result) -> str:
    value = json.dumps([path, *_identity(info)], separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_state(path: Path, logical: str, stream: BinaryIO, state: str) -> None:
    if _state(logical, os.fstat(stream.fileno())) != state or _state(logical, path.stat()) != state:
        raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "file_changed"})


def _is_text_type(value: str | None) -> bool:
    return bool(value and (value.startswith("text/") or value.endswith(("+json", "+xml")) or value in _TEXT_TYPES))


def _facts(sample: bytes, name: str, size: int) -> dict[str, JsonValue]:
    extension = mimetypes.guess_type(name)[0]
    detected: str | None = None
    reason: str | None = None
    detector = _content_detector()
    if detector is not None:
        try:
            detected = detector.from_buffer(sample[:_MAGIC_BYTES])
            if not isinstance(detected, str) or "/" not in detected:
                detected = None
                reason = "type_detection_failed"
        except Exception:
            # Per-buffer failures do not mean that the installed detector is absent.
            reason = "type_detection_failed"
    if detected is not None:
        detected = detected.split(";", 1)[0].strip().lower()
    text_extension = _is_text_type(extension)
    text_content = _is_text_type(detected)
    sample_text = not sample or not any(byte < 9 or 13 < byte < 32 for byte in sample)
    # Logs may contain invalid UTF-8 or a few NULs; known binary/container facts win.
    if not sample_text and (text_extension or Path(name).suffix.lower() in {".log", ".txt", ".csv"}):
        sample_text = sum(byte == 0 or byte < 9 or 13 < byte < 32 for byte in sample) <= max(1, len(sample) // 100)
    opaque = detected in {None, "application/octet-stream", "application/x-empty", "inode/x-empty"}
    is_text = size == 0 or text_content or (opaque and sample_text and (extension is None or text_extension or Path(name).suffix.lower() == ".log"))
    media_type = detected or extension or ("text/plain" if is_text else "application/octet-stream")
    source = "content" if detected else ("extension" if extension else "unknown")
    # Generic text recognition does not identify a structured text subtype.
    # Preserve the extension hint without claiming content validation of it.
    generic_text_hint = detected == "text/plain" and text_extension and extension != detected
    if generic_text_hint:
        media_type, source = extension, "extension"
    if detected and extension and detected != extension and not generic_text_hint:
        source = "conflict"
    if size == 0 and is_text:
        media_type = extension if text_extension else "text/plain"
    result: dict[str, JsonValue] = {
        "media_type": media_type, "type_source": source, "size": size,
        "is_text": is_text, "bytes_read": len(sample),
    }
    if extension is not None:
        result["extension_media_type"] = extension
    if reason is not None:
        result["detection_reason"] = reason
    return result


def _inspect_file(path: Path, logical: str, max_bytes: int | None = None) -> dict[str, JsonValue]:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if not stat.S_ISREG(path.stat().st_mode):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        state = _state(logical, info)
        sample = stream.read(min(max_bytes or _MAGIC_BYTES, _MAGIC_BYTES))
        result = _facts(sample, logical, info.st_size)
        _check_state(path, logical, stream, state)
    result.update(path=logical, state=state)
    return result


def _encode_cursor(value: Mapping[str, JsonValue]) -> str:
    payload = json.dumps({"version": 1, **value}, separators=(",", ":")).encode("utf-8")
    signature = hashlib.sha256(payload).digest()
    return base64.urlsafe_b64encode(signature + payload).decode("ascii")


def _decode_cursor(value: str) -> dict[str, JsonValue]:
    try:
        if not isinstance(value, str) or len(value) > 16 * 1024:
            raise ValueError
        raw = base64.b64decode(value, altchars=b"-_", validate=True)
        signature, payload = raw[:32], raw[32:]
        if not hmac.compare_digest(signature, hashlib.sha256(payload).digest()):
            raise ValueError
        result = json.loads(payload)
        if not isinstance(result, dict) or set(result) != {"version", "path", "state", "byte", "line", "column"}:
            raise ValueError
        if type(result["version"]) is not int or result["version"] != 1:
            raise ValueError
        if not isinstance(result["path"], str) or not isinstance(result["state"], str):
            raise ValueError
        if any(type(result[field]) is not int or result[field] < 0 for field in ("byte", "line", "column")):
            raise ValueError
        return result
    except (ValueError, TypeError, UnicodeError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_cursor"}) from error


class _ReadBudget:
    def __init__(self, stream: BinaryIO, maximum: int) -> None:
        self.stream = stream
        self.maximum = maximum
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        remaining = self.maximum - self.bytes_read
        if remaining <= 0:
            if self.stream.tell() == os.fstat(self.stream.fileno()).st_size:
                return b""
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_byte_limit"})
        value = self.stream.read(min(size, remaining))
        self.bytes_read += len(value)
        return value

    def seek(self, offset: int) -> int:
        return self.stream.seek(offset)


def _validate_cursor_position(reader: _ReadBudget, position: int, size: int) -> None:
    """The checksum detects corruption; authorization remains a per-call check."""
    if position > size:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_cursor_position"})
    start = max(0, position - 3)
    end = min(size, position + 3)
    reader.seek(start)
    sample = reader.read(end - start) if end > start else b""
    if len(sample) != end - start:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_byte_limit"})
    boundary = position - start
    consumed = 0
    boundaries = {0}
    for char in sample.decode("utf-8", "surrogateescape"):
        consumed += 1 if 0xDC80 <= ord(char) <= 0xDCFF else len(char.encode("utf-8"))
        boundaries.add(consumed)
    if boundary not in boundaries or (0 < boundary < len(sample) and sample[boundary - 1:boundary + 1] == b"\r\n"):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_cursor_boundary"})
    reader.seek(position)


def _characters(stream: _ReadBudget, max_scan: int, deadline: float) -> Iterator[tuple[str, int, bool]]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")
    consumed = 0
    pending_cr = False
    while True:
        if time.monotonic() > deadline:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_time_limit"})
        remaining = max_scan - consumed
        if remaining <= 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_byte_limit"})
        chunk = stream.read(min(_BLOCK_BYTES, remaining))
        consumed += len(chunk)
        value = decoder.decode(chunk, final=not chunk)
        for char in value:
            if pending_cr:
                pending_cr = False
                if char == "\n":
                    yield "\r\n", 2, False
                    continue
                yield "\r", 1, False
            if char == "\r":
                pending_cr = True
                continue
            invalid = 0xDC80 <= ord(char) <= 0xDCFF
            size = 1 if invalid else len(char.encode("utf-8"))
            lossy = invalid or char == "\x00"
            yield "\ufffd" if lossy else char, size, lossy
        if not chunk:
            if pending_cr:
                yield "\r", 1, False
            return


def _metadata(stream: _ReadBudget, size: int, deadline: float) -> tuple[str, int]:
    if size > _MAX_METADATA_BYTES:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "metadata_byte_limit"})
    stream.seek(0)
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    lines = 0
    pending = False
    previous_cr = False
    scanned = 0
    while True:
        if time.monotonic() > deadline:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_time_limit"})
        chunk = stream.read(_BLOCK_BYTES)
        scanned += len(chunk)
        if scanned > _MAX_METADATA_BYTES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "metadata_byte_limit"})
        digest.update(chunk)
        text = decoder.decode(chunk, final=not chunk)
        if text:
            skip_lf = previous_cr and text.startswith("\n")
            previous_cr = text.endswith("\r")
            if skip_lf:
                text = text[1:]
        if text:
            endings = list(re.finditer(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]", text))
            lines += len(endings)
            pending = not endings or endings[-1].end() != len(text)
            previous_cr = text.endswith("\r")
        if not chunk:
            return digest.hexdigest(), lines + int(pending)


def _read_text_range(
    path: Path, logical: str, *, offset: int | None,
    limit: int | None, cursor: str | None, full_metadata: bool,
    max_bytes: int | None, max_read_bytes: int | None,
) -> dict[str, JsonValue]:
    if offset is not None and (type(offset) is not int or offset < 0):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if type(full_metadata) is not bool or (cursor is not None and offset is not None):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 4):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if max_read_bytes is not None and (type(max_read_bytes) is not int or max_read_bytes < 4):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    maximum = min(max_bytes or _MAX_PREVIEW_BYTES, _MAX_PREVIEW_BYTES)
    requested_limit = limit or 200
    selected_limit = min(requested_limit, 2000)
    selected_offset = offset or 0
    position = line = column = 0
    deadline = time.monotonic() + _MAX_SCAN_SECONDS
    if not stat.S_ISREG(path.stat().st_mode):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        state = _state(logical, info)
        scan_ceiling = _MAX_SCAN_BYTES + (_MAX_METADATA_BYTES if full_metadata else 0)
        reader = _ReadBudget(stream, min(max_read_bytes or scan_ceiling, scan_ceiling))
        if cursor is not None:
            saved = _decode_cursor(cursor)
            if saved["path"] != logical:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "cursor_path_mismatch"})
            if saved["state"] != state:
                raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "stale_cursor"})
            position, line, column = saved["byte"], saved["line"], saved["column"]
            if line + column > position:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_cursor_position"})
            selected_offset = line
            _validate_cursor_position(reader, position, info.st_size)
        start_position, start_column = position, column
        pieces: list[str] = []
        output_bytes = 0
        lossy = truncated = False
        reached_range = line >= selected_offset
        try:
            for char, byte_count, damaged in _characters(reader, _MAX_SCAN_BYTES, deadline):
                if line >= selected_offset + selected_limit:
                    break
                if line >= selected_offset:
                    if not reached_range:
                        reached_range = True
                        start_position, start_column = position, column
                    rendered_bytes = len(char.encode("utf-8"))
                    if output_bytes + rendered_bytes > maximum:
                        truncated = True
                        break
                    pieces.append(char)
                    output_bytes += rendered_bytes
                    lossy |= damaged
                position += byte_count
                if char == "\r\n" or char in _LINE_ENDINGS:
                    line += 1
                    column = 0
                else:
                    column += 1
                if position == info.st_size or line >= selected_offset + selected_limit:
                    break
                if output_bytes == maximum:
                    truncated = True
                    break
        except AIError as error:
            if error.safe_details.get("reason") not in {"scan_byte_limit", "scan_time_limit"}:
                raise
            if not reached_range:
                raise
            truncated = True
        if not reached_range and info.st_size:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "offset_out_of_range"})
        eof = position >= info.st_size
        if eof:
            truncated = False
        elif requested_limit > selected_limit and line >= selected_offset + selected_limit:
            truncated = True
        if truncated and position == start_position:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "scan_byte_limit"})
        result: dict[str, JsonValue] = {
            "path": logical, "content": "".join(pieces), "size": info.st_size,
            "state": state, "offset": selected_offset, "end_offset": line,
            "byte_offset": start_position, "end_byte_offset": position,
            "line_offset": start_column, "end_line_offset": column,
            "lossy": lossy, "truncated": truncated, "eof": eof,
            "next_cursor": None if eof else _encode_cursor({
                "path": logical, "state": state, "byte": position,
                "line": line, "column": column,
            }),
        }
        if full_metadata:
            digest, lines = _metadata(reader, info.st_size, deadline)
            result.update(hash=digest, total_lines=lines)
        result["bytes_read"] = reader.bytes_read
        _check_state(path, logical, stream, state)
        return result


def _glob_matches(path: str, pattern: str) -> bool:
    patterns = pattern.split("/")

    def closure(states: set[int]) -> set[int]:
        result = set(states)
        for state in states:
            while state < len(patterns) and patterns[state] == "**":
                state += 1
                result.add(state)
        return result

    states = closure({0})
    for part in path.split("/"):
        following: set[int] = set()
        for state in states:
            if state == len(patterns):
                continue
            if patterns[state] == "**":
                following.add(state)
            elif fnmatch.fnmatchcase(part, patterns[state]):
                following.add(state + 1)
        states = closure(following)
        if not states:
            return False
    return len(patterns) in states
