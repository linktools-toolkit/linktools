from __future__ import annotations

from pathlib import Path


def load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def save(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = load(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:80]!r}")
    save(path, text.replace(old, new, 1))


def insert_before_once(path: str, marker: str, content: str) -> None:
    replace_once(path, marker, content + marker)


def insert_after_once(path: str, marker: str, content: str) -> None:
    replace_once(path, marker, marker + content)


path = "linktools-ai/src/linktools/ai/runtime/service_api.py"
replace_once(path, "from dataclasses import dataclass, field\n", "from dataclasses import dataclass, field\nfrom datetime import datetime\n")
insert_before_once(
    path,
    "\n\nclass ExecutionHistoryReader(Protocol):\n",
    '''\n\n@dataclass(frozen=True, slots=True)\nclass SessionTurnItem:\n    ordinal: int\n    item_kind: str\n    content: JsonValue\n    tool_name: "str | None" = None\n    tool_call_id: "str | None" = None\n\n    def __post_init__(self) -> None:\n        if self.ordinal < 1 or not self.item_kind:\n            raise ValueError("session timeline item is invalid")\n\n\n@dataclass(frozen=True, slots=True)\nclass SessionTurn:\n    execution_id: str\n    status: ExecutionStatus\n    created_at: datetime\n    updated_at: datetime\n    user_input: JsonValue\n    conversation_committed: bool\n    items: tuple[SessionTurnItem, ...]\n    error_code: "str | None" = None\n    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))\n''',
)
insert_after_once(
    path,
    '''    async def history(\n        self,\n        session_id: str,\n        *,\n        principal: Principal,\n        cursor: "str | None" = None,\n        limit: int = 100,\n    ) -> "Page[SessionHistoryItem]": ...\n''',
    '''    async def timeline(\n        self,\n        session_id: str,\n        *,\n        principal: Principal,\n        cursor: "str | None" = None,\n        limit: int = 100,\n    ) -> "Page[SessionTurn]": ...\n''',
)
replace_once(
    path,
    '''    "SessionHistoryReader",\n    "SessionService",\n    "SessionView",\n''',
    '''    "SessionHistoryReader",\n    "SessionService",\n    "SessionTurn",\n    "SessionTurnItem",\n    "SessionView",\n''',
)

path = "linktools-ai/src/linktools/ai/runtime/_session.py"
replace_once(
    path,
    "from typing import Protocol\n\nfrom linktools.core import environ\n",
    "from typing import Protocol\n\nfrom linktools.core import environ\nfrom pydantic_ai.messages import ModelRequest, ModelResponse\n",
)
replace_once(
    path,
    '''    SessionHistoryReader,\n    SessionView,\n''',
    '''    SessionHistoryReader,\n    SessionTurn,\n    SessionTurnItem,\n    SessionView,\n''',
)
replace_once(
    path,
    '''    SessionRecord,\n)\n''',
    '''    SessionRecord,\n    SessionTurnRef,\n)\n''',
)
insert_after_once(
    path,
    '''from .state._contracts import (\n    ConversationCursor,\n)\n''',
    '''from ._input import stored_user_input_view\nfrom .state._views import project_session_history_message\n''',
)
marker = '''    async def list(self, request: ListSessionRequest) -> Page[SessionView]:\n'''
timeline_code = '''    async def timeline(\n        self,\n        session_id: str,\n        *,\n        principal: Principal,\n        cursor: "str | None" = None,\n        limit: int = 100,\n    ) -> Page[SessionTurn]:\n        if not 1 <= limit <= 200:\n            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)\n        async with self._session_consumer(session_id, principal.tenant_id):\n            root = await self._authorized(\n                session_id,\n                principal,\n                AuthorizationAction.SESSION_READ,\n            )\n            if self._transcript_store is None:\n                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n            coordinate = _decode_timeline_cursor(\n                cursor,\n                tenant_id=principal.tenant_id,\n                session_id=session_id,\n                signer=self._cursor_signer,\n            )\n            blocks, next_coordinate = await self._timeline_blocks(\n                root,\n                coordinate=coordinate,\n                limit=limit,\n            )\n            refs = tuple(ref for _record, values in blocks for ref in values)\n            if not refs:\n                return Page((), None)\n            execution_ids = tuple(dict.fromkeys(ref.execution_id for ref in refs))\n            executions = await self._executions.get_many(\n                execution_ids,\n                tenant_id=principal.tenant_id,\n            )\n            if len(executions) != len(execution_ids):\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n\n            commits: dict[tuple[str, int], object] = {}\n            messages: dict[str, tuple[object, ...]] = {}\n            message_bases: dict[str, int] = {}\n            for record, values in blocks:\n                if not values:\n                    continue\n                start = values[0].sequence\n                end = values[-1].sequence + 1\n                committed = await self._conversation.sessions.list_timeline_commits(\n                    record.session_id,\n                    tenant_id=record.tenant_id,\n                    start_sequence=start,\n                    end_sequence=end,\n                )\n                for item in committed:\n                    commits[(record.session_id, item.sequence)] = item\n                if not committed:\n                    continue\n                if record.history_id is None:\n                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n                range_start = min(item.start_message_index for item in committed)\n                range_end = max(item.end_message_index for item in committed)\n                loaded = tuple(\n                    [\n                        item\n                        async for item in self._transcript_store.iter_session_message_range(\n                            record.history_id,\n                            tenant_id=record.tenant_id,\n                            start=range_start,\n                            end=range_end,\n                        )\n                    ]\n                )\n                if len(loaded) != range_end - range_start:\n                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n                messages[record.session_id] = loaded\n                message_bases[record.session_id] = range_start\n\n            turns: list[SessionTurn] = []\n            for ref in refs:\n                execution = executions[ref.execution_id]\n                if (\n                    execution.session_id != ref.session_id\n                    or execution.parent_execution_id is not None\n                ):\n                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n                commit = commits.get((ref.session_id, ref.sequence))\n                items: tuple[SessionTurnItem, ...] = ()\n                if commit is not None:\n                    base = message_bases[ref.session_id]\n                    source = messages[ref.session_id]\n                    start = commit.start_message_index - base\n                    end = commit.end_message_index - base\n                    if start < 0 or end > len(source) or end <= start:\n                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n                    items = _timeline_items(source[start:end])\n                elif execution.status is ExecutionStatus.SUCCEEDED:\n                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n                turns.append(\n                    SessionTurn(\n                        execution.execution_id,\n                        execution.status,\n                        execution.created_at,\n                        execution.updated_at,\n                        stored_user_input_view(execution.stored_user_input),\n                        commit is not None,\n                        items,\n                        execution.error_code,\n                        execution.safe_error_details,\n                    )\n                )\n            next_cursor = (\n                None\n                if next_coordinate is None\n                else _timeline_cursor(\n                    tenant_id=principal.tenant_id,\n                    session_id=session_id,\n                    source_session_id=next_coordinate[0],\n                    before_sequence=next_coordinate[1],\n                    signer=self._cursor_signer,\n                )\n            )\n            return Page(tuple(turns), next_cursor)\n\n    async def _timeline_blocks(\n        self,\n        root: SessionRecord,\n        *,\n        coordinate: "tuple[str, int] | None",\n        limit: int,\n    ) -> tuple[\n        tuple[tuple[SessionRecord, tuple[SessionTurnRef, ...]], ...],\n        "tuple[str, int] | None",\n    ]:\n        tenant_id = root.tenant_id\n        source_id = root.session_id if coordinate is None else coordinate[0]\n        source_before = None if coordinate is None else coordinate[1]\n        remaining = limit\n        newest_first: list[tuple[SessionRecord, tuple[SessionTurnRef, ...]]] = []\n        visited: set[str] = set()\n        current: SessionRecord | None = root if source_id == root.session_id else None\n        while remaining > 0:\n            if source_id in visited:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            visited.add(source_id)\n            if current is None or current.session_id != source_id:\n                current = await self._conversation.sessions.get(\n                    source_id, tenant_id=tenant_id\n                )\n                if current is None:\n                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n                if (\n                    current.tenant_id != root.tenant_id\n                    or current.owner_principal_id != root.owner_principal_id\n                ):\n                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            head = await self._conversation.sessions.timeline_head(\n                source_id, tenant_id=tenant_id\n            )\n            before = head + 1 if source_before is None else source_before\n            if before < 1 or before > head + 1:\n                raise AIError(ErrorCode.CURSOR_INVALID)\n            available = before - 1\n            if available:\n                count = min(remaining, available)\n                start = before - count\n                values = await self._conversation.sessions.list_timeline_turns(\n                    source_id,\n                    tenant_id=tenant_id,\n                    start_sequence=start,\n                    end_sequence=before,\n                )\n                newest_first.append((current, values))\n                remaining -= len(values)\n                before = start\n            source_before = before\n            if remaining == 0:\n                break\n            if before > 1:\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            parent = current.timeline_parent_session_id\n            if parent is None:\n                source_id = ""\n                break\n            source_id = parent\n            source_before = current.timeline_parent_turn_sequence + 1\n            current = None\n\n        next_coordinate: tuple[str, int] | None = None\n        if source_id:\n            if current is None or current.session_id != source_id:\n                current = await self._conversation.sessions.get(\n                    source_id, tenant_id=tenant_id\n                )\n                if current is None:\n                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n            before = 1 if source_before is None else source_before\n            if before > 1 or current.timeline_parent_session_id is not None:\n                next_coordinate = (source_id, before)\n        return tuple(reversed(newest_first)), next_coordinate\n\n'''
insert_before_once(path, marker, timeline_code)
replace_once(
    path,
    '''            source = await self._authorized(\n                session_id, request.principal, AuthorizationAction.SESSION_READ\n            )\n            await self._authorization.authorize(\n''',
    '''            source = await self._authorized(\n                session_id, request.principal, AuthorizationAction.SESSION_READ\n            )\n            source = await self._reconcile_terminal_admission(source)\n            await self._authorization.authorize(\n''',
)
insert_before_once(
    path,
    '''_logger = environ.get_logger("ai.runtime.session")\n''',
    '''_TIMELINE_PROJECTION_VERSION = 1\n\n\ndef _timeline_items(messages: tuple[object, ...]) -> tuple[SessionTurnItem, ...]:\n    values: list[SessionTurnItem] = []\n    for message in messages:\n        projected = project_session_history_message(message)\n        if isinstance(message, ModelRequest):\n            projected = tuple(\n                item for item in projected if item.item_kind == "tool_result"\n            )\n        elif not isinstance(message, ModelResponse):\n            continue\n        for item in projected:\n            values.append(\n                SessionTurnItem(\n                    len(values) + 1,\n                    item.item_kind,\n                    item.content,\n                    item.tool_name,\n                    item.tool_call_id,\n                )\n            )\n    return tuple(values)\n\n\ndef _timeline_filter_digest(session_id: str) -> str:\n    return canonical_sha256(\n        {\n            "session_id": session_id,\n            "projection_version": _TIMELINE_PROJECTION_VERSION,\n        }\n    )\n\n\ndef _timeline_cursor(\n    *,\n    tenant_id: str,\n    session_id: str,\n    source_session_id: str,\n    before_sequence: int,\n    signer: CursorSigner,\n) -> str:\n    return signer.encode(\n        CursorPayload(\n            1,\n            tenant_id,\n            "SESSION_TIMELINE",\n            _timeline_filter_digest(session_id),\n            json.dumps(\n                [source_session_id, before_sequence],\n                separators=(",", ":"),\n            ),\n            0,\n            int(time.time()) + 3600,\n            projection_version=_TIMELINE_PROJECTION_VERSION,\n        )\n    )\n\n\ndef _decode_timeline_cursor(\n    cursor: str | None,\n    *,\n    tenant_id: str,\n    session_id: str,\n    signer: CursorSigner,\n) -> "tuple[str, int] | None":\n    if cursor is None:\n        return None\n    try:\n        payload = signer.decode(cursor)\n        coordinate = json.loads(payload.sort_key)\n    except (AIError, json.JSONDecodeError, TypeError, ValueError) as error:\n        raise AIError(ErrorCode.CURSOR_INVALID) from error\n    if (\n        payload.cursor_version != 1\n        or payload.tenant_id != tenant_id\n        or payload.resource_kind != "SESSION_TIMELINE"\n        or payload.filter_digest != _timeline_filter_digest(session_id)\n        or payload.snapshot_or_store_revision != 0\n        or payload.projection_version != _TIMELINE_PROJECTION_VERSION\n        or not isinstance(coordinate, list)\n        or len(coordinate) != 2\n        or not isinstance(coordinate[0], str)\n        or not coordinate[0]\n        or isinstance(coordinate[1], bool)\n        or not isinstance(coordinate[1], int)\n        or coordinate[1] < 1\n    ):\n        raise AIError(ErrorCode.CURSOR_INVALID)\n    return coordinate[0], coordinate[1]\n\n\n''',
)

path = "linktools-ai/src/linktools/ai/runtime/_agent.py"
replace_once(
    path,
    '''    SessionHistoryItem,\n    SessionView,\n''',
    '''    SessionHistoryItem,\n    SessionTurn,\n    SessionView,\n''',
)
insert_after_once(
    path,
    '''    async def history(\n        self,\n        *,\n        principal: "Principal | None" = None,\n        cursor: "str | None" = None,\n        limit: int = 100,\n    ) -> "Page[SessionHistoryItem]":\n        return await self._runtime.session.history(\n            self.session_id,\n            principal=self._runtime._resolve_principal(principal or self._principal),\n            cursor=cursor,\n            limit=limit,\n        )\n''',
    '''\n    async def timeline(\n        self,\n        *,\n        principal: "Principal | None" = None,\n        cursor: "str | None" = None,\n        limit: int = 100,\n    ) -> "Page[SessionTurn]":\n        return await self._runtime.session.timeline(\n            self.session_id,\n            principal=self._runtime._resolve_principal(principal or self._principal),\n            cursor=cursor,\n            limit=limit,\n        )\n''',
)

path = "linktools-ai/src/linktools/ai/runtime/__init__.py"
replace_once(
    path,
    '''    SessionHistoryReader,\n    SessionService,\n    SessionView,\n''',
    '''    SessionHistoryReader,\n    SessionService,\n    SessionTurn,\n    SessionTurnItem,\n    SessionView,\n''',
)
replace_once(
    path,
    '''    "SessionHistoryReader",\n    "SessionService",\n    "SessionView",\n''',
    '''    "SessionHistoryReader",\n    "SessionService",\n    "SessionTurn",\n    "SessionTurnItem",\n    "SessionView",\n''',
)

print("session timeline patch part 3 applied")
