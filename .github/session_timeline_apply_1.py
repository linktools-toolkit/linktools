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


# ---------------------------------------------------------------------------
# runtime/_input.py: persist a lightweight display view captured before model
# materialization helpers are injected. The durable replay payload remains the
# sole input truth and keeps its existing digest semantics.
# ---------------------------------------------------------------------------
path = "linktools-ai/src/linktools/ai/runtime/_input.py"
insert_before_once(
    path,
    "\n\ndef input_intent(\n",
    '''\n\nclass _MaterializedUserContent(tuple):\n    view: Mapping[str, JsonValue]\n\n    def __new__(\n        cls,\n        items: Sequence[UserContent],\n        view: Mapping[str, JsonValue],\n    ) -> "_MaterializedUserContent":\n        value = super().__new__(cls, items)\n        value.view = dict(view)\n        return value\n''',
)
replace_once(
    path,
    '''        additions: list[UserContent] = []\n        for path in files:\n            media_type = self._media_type(path)\n            remaining = self._policy.max_binary_input_bytes - total_bytes\n            if remaining < 0:\n                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n            try:\n                body = await self._access.read_bytes(path, max_bytes=remaining)\n            except AIError as error:\n                mapped = _file_request_error(error, request_invalid_reason="file_invalid")\n                if mapped is None:\n                    raise\n                raise mapped from error\n            total_bytes += len(body)\n            if total_bytes > self._policy.max_binary_input_bytes:\n                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n            additions.extend(\n                (\n                    f"Workspace file path: {json.dumps(path)}",\n                    BinaryContent(\n                        data=body,\n                        media_type=media_type,\n                    ),\n                )\n            )\n        if isinstance(canonical, str):\n            materialized: CanonicalUserInput = (canonical, *additions)\n        else:\n            materialized = (*canonical, *additions)\n        validate_user_content(materialized)\n        _logger.info(\n            "execution input materialized: files=%s binary_bytes=%s",\n            len(files),\n            total_bytes,\n        )\n        return materialized\n''',
    '''        additions: list[UserContent] = []\n        file_views: list[dict[str, JsonValue]] = []\n        for path in files:\n            media_type = self._media_type(path)\n            remaining = self._policy.max_binary_input_bytes - total_bytes\n            if remaining < 0:\n                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n            try:\n                body = await self._access.read_bytes(path, max_bytes=remaining)\n            except AIError as error:\n                mapped = _file_request_error(error, request_invalid_reason="file_invalid")\n                if mapped is None:\n                    raise\n                raise mapped from error\n            total_bytes += len(body)\n            if total_bytes > self._policy.max_binary_input_bytes:\n                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n            file_views.append(\n                {\n                    "path": path,\n                    "media_type": media_type,\n                    "size": len(body),\n                    "digest": hashlib.sha256(body).hexdigest(),\n                }\n            )\n            additions.extend(\n                (\n                    f"Workspace file path: {json.dumps(path)}",\n                    BinaryContent(\n                        data=body,\n                        media_type=media_type,\n                    ),\n                )\n            )\n        if isinstance(canonical, str):\n            materialized: CanonicalUserInput = (canonical, *additions)\n        else:\n            materialized = (*canonical, *additions)\n        validate_user_content(materialized)\n        _logger.info(\n            "execution input materialized: files=%s binary_bytes=%s",\n            len(files),\n            total_bytes,\n        )\n        return cast(\n            CanonicalUserInput,\n            _MaterializedUserContent(\n                cast(Sequence[UserContent], materialized),\n                _input_view(canonical, file_views),\n            ),\n        )\n''',
)
replace_once(
    path,
    '''        from .state._contracts import StoredUserInput\n\n        canonical = validate_user_input(value)\n        if isinstance(canonical, str):\n            return StoredUserInput(_TEXT_CODEC, StoredPayload.inline_text(canonical))\n        payload = StoredPayload.inline_json(_encode_user_content(canonical))\n''',
    '''        from .state._contracts import StoredUserInput\n\n        view = (\n            dict(value.view)\n            if isinstance(value, _MaterializedUserContent)\n            else _input_view(value)\n        )\n        canonical = validate_user_input(value)\n        if isinstance(canonical, str):\n            return StoredUserInput(\n                _TEXT_CODEC,\n                StoredPayload.inline_text(canonical),\n                view,\n            )\n        payload = StoredPayload.inline_json(_encode_user_content(canonical))\n''',
)
replace_once(
    path,
    '''        return StoredUserInput(_USER_CONTENT_CODEC, payload)\n\n    async def restore''',
    '''        return StoredUserInput(_USER_CONTENT_CODEC, payload, view)\n\n    async def restore''',
)
insert_before_once(
    path,
    "\n\ndef _json_object_or_none(value: object) -> JsonValue:\n",
    '''\n\ndef _input_view(\n    value: _UserPromptInput,\n    files: Sequence[Mapping[str, JsonValue]] = (),\n) -> dict[str, JsonValue]:\n    canonical = validate_user_input(value)\n    prompt = _draft_prompt(canonical)\n    if not isinstance(canonical, str):\n        prompt = {"kind": "items", "items": prompt}\n    try:\n        normalized = normalize_json_value(\n            {\n                "version": 1,\n                "prompt": prompt,\n                "files": [dict(item) for item in files],\n            }\n        )\n    except (TypeError, ValueError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n    if not isinstance(normalized, dict):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    return normalized\n\n\ndef stored_user_input_view(value: "StoredUserInput") -> dict[str, JsonValue]:\n    from .state._contracts import StoredUserInput\n\n    if not isinstance(value, StoredUserInput):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    if value.view is not None:\n        return dict(value.view)\n    if value.payload.kind != "inline":\n        raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)\n    decoded = value.payload.decode()\n    if value.codec == _TEXT_CODEC:\n        if not isinstance(decoded, str):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        return _input_view(decoded)\n    if value.codec != _USER_CONTENT_CODEC or not isinstance(decoded, Mapping):\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    content = _decode_user_content(cast(dict[str, JsonValue], decoded))\n    return _input_view(content)\n''',
)
replace_once(
    path,
    '''    "input_intent",\n    "task_prompt_draft",\n    "validate_user_input",\n]''',
    '''    "input_intent",\n    "stored_user_input_view",\n    "task_prompt_draft",\n    "validate_user_input",\n]''',
)

# ---------------------------------------------------------------------------
# Runtime durable contracts: display metadata stays outside replay digest;
# Session carries only a frozen ancestor/cutoff for forked timeline reads.
# ---------------------------------------------------------------------------
path = "linktools-ai/src/linktools/ai/runtime/state/_contracts.py"
replace_once(
    path,
    '''class StoredUserInput:\n    codec: str\n    payload: StoredPayload\n\n    def __post_init__(self) -> None:\n        if self.codec not in {"text", "user-content-v1"}:\n            raise ValueError("stored user input codec is invalid")\n        if not isinstance(self.payload, StoredPayload):\n            raise TypeError("stored user input payload is invalid")\n''',
    '''class StoredUserInput:\n    codec: str\n    payload: StoredPayload\n    view: Mapping[str, JsonValue] | None = None\n\n    def __post_init__(self) -> None:\n        if self.codec not in {"text", "user-content-v1"}:\n            raise ValueError("stored user input codec is invalid")\n        if not isinstance(self.payload, StoredPayload):\n            raise TypeError("stored user input payload is invalid")\n        if self.view is not None:\n            if not isinstance(self.view, Mapping) or self.view.get("version") != 1:\n                raise ValueError("stored user input view is invalid")\n            try:\n                canonical_sha256(self.view)\n            except (TypeError, ValueError) as error:\n                raise ValueError("stored user input view is invalid") from error\n            object.__setattr__(self, "view", dict(self.view))\n''',
)
insert_before_once(
    path,
    "\n\n@dataclass(frozen=True, slots=True)\nclass TranscriptChunk:\n",
    '''\n\n@dataclass(frozen=True, slots=True)\nclass SessionTurnRef:\n    session_id: str\n    sequence: int\n    execution_id: str\n\n    def __post_init__(self) -> None:\n        if not self.session_id or not self.execution_id or self.sequence < 1:\n            raise ValueError("session turn reference is invalid")\n\n\n@dataclass(frozen=True, slots=True)\nclass SessionTurnCommitRef:\n    session_id: str\n    sequence: int\n    execution_id: str\n    start_message_index: int\n    end_message_index: int\n\n    def __post_init__(self) -> None:\n        if (\n            not self.session_id\n            or not self.execution_id\n            or self.sequence < 1\n            or self.start_message_index < 0\n            or self.end_message_index <= self.start_message_index\n        ):\n            raise ValueError("session turn commit reference is invalid")\n''',
)
replace_once(
    path,
    '''    history_quality: str = "complete"\n    history_id: str | None = None\n\n    def __post_init__(self) -> None:\n''',
    '''    history_quality: str = "complete"\n    history_id: str | None = None\n    timeline_parent_session_id: str | None = None\n    timeline_parent_turn_sequence: int = 0\n\n    def __post_init__(self) -> None:\n''',
)
replace_once(
    path,
    '''        if self.history_quality not in {"complete", "conservative"}:\n            raise ValueError("session history quality summary is invalid")\n''',
    '''        if self.history_quality not in {"complete", "conservative"}:\n            raise ValueError("session history quality summary is invalid")\n        if self.timeline_parent_session_id is None:\n            if self.timeline_parent_turn_sequence != 0:\n                raise ValueError("root session timeline cannot have an ancestor cutoff")\n        elif (\n            not self.timeline_parent_session_id\n            or self.timeline_parent_session_id == self.session_id\n            or self.timeline_parent_turn_sequence < 1\n        ):\n            raise ValueError("session timeline ancestor is invalid")\n''',
)
insert_after_once(
    path,
    '''    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None: ...\n''',
    '''    async def timeline_head(self, session_id: str, *, tenant_id: str) -> int: ...\n    async def list_timeline_turns(\n        self,\n        session_id: str,\n        *,\n        tenant_id: str,\n        start_sequence: int,\n        end_sequence: int,\n    ) -> tuple[SessionTurnRef, ...]: ...\n    async def list_timeline_commits(\n        self,\n        session_id: str,\n        *,\n        tenant_id: str,\n        start_sequence: int,\n        end_sequence: int,\n    ) -> tuple[SessionTurnCommitRef, ...]: ...\n    async def commit_timeline_turn_in_transaction(\n        self,\n        transaction: StateTransaction,\n        session_id: str,\n        *,\n        tenant_id: str,\n        execution_id: str,\n        start_message_index: int | None,\n        end_message_index: int,\n    ) -> SessionTurnCommitRef: ...\n''',
)

# ---------------------------------------------------------------------------
# Codec: omit the derived view when absent so existing persisted v1 fixtures
# stay byte-for-byte stable; accept/read it when new Runtime data includes it.
# ---------------------------------------------------------------------------
path = "linktools-ai/src/linktools/ai/runtime/state/_codec.py"
insert_before_once(
    path,
    "\n\ndef _decode_v1_stored_user_input(\n",
    """


def _encode_v1_stored_user_input(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, StoredUserInput):
        raise TypeError("V1 stored_user_input encoder received the wrong type")
    encoded: dict[str, JsonValue] = {
        "codec": _encode_domain(value.codec, codec, persisted=persisted),
        "payload": _encode_domain(value.payload, codec, persisted=persisted),
    }
    if value.view is not None:
        encoded["view"] = _encode_domain(value.view, codec, persisted=persisted)
    return encoded
""",
)
replace_once(
    path,
    """    _require_contract_fields(
        raw_fields,
        frozenset({"codec", "payload"}),
        persisted=persisted,
    )
""",
    """    required = frozenset({"codec", "payload"})
    if "view" in raw_fields:
        required = frozenset({"codec", "payload", "view"})
    _require_contract_fields(
        raw_fields,
        required,
        persisted=persisted,
    )
""",
)
replace_once(
    path,
    """    try:
        return StoredUserInput(codec_name, payload)
    except (TypeError, ValueError) as error:
""",
    """    view = (
        None
        if "view" not in raw_fields
        else _decode_domain(
            raw_fields["view"],
            Mapping[str, JsonValue] | None,
            codec,
            persisted=persisted,
        )
    )
    try:
        return StoredUserInput(
            codec_name,
            payload,
            cast(Mapping[str, JsonValue] | None, view),
        )
    except (TypeError, ValueError) as error:
""",
)
replace_once(
    path,
    """_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {
        "object_ref": _encode_v1_object_ref,
""",
    """_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {
        "object_ref": _encode_v1_object_ref,
        "stored_user_input": _encode_v1_stored_user_input,
""",
)

print("session timeline patch part 1 applied")
