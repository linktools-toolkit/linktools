"""Canonical SQL turns retain multipart model input with either Step setting."""

import asyncio
import base64
import json

import pytest
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import SessionCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import StoredUserInput
from linktools.ai.model._attachment import is_binary_attachment_text
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _MultipartAuthorization(TenantAuthorizationPolicy):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("step_enabled", [False, True], ids=["step-off", "step-on"])
async def test_canonical_multipart_prompt_preserves_input_without_exposing_body(tmp_path, step_enabled):
    raw = b"model-only sample attachment"
    namespace = "canonical-multipart"
    storage = tmp_path / "state"
    encoded = base64.b64encode(raw).decode()
    calls = []
    marker = None

    async def respond(messages, _info):
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        multipart = prompts[0]
        assert not isinstance(multipart, str)
        nonlocal marker
        if not calls:
            before, marker, binary, after = multipart
            assert isinstance(binary, BinaryContent)
            assert is_binary_attachment_text(marker, binary)
            assert (binary.data, binary.media_type) == (raw, "text/plain")
            calls.append(binary.data)
        else:
            # The provider view retires binary bytes after the first complete
            # response; the canonical original is not lost from the archive.
            before, retained_marker, after = multipart
            assert retained_marker == marker
            calls.append(None)
        assert (before, after) == ("Before attachment", "After attachment")
        return ModelResponse(parts=[TextPart("Ready.")])

    def features():
        value = RuntimeFeatures(namespace, defaults={"execution": {"retain_history": True}})
        if step_enabled:
            value.capability(SessionCapability(checkpoints=True))
        value.agent("default", model="default", allow_tools=())
        return model_features(_Models(_UsageFunctionModel(respond)), value)

    def input_for(data):
        return ("Before attachment", BinaryContent(data, media_type="text/plain"), "After attachment")

    async def send(runtime, prompt, key):
        return await runtime.sessions.start_turn(
            "chat", prompt, agent="default", principal=runtime.default_principal, idempotency_key=key,
        )

    async def send_and_commit(runtime, prompt, key):
        frames = asyncio.Queue()
        detach = runtime.sessions.observe_turn_session("chat", frames.put_nowait)
        try:
            identity = await send(runtime, prompt, key)
            async with asyncio.timeout(10):
                while True:
                    frame = await frames.get()
                    assert frame["turn_id"] == identity
                    wire = json.dumps(frame)
                    assert raw.decode() not in wire and encoded not in wire
                    if frame["type"] == "terminal":
                        assert frame["state"] == "succeeded", frame
                        return identity
        finally:
            detach()

    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_root(storage)) as persistence,
        Runtime.open(persistence=persistence, features=features(), authorization=_MultipartAuthorization()) as runtime,
    ):
        await runtime.sessions.create_turn_session("chat", principal=runtime.default_principal)
        turn_id = await send_and_commit(runtime, input_for(raw), "multipart")

    async with (
        Persistence.open(namespace, storage=PersistenceStorage.from_root(storage)) as persistence,
        Runtime.open(persistence=persistence, features=features(), authorization=_MultipartAuthorization()) as runtime,
    ):
        result = await runtime.history.turn_result("chat", turn_id, principal=runtime.default_principal)
        assert result["conversation_committed"]
        assert result["output"] == {"text": "Ready."}
        assert result["user_input"]["prompt"] == {"kind": "items", "items": [
            {"kind": "text", "text": "Before attachment"},
            {"kind": "text", "text": "After attachment"},
        ]}
        accepted = await persistence.session_turns.get_turn(SessionTurnKey("chat", turn_id))
        stored = decode_domain(accepted["request_json"]["input"], StoredUserInput)
        (source,) = stored.payload.object_refs()
        assert await read_object(
            persistence.object_store(PersistenceDomain.EXECUTION), source.key,
            expected_digest=source.digest, expected_size=source.size,
        ) == raw
        attachment, = result["user_input"]["attachments"]
        assert attachment["media_type"] == "text/plain"
        assert attachment["size"] == len(raw)
        assert set(attachment) == {"attachment_id", "source", "media_type", "size", "position"}
        wire = json.dumps(result)
        assert raw.decode() not in wire and encoded not in wire
        assert await send(runtime, input_for(raw), "multipart") == turn_id
        assert calls == [raw]
        with pytest.raises(AIError) as conflict:
            await send(runtime, input_for(b"different sample attachment"), "multipart")
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert calls == [raw]
        assert await send_and_commit(runtime, "Use earlier input", "next") != turn_id
    assert calls == [raw, None]
