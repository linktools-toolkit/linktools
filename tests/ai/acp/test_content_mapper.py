import base64

import acp.schema as schema
import pytest

from linktools.ai.acp.codec import AcpCodec
from linktools.ai.prompt import ImagePromptPart, ResourceLinkPromptPart, TextPromptPart


def test_content_mapper_preserves_order_and_binary_content() -> None:
    blocks = [
        schema.TextContentBlock(type="text", text="hello"),
        schema.ImageContentBlock(
            type="image",
            data=base64.b64encode(b"image").decode(),
            mimeType="image/png",
        ),
        schema.ResourceContentBlock(
            type="resource_link", uri="https://example.test/data", name="data"
        ),
    ]

    prompt = AcpCodec(image=True).decode_prompt(blocks)

    assert isinstance(prompt.parts[0], TextPromptPart)
    assert isinstance(prompt.parts[1], ImagePromptPart)
    assert isinstance(prompt.parts[2], ResourceLinkPromptPart)
    assert prompt.parts[1].data == b"image"


def test_content_mapper_rejects_unsupported_capability_and_invalid_base64() -> None:
    image = schema.ImageContentBlock(
        type="image", data="not-base64", mimeType="image/png"
    )

    with pytest.raises(Exception):
        AcpCodec().decode_prompt([image])
    with pytest.raises(Exception):
        AcpCodec(image=True).decode_prompt([image])
