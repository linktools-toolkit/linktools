#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input identity comparisons follow the durable request's declared projection."""

import pytest
from pydantic_ai.messages import BinaryContent, StructuredContent, TextContent

from linktools.ai.inputs import input_intent


def test_legacy_projection_matches_saved_text_without_losing_current_structure() -> None:
    value = StructuredContent({'count': 1}, content='fixed rendering', tag='state')
    old_text = TextContent('fixed rendering')
    assert input_intent((value,), (), identity_version=1) == input_intent((old_text,), ())
    assert input_intent((value,), ()).digest != input_intent((old_text,), ()).digest
    assert value.value == {'count': 1}
    assert value.tag == 'state'


@pytest.mark.parametrize('value', ['text', (TextContent('text'),), (BinaryContent(b'body', media_type='text/plain'),)])
def test_unaffected_inputs_keep_their_existing_digest(value: object) -> None:
    assert input_intent(value, ('file.txt',), identity_version=1) == input_intent(value, ('file.txt',))


@pytest.mark.parametrize('version', [True, 0, 3, '1'])
def test_unknown_input_identity_projection_is_rejected(version: object) -> None:
    with pytest.raises(ValueError):
        input_intent('text', (), identity_version=version)
