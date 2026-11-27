#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input intent identifies canonical user content and attachment paths."""

from pydantic_ai.messages import StructuredContent, TextContent

from linktools.ai.inputs import input_intent


def test_structured_input_identity_includes_structure_not_only_rendering() -> None:
    value = StructuredContent({'count': 1}, content='fixed rendering', tag='state')
    changed = StructuredContent({'count': 2}, content='fixed rendering', tag='state')
    plain = TextContent('fixed rendering')
    intent = input_intent((value,), ())
    assert intent == input_intent((value,), ())
    assert intent.digest != input_intent((changed,), ()).digest
    assert intent.digest != input_intent((plain,), ()).digest
    assert intent.digest != input_intent((value,), ('file.txt',)).digest
