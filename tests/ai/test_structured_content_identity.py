#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured content retains durable meaning and request identity."""

import json
from collections.abc import Callable
from datetime import date

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_ai.messages import (
    BinaryContent, ModelMessagesTypeAdapter, ModelRequest, StructuredContent, TextContent, ToolReturn, ToolReturnPart, UserPromptPart,
)

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import decode_user_content_payload, encode_user_content, input_intent
from linktools.ai.model.messages import (
    binary_attachment_text, decode_tool_return_content, encode_tool_return_content, is_binary_attachment_text, tool_return_content_matches,
)


def test_structured_content_retains_value_and_tag_after_reload() -> None:
    original = StructuredContent({'count': 1}, content='fixed rendering', tag='state')
    restored = decode_user_content_payload(encode_user_content((original,)))[0]
    assert isinstance(restored, StructuredContent)
    assert restored.value == original.value
    assert restored.tag == original.tag
    assert restored.content == original.content


@pytest.mark.parametrize('changed', [
    StructuredContent({'count': 2}, content='fixed rendering', tag='state'),
    StructuredContent({'count': 1}, content='fixed rendering', tag='other'),
], ids=['value', 'tag'])
def test_structured_request_identity_keeps_native_meaning(changed: StructuredContent) -> None:
    original = StructuredContent({'count': 1}, content='fixed rendering', tag='state')
    assert input_intent((original,), ()).digest != input_intent((changed,), ()).digest


@pytest.mark.parametrize('route', ['input', 'tool-result'])
def test_structured_content_accepts_native_serializable_values(route: str) -> None:
    class State(BaseModel):
        count: int

    original = StructuredContent(State(count=1), content='fixed rendering', tag='state',
                                 description='Current state', metadata={'source': 'test'})
    if route == 'input':
        restored = decode_user_content_payload(encode_user_content((original,)))[0]
    else:
        restored = decode_tool_return_content(encode_tool_return_content(original))
    assert isinstance(restored, StructuredContent)
    assert restored.value == {'count': 1}
    assert restored.tag == original.tag
    assert restored.content == original.content
    assert restored.description == original.description
    assert restored.metadata == original.metadata


def test_legacy_text_content_stays_text() -> None:
    payload = {'items': [{'kind': 'text-content', 'content': 'fixed rendering', 'metadata': None}]}
    restored = decode_user_content_payload(payload)[0]
    assert type(restored) is TextContent
    assert encode_user_content((restored,)) == payload


@pytest.mark.parametrize('wrap,unwrap', [
    (lambda item: item, lambda result: result),
    (lambda item: [item], lambda result: result[0]),
    (lambda item: {'state': item}, lambda result: result['state']),
    (lambda item: ToolReturn(item), lambda result: result.return_value),
    (lambda item: ToolReturn('ok', content=[item]), lambda result: result.content[0]),
], ids=['direct', 'sequence', 'mapping', 'return-value', 'content'])
def test_tool_return_preserves_structured_content_at_supported_positions(
    wrap: Callable[[object], object], unwrap: Callable[[object], object],
) -> None:
    item = StructuredContent({'count': 1}, format=lambda value: 'custom rendering', tag='state',
                             description='Current state', metadata={'source': 'test'})
    payload = encode_tool_return_content(wrap(item))
    assert payload['version'] == 2
    restored = unwrap(decode_tool_return_content(payload))
    assert type(restored) is StructuredContent
    assert restored == item


def _native_structured_value(item: StructuredContent) -> object:
    encoded = ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])])
    return json.loads(encoded)[0]['parts'][0]['content'][0]['value']


def test_structured_native_values_match_sdk_json_representation() -> None:
    item = StructuredContent({'values': {3, 1, 2}, 'bytes': b'\xff', 'date': date(2026, 11, 8)}, content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item)
    nonfinite = StructuredContent(float('inf'), content='fixed')
    restored = decode_user_content_payload(encode_user_content((nonfinite,)))[0]
    assert restored.value is _native_structured_value(nonfinite) is None


@pytest.mark.parametrize('route', ['input', 'tool-result'])
@pytest.mark.parametrize('by_alias', [False, True], ids=['field-names', 'aliases'])
@pytest.mark.parametrize('model_kind', ['model', 'dataclass'])
def test_structured_values_preserve_native_json_only_serializers(
    route: str, by_alias: bool, model_kind: str,
) -> None:
    class CustomValue:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier

    class StateFields:
        count: int = Field(serialization_alias='total')
        custom: CustomValue = Field(serialization_alias='native_payload')

        @field_serializer('count', when_used='json')
        def encode_count(self, value: int) -> str:
            return f'count:{value}'

        @field_serializer('custom', when_used='json')
        def encode_custom(self, value: CustomValue) -> dict[str, str]:
            return {'identifier': value.identifier}

    config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=by_alias)
    if model_kind == 'model':
        class State(StateFields, BaseModel):
            model_config = config
    else:
        State = pydantic_dataclass(config=config)(StateFields)
    item = StructuredContent(State(count=1, custom=CustomValue('one')), content='fixed rendering')
    if route == 'input':
        restored = decode_user_content_payload(encode_user_content((item,)))[0]
    else:
        restored = decode_tool_return_content(encode_tool_return_content(item))
    assert restored.value == _native_structured_value(item)
    assert restored.value == {
        'total' if by_alias else 'count': 'count:1',
        'native_payload' if by_alias else 'custom': {'identifier': 'one'},
    }


@pytest.mark.parametrize('field,value', [('value', float('inf')), ('content', 42), ('tag', []),
                                        ('description', 42)])
def test_structured_content_rejects_invalid_recorded_fields(field: str, value: object) -> None:
    payload = encode_user_content((StructuredContent({'count': 1}, content='fixed'),))
    payload['items'][0][field] = value
    with pytest.raises((AIError, TypeError, ValueError)):
        decode_user_content_payload(payload)


def test_structured_content_can_describe_binary_without_losing_its_identity() -> None:
    binary = BinaryContent(b'image', media_type='image/png', identifier='attachment')
    generated = binary_attachment_text(binary)
    structured = StructuredContent({'count': 1}, content=generated.content, metadata=generated.metadata)
    assert is_binary_attachment_text(generated, binary)
    assert is_binary_attachment_text(structured, binary)
    assert input_intent((structured, binary), ()).digest != input_intent((generated, binary), ()).digest


@pytest.mark.parametrize('version', [0, 3, True, '2'])
def test_tool_return_rejects_unknown_envelope_versions(version: object) -> None:
    payload = encode_tool_return_content('ok')
    payload['version'] = version
    with pytest.raises(AIError) as error:
        decode_tool_return_content(payload)
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_structured_metadata_cannot_be_reinterpreted_as_retired_reference() -> None:
    item = StructuredContent({'count': 1}, content='fixed',
                             metadata={'referenced_content': 'application metadata'})
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert type(restored) is StructuredContent
    assert restored.metadata == item.metadata


@pytest.mark.asyncio
async def test_structured_attachment_label_keeps_its_native_value_after_materialization() -> None:
    from linktools.ai.core import PromptLimits
    from linktools.ai.inputs import ExecutionInputMaterializer

    binary = BinaryContent(b'image', media_type='image/png', identifier='attachment')
    generated = binary_attachment_text(binary)
    structured = StructuredContent({'count': 1}, content=generated.content, metadata=generated.metadata)
    materialized = await ExecutionInputMaterializer(None, PromptLimits()).materialize((structured, binary), ())
    assert len(materialized) == 2
    assert type(materialized[0]) is StructuredContent
    assert materialized[0].value == {'count': 1}
    assert materialized[1] is binary


@pytest.mark.parametrize('node,wrap', [
    ({'type': 'sequence', 'items': [{'type': 'text-content', 'value': {
        'kind': 'text-content', 'content': 'fixed', 'metadata': None,
    }}]}, lambda item: [item]),
    ({'type': 'mapping', 'items': {'state': {'type': 'text-content', 'value': {
        'kind': 'text-content', 'content': 'fixed', 'metadata': None,
    }}}}, lambda item: {'state': item}),
    ({'type': 'tool-return', 'return_value': {'type': 'text-content', 'value': {
        'kind': 'text-content', 'content': 'fixed', 'metadata': None,
    }}, 'content': None, 'metadata': None, 'tools': None}, lambda item: ToolReturn(item)),
    ({'type': 'tool-return', 'return_value': {'type': 'scalar', 'value': 'ok'},
      'content': [{'kind': 'text-content', 'content': 'fixed', 'metadata': None}],
      'metadata': None, 'tools': None}, lambda item: ToolReturn('ok', content=[item])),
], ids=['sequence', 'mapping', 'return-value', 'content'])
def test_legacy_tool_result_projection_follows_only_typed_content_nodes(
    node: dict[str, object], wrap: Callable[[object], object],
) -> None:
    payload = {'contract': 'linktools.tool-return', 'version': 1, 'value': node}
    item = StructuredContent({'count': 1}, content='fixed', tag='state')
    assert tool_return_content_matches(wrap(item), payload)
    assert tool_return_content_matches(wrap(StructuredContent({'count': 2}, content='fixed', tag='other')), payload)
    assert not tool_return_content_matches(wrap(StructuredContent({'count': 1}, content='changed')), payload)


def test_legacy_tool_result_projection_does_not_reinterpret_business_json() -> None:
    original = {'kind': 'structured-content', 'value': {'count': 1}, 'content': 'fixed', 'metadata': None}
    payload = encode_tool_return_content(original)
    payload['version'] = 1
    assert tool_return_content_matches(original, payload)
    assert not tool_return_content_matches(dict(original, value={'count': 2}), payload)



def test_structured_tool_content_is_not_lowered_as_generated_attachment_text() -> None:
    from pydantic_ai.models.test import TestModel
    from linktools.ai.runtime._harness import SummaryProviderModel

    binary = BinaryContent(b'image', media_type='image/png', identifier='attachment')
    generated = binary_attachment_text(binary)
    structured = StructuredContent({'count': 1}, content=generated.content, tag='state', metadata=generated.metadata)
    message = ModelRequest(parts=[ToolReturnPart('tool', [structured, generated, binary], tool_call_id='call')])
    projected = SummaryProviderModel(TestModel()).project_messages([message])[0].parts[0].content
    assert type(projected[0]) is StructuredContent
    assert projected[0].value == {'count': 1}
    assert projected[0].tag == 'state'
    assert projected[1] == generated.content
    assert projected[2] is binary



@pytest.mark.parametrize('value', [
    {1: 'number', '1': 'text'}, {'nested': [{1: 'number', '1': 'text'}]},
])
def test_structured_direct_mapping_keys_cannot_collide(value: object) -> None:
    with pytest.raises(AIError) as error:
        encode_user_content((StructuredContent(value, content='fixed'),))
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_native_array_order_is_the_structured_request_identity() -> None:
    import os
    import subprocess
    import sys

    script = """
import json
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, StructuredContent, UserPromptPart
from linktools.ai.inputs import decode_user_content_payload, encode_user_content, input_intent
item = StructuredContent({'nested': [{'beta', 'alpha', 'gamma'}]}, content='fixed')
expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
restored = decode_user_content_payload(encode_user_content((item,)))[0]
assert restored.value == expected
assert input_intent((item,), ()).digest == input_intent((StructuredContent(expected, content='fixed'),), ()).digest
"""
    for seed in ('1', '77'):
        subprocess.check_call([sys.executable, '-c', script], env=dict(os.environ, PYTHONHASHSEED=seed))
    first = StructuredContent(['alpha', 'beta'], content='fixed')
    second = StructuredContent(['beta', 'alpha'], content='fixed')
    assert input_intent((first,), ()).digest != input_intent((second,), ()).digest



def test_native_model_serializer_order_does_not_become_set_order() -> None:
    class State(BaseModel):
        values: list[str]

        @field_serializer('values', when_used='json')
        def ordered_values(self, value: list[str]) -> list[str]:
            return sorted(value, reverse=True)

    item = StructuredContent(State(values=['alpha', 'beta']), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item) == {'values': ['beta', 'alpha']}


def test_structured_native_serialization_does_not_change_plain_tool_snapshots() -> None:
    class State(BaseModel):
        count: int = Field(serialization_alias='total')

        @field_serializer('count', when_used='json')
        def json_count(self, value: int) -> str:
            return f'count:{value}'

    state = State(count=1)
    assert decode_tool_return_content(encode_tool_return_content(state)) == {'total': 1}
    structured = StructuredContent(state, content='fixed')
    restored = decode_tool_return_content(encode_tool_return_content(structured))
    assert restored.value == _native_structured_value(structured) == {'count': 'count:1'}



@pytest.mark.parametrize('model_kind', ['model', 'dataclass'])
def test_native_stable_set_serializers_keep_identity_across_process_hash_seeds(model_kind: str) -> None:
    import os
    import subprocess
    import sys

    script = """
import sys
from pydantic import BaseModel, field_serializer
from pydantic.dataclasses import dataclass
from pydantic_ai.messages import StructuredContent
from linktools.ai.inputs import input_intent
class Fields:
    values: set[str]
    ordered: list[str]
    count: int
    @field_serializer('count', when_used='json')
    def json_count(self, value):
        return f'count:{value}'
    @field_serializer('values', when_used='json')
    def sorted_values(self, value):
        return sorted(value)
if sys.argv[1] == 'model':
    class State(Fields, BaseModel):
        pass
else:
    State = dataclass(Fields)
item = StructuredContent(State(values={'alpha','beta','gamma'}, ordered=['z','a'], count=1), content='fixed')
print(input_intent((item,), ()).digest)
"""
    digests = [subprocess.check_output([sys.executable, '-c', script, model_kind],
                                      env=dict(os.environ, PYTHONHASHSEED=seed), text=True).strip()
               for seed in ('1', '77')]
    assert digests[0] == digests[1]


def test_native_model_serialized_mapping_keys_cannot_collide() -> None:
    class State(BaseModel):
        values: dict[int | str, str]

    item = StructuredContent(State(values={1: 'number', '1': 'text'}), content='fixed')
    with pytest.raises(AIError) as error:
        encode_user_content((item,))
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_native_set_serializer_can_choose_non_array_representation() -> None:
    class State(BaseModel):
        values: set[str]

        @field_serializer('values', when_used='json')
        def as_object(self, value: set[str]) -> dict[str, list[str]]:
            return {'ordered': sorted(value, reverse=True)}

    item = StructuredContent(State(values={'alpha', 'beta'}), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item) == {'values': {'ordered': ['beta', 'alpha']}}


@pytest.mark.parametrize('owner_kind', ['model', 'dataclass', 'annotated-dataclass', 'root'])
@pytest.mark.parametrize('member_kind', ['model', 'dataclass'])
def test_native_frozen_members_with_stable_serializers_keep_identity_across_process_hash_seeds(owner_kind: str, member_kind: str) -> None:
    import os
    import subprocess
    import sys

    script = """
from __future__ import annotations
import json
import sys
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_serializer
from pydantic.dataclasses import dataclass
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, StructuredContent, UserPromptPart
from linktools.ai.inputs import decode_user_content_payload, encode_user_content, input_intent
class ItemFields:
    key: str
    groups: frozenset[str]
    @field_serializer('key', when_used='json')
    def json_key(self, value):
        return value.upper()
    @field_serializer('groups', when_used='json')
    def sorted_groups(self, value):
        return sorted(value)
if sys.argv[2] == 'model':
    class Item(ItemFields, BaseModel):
        model_config = ConfigDict(frozen=True)
else:
    Item = dataclass(frozen=True)(ItemFields)
members = {Item(key='a',groups=frozenset({'a','z'})), Item(key='b',groups=frozenset({'beta','gamma'}))}
class StableMembers:
    @field_serializer('values', when_used='json')
    def sorted_values(self, value):
        return sorted(value, key=lambda item: item.key)
result = {}
for alias in (False, True):
    if sys.argv[1] == 'model':
        class State(StableMembers, BaseModel):
            model_config = ConfigDict(serialize_by_alias=alias)
            values: set[Item] = Field(serialization_alias='items')
        state = State(values=members)
    elif sys.argv[1] == 'dataclass':
        @dataclass(config=ConfigDict(serialize_by_alias=alias))
        class State(StableMembers):
            values: set[Item] = Field(serialization_alias='items')
        state = State(values=members)
    elif sys.argv[1] == 'annotated-dataclass':
        @dataclass(config=ConfigDict(serialize_by_alias=alias))
        class State(StableMembers):
            values: Annotated[set[Item], Field(serialization_alias='items')]
        state = State(values=members)
    else:
        class State(RootModel[set[Item]]):
            @field_serializer('root', when_used='json')
            def sorted_values(self, value):
                return sorted(value, key=lambda item: item.key)
        state = State(members)
    item = StructuredContent(state,content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
    assert restored.value == expected
    expected_members = [{'key':'A','groups':['a','z']}, {'key':'B','groups':['beta','gamma']}]
    assert restored.value == (expected_members if sys.argv[1] == 'root' else {'items' if alias else 'values':expected_members})
    result[str(alias)] = input_intent((item,), ()).digest
print(json.dumps(result,sort_keys=True))
"""
    digests = [subprocess.check_output([sys.executable, '-c', script, owner_kind, member_kind],
                                      env=dict(os.environ, PYTHONHASHSEED=seed), text=True).strip()
               for seed in ('1', '77')]
    assert digests[0] == digests[1]


@pytest.mark.parametrize('frozen_members', [False, True])
def test_native_set_serializer_can_choose_unrelated_ordered_array(frozen_members: bool) -> None:
    class Item(BaseModel):
        model_config = ConfigDict(frozen=True)
        key: str

    class State(BaseModel):
        values: set[Item] | set[str]

        @field_serializer('values', when_used='json')
        def unrelated_values(self, value: object) -> list[str]:
            return ['z', 'x']

    values = {Item(key='a'), Item(key='b')} if frozen_members else {'a', 'b'}
    item = StructuredContent(State(values=values), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item) == {'values': ['z', 'x']}


def test_native_model_serializer_keeps_its_different_output_shape() -> None:
    from pydantic import model_serializer

    class Item(BaseModel):
        model_config = ConfigDict(frozen=True)
        key: str

    class State(BaseModel):
        values: set[Item]

        @model_serializer(when_used='json')
        def explicit_output(self) -> dict[str, list[str]]:
            return {'explicit': ['z', 'a']}

    item = StructuredContent(State(values={Item(key='a'), Item(key='b')}), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item) == {'explicit': ['z', 'a']}


def test_native_set_serializer_keeps_its_explicit_member_order() -> None:
    class Item(BaseModel):
        model_config = ConfigDict(frozen=True)
        key: str
        ordered: tuple[str, ...]
        unordered: frozenset[str]

    class State(BaseModel):
        values: set[Item]

        @field_serializer('values', when_used='json')
        def reversed_members(self, value: set[Item]) -> list[dict[str, object]]:
            return [item.model_dump(mode='json') for item in sorted(value, key=lambda item: item.key, reverse=True)]

    item = StructuredContent(State(values={
        Item(key='a', ordered=('z', 'x'), unordered=frozenset({'alpha', 'beta'})),
        Item(key='b', ordered=('q', 'c'), unordered=frozenset({'one', 'two'})),
    }), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item)
    assert [member['key'] for member in restored.value['values']] == ['b', 'a']
    assert [member['ordered'] for member in restored.value['values']] == [['q', 'c'], ['z', 'x']]


@pytest.mark.parametrize('value,valid_json', [(object(), False), (float('inf'), True)], ids=['unsupported', 'native-null'])
def test_native_json_serializer_outcome_is_preserved(value: object, valid_json: bool) -> None:
    from pydantic import model_serializer

    class Item(BaseModel):
        model_config = ConfigDict(frozen=True)
        key: str

    class State(BaseModel):
        values: set[Item]

        @model_serializer(when_used='json')
        def serialized_output(self) -> object:
            return value

    item = StructuredContent(State(values={Item(key='a')}), content='fixed')
    if valid_json:
        restored = decode_user_content_payload(encode_user_content((item,)))[0]
        assert restored.value is _native_structured_value(item) is None
    else:
        with pytest.raises(AIError) as error:
            encode_user_content((item,))
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID



def test_native_mapping_serializer_can_preserve_colliding_keys_as_pairs() -> None:
    class State(BaseModel):
        values: dict[int | str, str]

        @field_serializer('values', when_used='json')
        def pairs(self, value: dict[int | str, str]) -> list[list[object]]:
            return [[key, item] for key, item in value.items()]

    item = StructuredContent(State(values={1: 'number', '1': 'text'}), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item) == {'values': [[1, 'number'], ['1', 'text']]}


def test_structured_scalar_enums_are_portable_native_json() -> None:
    from enum import Enum, IntEnum
    from linktools.ai.storage import StoredPayload

    class Label(str, Enum):
        READY = 'ready'

    class Count(IntEnum):
        ONE = 1

    for value in (Label.READY, Count.ONE):
        item = StructuredContent(value, content='fixed')
        stored = StoredPayload.inline_json(encode_user_content((item,)))
        restored = decode_user_content_payload(stored.decode())[0]
        assert restored.value == _native_structured_value(item)
        assert type(restored.value) in (str, int)


def test_native_enum_mapping_keys_are_portable_json() -> None:
    from enum import Enum
    from linktools.ai.storage import StoredPayload

    class Label(str, Enum):
        READY = 'ready'

    item = StructuredContent({Label.READY: 1}, content='fixed')
    stored = StoredPayload.inline_json(encode_user_content((item,)))
    restored = decode_user_content_payload(stored.decode())[0]
    assert restored.value == _native_structured_value(item) == {'ready': 1}


def test_native_duplicated_serialization_aliases_cannot_drop_a_value() -> None:
    class State(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)
        first: int = Field(serialization_alias='shared')
        second: int = Field(serialization_alias='shared')

    item = StructuredContent(State(first=1, second=2), content='fixed')
    with pytest.raises(AIError) as error:
        encode_user_content((item,))
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_native_owner_temporal_configuration_is_preserved() -> None:
    from datetime import datetime

    class State(BaseModel):
        model_config = ConfigDict(ser_json_temporal='seconds')
        values: set[datetime]

    item = StructuredContent(State(values={datetime(2026, 11, 8), datetime(2020, 1, 1)}), content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == _native_structured_value(item)
    assert sorted(restored.value['values']) == [1577836800.0, 1794096000.0]


def test_native_nonfinite_json_constants_are_rejected() -> None:
    class State(BaseModel):
        model_config = ConfigDict(ser_json_inf_nan='constants')
        value: float

    item = StructuredContent(State(value=float('inf')), content='fixed')
    with pytest.raises(AIError) as error:
        encode_user_content((item,))
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize('owner_kind', ['model', 'annotated-dataclass', 'root'])
def test_native_frozen_member_containers_keep_sdk_values_and_order(owner_kind: str) -> None:
    import os
    import subprocess
    import sys

    script = """
from __future__ import annotations
import json
import sys
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.dataclasses import dataclass
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, StructuredContent, UserPromptPart
from linktools.ai.inputs import decode_user_content_payload, encode_user_content
class Item(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str
    groups: frozenset[str]
members = {Item(key='a',groups=frozenset({'a','z'})), Item(key='b',groups=frozenset({'beta','gamma'}))}
if sys.argv[1] == 'model':
    class State(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)
        values: set[Item] = Field(serialization_alias='items')
    state = State(values=members)
elif sys.argv[1] == 'annotated-dataclass':
    @dataclass(config=ConfigDict(serialize_by_alias=True))
    class State:
        values: Annotated[set[Item], Field(serialization_alias='items')]
    state = State(values=members)
else:
    state = RootModel[set[Item]](members)
item = StructuredContent(state,content='fixed')
restored = decode_user_content_payload(encode_user_content((item,)))[0]
expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
assert restored.value == expected
if sys.argv[1] != 'root':
    assert set(restored.value) == {'items'}
"""
    for seed in ('1', '77'):
        subprocess.check_call([sys.executable, '-c', script, owner_kind],
                              env=dict(os.environ, PYTHONHASHSEED=seed))
