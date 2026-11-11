#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise native content identities once per fresh Python hash seed."""

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_serializer
from pydantic.dataclasses import dataclass
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, StructuredContent, UserPromptPart

from linktools.ai.inputs import decode_user_content_payload, encode_user_content, input_intent


def native_array_order() -> bool:
    item = StructuredContent({'nested': [{'beta', 'alpha', 'gamma'}]}, content='fixed')
    expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    assert restored.value == expected
    assert input_intent((item,), ()).digest == input_intent((StructuredContent(expected, content='fixed'),), ()).digest
    return True


def stable_set_digest(model_kind: str) -> str:
    class Fields:
        values: set[str]
        ordered: list[str]
        count: int
        @field_serializer('count', when_used='json')
        def json_count(self, value: int) -> str:
            return f'count:{value}'
        @field_serializer('values', when_used='json')
        def sorted_values(self, value: set) -> list:
            return sorted(value)
    if model_kind == 'model':
        class State(Fields, BaseModel):
            pass
    else:
        State = dataclass(Fields)
    item = StructuredContent(State(values={'alpha','beta','gamma'}, ordered=['z','a'], count=1), content='fixed')
    return input_intent((item,), ()).digest


def frozen_member_digests(owner_kind: str, member_kind: str) -> dict[str, str]:
    class ItemFields:
        key: str
        groups: frozenset[str]
        @field_serializer('key', when_used='json')
        def json_key(self, value: str) -> str:
            return value.upper()
        @field_serializer('groups', when_used='json')
        def sorted_groups(self, value: frozenset[str]) -> list[str]:
            return sorted(value)
    if member_kind == 'model':
        class Item(ItemFields, BaseModel):
            model_config = ConfigDict(frozen=True)
    else:
        Item = dataclass(frozen=True)(ItemFields)
    members = {Item(key='a',groups=frozenset({'a','z'})), Item(key='b',groups=frozenset({'beta','gamma'}))}
    class StableMembers:
        @field_serializer('values', when_used='json')
        def sorted_values(self, value: set) -> list:
            return sorted(value, key=lambda item: item.key)
    result = {}
    for alias in (False, True):
        if owner_kind == 'model':
            class State(StableMembers, BaseModel):
                model_config = ConfigDict(serialize_by_alias=alias)
                values: set[Item] = Field(serialization_alias='items')
            state = State(values=members)
        elif owner_kind == 'dataclass':
            @dataclass(config=ConfigDict(serialize_by_alias=alias))
            class State(StableMembers):
                values: set[Item] = Field(serialization_alias='items')
            state = State(values=members)
        elif owner_kind == 'annotated-dataclass':
            @dataclass(config=ConfigDict(serialize_by_alias=alias))
            class State(StableMembers):
                values: Annotated[set[Item], Field(serialization_alias='items')]
            state = State(values=members)
        else:
            class State(RootModel[set[Item]]):
                @field_serializer('root', when_used='json')
                def sorted_values(self, value: set) -> list:
                    return sorted(value, key=lambda item: item.key)
            state = State(members)
        item = StructuredContent(state,content='fixed')
        restored = decode_user_content_payload(encode_user_content((item,)))[0]
        expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
        assert restored.value == expected, (owner_kind, member_kind, alias)
        expected_members = [{'key':'A','groups':['a','z']}, {'key':'B','groups':['beta','gamma']}]
        assert restored.value == (expected_members if owner_kind == 'root' else {'items' if alias else 'values':expected_members}), (owner_kind, member_kind, alias)
        result[str(alias)] = input_intent((item,), ()).digest
    return result


def frozen_member_containers(owner_kind: str) -> bool:
    class Item(BaseModel):
        model_config = ConfigDict(frozen=True)
        key: str
        groups: frozenset[str]
    members = {Item(key='a',groups=frozenset({'a','z'})), Item(key='b',groups=frozenset({'beta','gamma'}))}
    if owner_kind == 'model':
        class State(BaseModel):
            model_config = ConfigDict(serialize_by_alias=True)
            values: set[Item] = Field(serialization_alias='items')
        state = State(values=members)
    elif owner_kind == 'annotated-dataclass':
        @dataclass(config=ConfigDict(serialize_by_alias=True))
        class State:
            values: Annotated[set[Item], Field(serialization_alias='items')]
        state = State(values=members)
    else:
        state = RootModel[set[Item]](members)
    item = StructuredContent(state,content='fixed')
    restored = decode_user_content_payload(encode_user_content((item,)))[0]
    expected = json.loads(ModelMessagesTypeAdapter.dump_json([ModelRequest(parts=[UserPromptPart([item])])]))[0]['parts'][0]['content'][0]['value']
    assert restored.value == expected, owner_kind
    if owner_kind != 'root':
        assert set(restored.value) == {'items'}
    return True


def main() -> None:
    result: dict[str, object] = {'array-order': native_array_order()}
    for model_kind in ('model', 'dataclass'):
        result[f'stable-{model_kind}'] = stable_set_digest(model_kind)
    for owner_kind in ('model', 'dataclass', 'annotated-dataclass', 'root'):
        for member_kind in ('model', 'dataclass'):
            result[f'frozen-{owner_kind}-{member_kind}'] = frozen_member_digests(owner_kind, member_kind)
    for owner_kind in ('model', 'annotated-dataclass', 'root'):
        result[f'container-{owner_kind}'] = frozen_member_containers(owner_kind)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
