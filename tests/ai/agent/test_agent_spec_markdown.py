import pytest

from linktools.ai.agent import parse_agent_spec_markdown


def test_public_agent_spec_markdown_parser_requires_explicit_id():
    content = "---\nmodel:\n  primary: test\n---\nhello"
    spec = parse_agent_spec_markdown(content, agent_id="catalog-id")
    assert spec.id == "catalog-id"


def test_public_agent_spec_markdown_parser_is_strict():
    with pytest.raises(Exception):
        parse_agent_spec_markdown("---\nmodel:\n  primary: test\nunknown: true\n---\nhello", agent_id="a")
