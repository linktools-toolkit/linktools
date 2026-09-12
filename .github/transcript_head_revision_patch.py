#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, got {count}")
    return text.replace(old, new, 1)


path = Path("linktools-ai/src/linktools/ai/runtime/state/_contracts.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    message_count: int
    chunk_count: int
    quality: HistoryQuality
    revision: int

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.message_count,
                self.chunk_count,
                self.revision,
            )
        ):
            raise ValueError("transcript head counts cannot be negative")
''',
    '''    message_count: int
    chunk_count: int
    quality: HistoryQuality

    def __post_init__(self) -> None:
        if self.message_count < 0 or self.chunk_count < 0:
            raise ValueError("transcript head counts cannot be negative")
''',
    "TranscriptHeadRecord revision",
)
path.write_text(text, encoding="utf-8")

path = Path("linktools-ai/src/linktools/ai/runtime/state/_history.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''            0,
            0,
            HistoryQuality.COMPLETE,
            0,
        )
''',
    '''            0,
            0,
            HistoryQuality.COMPLETE,
        )
''',
    "empty transcript head",
)
text = replace_once(
    text,
    '''            quality=base_head.quality if quality is None else quality,
            revision=base_head.revision + 1,
        )
''',
    '''            quality=base_head.quality if quality is None else quality,
        )
''',
    "transcript head append revision",
)
path.write_text(text, encoding="utf-8")

path = Path("linktools-ai/src/linktools/ai/runtime/state/_repositories.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''                        0,
                        0,
                        HistoryQuality.COMPLETE,
                        0,
                    ),
''',
    '''                        0,
                        0,
                        HistoryQuality.COMPLETE,
                    ),
''',
    "conversation history transcript head",
)
text = replace_once(
    text,
    '''        history_id,
        0,
        0,
        HistoryQuality.COMPLETE,
        0,
    )
''',
    '''        history_id,
        0,
        0,
        HistoryQuality.COMPLETE,
    )
''',
    "empty conversation transcript head",
)
path.write_text(text, encoding="utf-8")

path = Path("tests/ai/test_durable_model_convergence.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from linktools.ai.runtime.state._contracts import ArtifactRecord, ContextProjection\n",
    '''from linktools.ai.runtime.state._contracts import (
    ArtifactRecord,
    ContextProjection,
    HistoryQuality,
    TranscriptHeadRecord,
    TranscriptOwnerDomain,
)
''',
    "durable convergence imports",
)
text += '''\n\ndef test_transcript_head_uses_stored_record_as_its_only_revision_owner() -> None:\n    head = TranscriptHeadRecord(\n        TranscriptOwnerDomain.EXECUTION,\n        "run",\n        3,\n        1,\n        HistoryQuality.COMPLETE,\n    )\n\n    payload = _encode_persisted_domain(head)\n\n    assert set(payload["fields"]) == {\n        "owner_domain",\n        "owner_id",\n        "message_count",\n        "chunk_count",\n        "quality",\n    }\n'''
path.write_text(text, encoding="utf-8")
