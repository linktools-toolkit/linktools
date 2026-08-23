#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_history.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    marker = "except AIError:\n            raise\n        except (TypeError, ValueError) as error:"
    if text.count(marker) >= 3:
        print("transcript schema errors already preserved")
        return

    old = '''        try:\n            head = _decode_enveloped_domain(record.data, TranscriptHeadRecord)\n        except (TypeError, ValueError, AIError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
    new = '''        try:\n            head = _decode_enveloped_domain(record.data, TranscriptHeadRecord)\n        except AIError:\n            raise\n        except (TypeError, ValueError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
    text = replace_once(text, old, new, "transcript head error contract")

    old = '''        try:\n            return _decode_enveloped_domain(record.data, TranscriptSeekRecord)\n        except (TypeError, ValueError, AIError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
    new = '''        try:\n            return _decode_enveloped_domain(record.data, TranscriptSeekRecord)\n        except AIError:\n            raise\n        except (TypeError, ValueError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
    text = replace_once(text, old, new, "transcript seek error contract")

    old = '''    def decode_chunk(self, fact: StoredFact) -> TranscriptChunk:\n        try:\n            return _decode_enveloped_domain(fact.data, TranscriptChunk)\n        except AIError as error:\n            raise ValueError("transcript chunk payload is invalid") from error\n'''
    new = '''    def decode_chunk(self, fact: StoredFact) -> TranscriptChunk:\n        try:\n            return _decode_enveloped_domain(fact.data, TranscriptChunk)\n        except AIError:\n            raise\n        except (TypeError, ValueError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
    text = replace_once(text, old, new, "transcript chunk error contract")

    PATH.write_text(text, encoding="utf-8")
    print("transcript schema error classification preserved")


def close_exception_gates() -> None:
    runpy.run_path(
        str(ROOT / ".bootstrap/close_persistence_exception_gates.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
    close_exception_gates()
