#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""JSON codec used at the execution persistence boundary."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def encode(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, datetime):
            return {"__datetime__": item.isoformat()}
        if isinstance(item, Enum):
            return item.value
        if is_dataclass(item):
            return asdict(item)
        raise TypeError(f"not JSON encodable: {type(item).__name__}")

    return json.dumps(value, default=default, ensure_ascii=False, separators=(",", ":"))


def decode(value: str) -> Any:
    def hook(item: "dict[str, Any]") -> Any:
        if set(item) == {"__datetime__"}:
            return datetime.fromisoformat(item["__datetime__"])
        return item

    return json.loads(value, object_hook=hook)
