#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document: a retrieved knowledge chunk (the Retriever's unit of output)."""

from dataclasses import dataclass
from typing import Mapping

from .._typing import JSONValue


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    content: str
    score: "float | None"
    source: "str | None"
    metadata: "Mapping[str, JSONValue]"
    trust_level: str = "trusted"
