#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy-backed storage. Importing this package pulls in SQLAlchemy, which
is an optional dependency -- prefer ``from linktools.ai.storage import
SqlAlchemyStorage`` (lazy) unless you specifically want eager import."""

from .facade import SqlAlchemyStorage

__all__ = ["SqlAlchemyStorage"]
