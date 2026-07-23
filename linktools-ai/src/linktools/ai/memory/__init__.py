#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""memory: the Memory subsystem's pure domain model (MemoryRecord, MemoryMatch)
and the persistence/search Protocol (MemoryStore). The store is the single
source of truth -- search returns scored MemoryMatch results directly; the
FilesystemMemoryStore / SqlAlchemyMemoryStore backends live under
storage/filesystem and storage/sqlalchemy."""
