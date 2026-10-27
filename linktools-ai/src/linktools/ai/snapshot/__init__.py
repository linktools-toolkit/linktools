#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent snapshot domain over shared persistence."""

from ._snapshot import (RestoredSnapshot, Snapshot, SnapshotExclusiveGuard, SnapshotLimits, SnapshotTargetInfo)

__all__ = ['RestoredSnapshot', 'Snapshot', 'SnapshotExclusiveGuard', 'SnapshotLimits', 'SnapshotTargetInfo']
