#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9 op 6: the RuntimeBuilder capability gate. A declared
RuntimeRequirements minimum is enforced against StorageFeatures at build time
-- a shortfall fails fast (StorageRequirementsNotMetError), never silently
degrades. The gate branches on capability scope/flag values only, never on
isinstance against a concrete backend."""

import pytest

from linktools.ai.errors import StorageRequirementsNotMetError
from linktools.ai.run.requirements import (
    RuntimeRequirements,
    enforce_storage_capability_gate,
)
from linktools.ai.storage.features import (
    CoordinationScope,
    FILE_STORAGE_FEATURES,
    SQLALCHEMY_STORAGE_FEATURES,
    StorageFeatures,
    TransactionScope,
)


def _features(**overrides) -> StorageFeatures:
    base = dict(
        transactions=TransactionScope.DATABASE,
        coordination=CoordinationScope.DISTRIBUTED,
        optimistic_concurrency=True,
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_blobs=True,
        full_text_search=False,
        semantic_search=False,
        multi_process_swarm=False,
    )
    base.update(overrides)
    return StorageFeatures(**base)


def test_no_requirements_is_a_noop():
    # The default path (requirements=None) imposes no gate -- existing callers
    # are unaffected.
    enforce_storage_capability_gate(FILE_STORAGE_FEATURES, None)
    enforce_storage_capability_gate(FILE_STORAGE_FEATURES, RuntimeRequirements())


def test_scope_meet_or_exceed_passes():
    # Storage at DISTRIBUTED satisfies a PROCESS_LOCAL requirement.
    enforce_storage_capability_gate(
        _features(coordination=CoordinationScope.DISTRIBUTED),
        RuntimeRequirements(coordination=CoordinationScope.PROCESS_LOCAL),
    )
    # Equal scope passes.
    enforce_storage_capability_gate(
        _features(transactions=TransactionScope.DATABASE),
        RuntimeRequirements(transactions=TransactionScope.DATABASE),
    )


def test_coordination_shortfall_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="coordination"):
        enforce_storage_capability_gate(
            _features(coordination=CoordinationScope.PROCESS_LOCAL),
            RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED),
        )


def test_transaction_shortfall_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="transaction"):
        enforce_storage_capability_gate(
            _features(transactions=TransactionScope.PROCESS_LOCAL),
            RuntimeRequirements(transactions=TransactionScope.DATABASE),
        )


def test_missing_bool_capability_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="fencing"):
        enforce_storage_capability_gate(
            _features(fencing=False),
            RuntimeRequirements(fencing=True),
        )


def test_gate_through_runtime_build_rejects_process_local_for_distributed(tmp_path):
    # The gate is wired into Runtime.build: a FilesystemStorage (process-local)
    # rejected when the caller declares a distributed-coordination requirement.
    from linktools.ai.errors import StorageRequirementsNotMetError as _Err
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.facade import FilesystemStorage

    rt_storage = FilesystemStorage(root=tmp_path)
    with pytest.raises(_Err, match="coordination"):
        Runtime.build(
            storage=rt_storage,
            requirements=RuntimeRequirements(
                coordination=CoordinationScope.DISTRIBUTED
            ),
        )


def test_gate_through_runtime_build_passes_when_met(tmp_path):
    # No requirements -> build succeeds (backward compatible). A requirement
    # the reference Filesystem storage DOES meet (process-local coordination)
    # also succeeds.
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.facade import FilesystemStorage

    rt_storage = FilesystemStorage(root=tmp_path)
    Runtime.build(storage=rt_storage)  # no gate
    Runtime.build(
        storage=rt_storage,
        requirements=RuntimeRequirements(
            coordination=CoordinationScope.PROCESS_LOCAL
        ),
    )


# --- §4.11 multi-worker Job + multi-process Swarm topology rules ---


def test_for_multi_worker_jobs_encodes_the_section_4_11_rule():
    req = RuntimeRequirements.for_multi_worker_jobs()
    assert req.coordination is CoordinationScope.DISTRIBUTED
    assert req.leasing is True
    assert req.fencing is True


def test_for_multi_process_swarm_encodes_the_section_4_11_rule():
    req = RuntimeRequirements.for_multi_process_swarm()
    assert req.coordination is CoordinationScope.DISTRIBUTED
    assert req.fencing is True
    assert req.multi_process_swarm is True


def test_multi_worker_jobs_rejects_process_local_storage(tmp_path):
    # §7.7 external-adapter step 5: remove the distributed capability and a
    # multi-worker Job topology must FAIL TO CONSTRUCT against process-local
    # storage -- no silent fallback to the in-process coordinator.
    from linktools.ai.errors import StorageRequirementsNotMetError as _Err
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.facade import FilesystemStorage

    rt_storage = FilesystemStorage(root=tmp_path)  # process-local coordination
    with pytest.raises(_Err, match="coordination"):
        Runtime.build(
            storage=rt_storage,
            requirements=RuntimeRequirements.for_multi_worker_jobs(),
        )


def test_multi_process_swarm_rejects_process_local_coordination():
    # The multi_process_swarm flag requires DISTRIBUTED coordination even when
    # the scalar coordination requirement is left at its NONE default -- a
    # bare RuntimeRequirements(multi_process_swarm=True) against process-local
    # storage is rejected by the swarm-specific check (not the scalar one).
    with pytest.raises(StorageRequirementsNotMetError, match="DISTRIBUTED"):
        enforce_storage_capability_gate(
            _features(coordination=CoordinationScope.PROCESS_LOCAL),
            RuntimeRequirements(multi_process_swarm=True),
        )


def test_multi_process_swarm_rejects_distributed_without_fencing():
    # Meets the DISTRIBUTED scope but lacks fencing -> still rejected.
    with pytest.raises(StorageRequirementsNotMetError, match="fencing"):
        enforce_storage_capability_gate(
            _features(
                coordination=CoordinationScope.DISTRIBUTED, fencing=False
            ),
            RuntimeRequirements(multi_process_swarm=True),
        )


def test_multi_process_swarm_accepts_distributed_with_fencing():
    enforce_storage_capability_gate(
        _features(
            coordination=CoordinationScope.DISTRIBUTED, fencing=True
        ),
        RuntimeRequirements(multi_process_swarm=True),
    )
