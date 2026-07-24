#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the RuntimeBuilder capability gate. A declared
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
    StorageComponent,
    StorageFeatures,
    TransactionScope,
)


def _features(**overrides) -> StorageFeatures:
    base = dict(
        transaction_scope=TransactionScope.DATABASE,
        transactional_components=frozenset(StorageComponent),
        coordination_scope=CoordinationScope.DISTRIBUTED,
        optimistic_concurrency=frozenset(StorageComponent),
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_artifacts=True,
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
        _features(coordination_scope=CoordinationScope.DISTRIBUTED),
        RuntimeRequirements(coordination=CoordinationScope.PROCESS_LOCAL),
    )
    # Equal scope passes.
    enforce_storage_capability_gate(
        _features(transaction_scope=TransactionScope.DATABASE),
        RuntimeRequirements(transactions=TransactionScope.DATABASE),
    )


def test_coordination_shortfall_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="coordination"):
        enforce_storage_capability_gate(
            _features(coordination_scope=CoordinationScope.PROCESS_LOCAL),
            RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED),
        )


def test_transaction_shortfall_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="transaction"):
        enforce_storage_capability_gate(
            _features(transaction_scope=TransactionScope.PROCESS_LOCAL),
            RuntimeRequirements(transactions=TransactionScope.DATABASE),
        )


def test_missing_bool_capability_rejected():
    with pytest.raises(StorageRequirementsNotMetError, match="fencing"):
        enforce_storage_capability_gate(
            _features(fencing=False),
            RuntimeRequirements(fencing=True),
        )


def test_gate_through_runtime_build_rejects_process_local_for_distributed(tmp_path):
    # The gate is wired into build_runtime: a FilesystemStorage (process-local)
    # rejected when the caller declares a distributed-coordination requirement.
    from linktools.ai.errors import StorageRequirementsNotMetError as _Err
    from linktools.ai.runtime import Runtime, build_runtime
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    rt_storage = FilesystemStorage(root=tmp_path)
    with pytest.raises(_Err, match="coordination"):
        build_runtime(
            storage=rt_storage,
            requirements=RuntimeRequirements(
                coordination=CoordinationScope.DISTRIBUTED
            ),
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(rt_storage),
        )


def test_gate_through_runtime_build_passes_when_met(tmp_path):
    # No requirements -> build succeeds (backward compatible). A requirement
    # the reference Filesystem storage DOES meet (process-local coordination)
    # also succeeds.
    from linktools.ai.runtime import Runtime, build_runtime
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    rt_storage = FilesystemStorage(root=tmp_path)
    build_runtime(
        storage=rt_storage,
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(rt_storage),
    )  # no gate
    build_runtime(
        storage=rt_storage,
        requirements=RuntimeRequirements(
            coordination=CoordinationScope.PROCESS_LOCAL
        ),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(rt_storage),
    )


# --- multi-worker Job topology rules + feature/object consistency ---


def test_for_multi_worker_jobs_encodes_the_section_4_8_rule():
    req = RuntimeRequirements.for_multi_worker_jobs()
    assert req.coordination is CoordinationScope.DISTRIBUTED
    assert req.leasing is True
    assert req.fencing is True


def test_derive_uses_settings_topology_via_frozen_signature(tmp_path):
    # B-C4: derive_runtime_requirements takes the frozen (*, settings,
    # dependencies) signature and derives from settings.topology -- SINGLE_PROCESS
    # imposes no minimums, MULTI_WORKER demands distributed coordination +
    # leasing + fencing. dependencies is accepted (the gate reads it for
    # real-object checks) but the capability minimums are a pure function of
    # the topology.
    from linktools.ai.runtime.builder import RuntimeSettings
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.run.requirements import (
        RuntimeTopology,
        derive_runtime_requirements,
    )

    single = derive_runtime_requirements(
        settings=RuntimeSettings(topology=RuntimeTopology.SINGLE_PROCESS),
        dependencies=RuntimeDependencies(),
    )
    assert single == RuntimeRequirements()

    # MULTI_WORKER needs a populated dependencies bundle (real Storage +
    # RunCommitCoordinator). The external-adapter in-memory storage satisfies
    # both, so derive returns the multi-worker minimums without raising.
    from external_adapter import build_in_memory_external_storage

    storage = build_in_memory_external_storage(root=tmp_path / "derive_multi")
    multi = derive_runtime_requirements(
        settings=RuntimeSettings(topology=RuntimeTopology.MULTI_WORKER),
        dependencies=RuntimeDependencies(
            storage=storage,
            run_commit_coordinator=object(),  # any non-None coordinator
        ),
    )
    assert multi == RuntimeRequirements.for_multi_worker_jobs()


def test_derive_multi_worker_refuses_dependencies_missing_storage():
    # The MULTI_WORKER real-object rule: a dependencies bundle
    # without a real Storage cannot serve a multi-worker topology (JobStore
    # rows are shared across workers). derive refuses it rather than returning
    # minimums the gate would then fail opaquely.
    from linktools.ai.runtime.builder import RuntimeSettings
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.errors import StorageRequirementsNotMetError
    from linktools.ai.run.requirements import (
        RuntimeTopology,
        derive_runtime_requirements,
    )

    with pytest.raises(StorageRequirementsNotMetError, match="storage"):
        derive_runtime_requirements(
            settings=RuntimeSettings(topology=RuntimeTopology.MULTI_WORKER),
            dependencies=RuntimeDependencies(run_commit_coordinator=object()),
        )


def test_derive_multi_worker_refuses_dependencies_missing_coordinator(tmp_path):
    # The MULTI_WORKER real-object rule: a dependencies bundle without an
    # injected RunCommitCoordinator cannot serve a multi-worker topology (the
    # build kernel no longer selects one).
    from linktools.ai.runtime.builder import RuntimeSettings
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.errors import StorageRequirementsNotMetError
    from linktools.ai.run.requirements import (
        RuntimeTopology,
        derive_runtime_requirements,
    )

    from external_adapter import build_in_memory_external_storage

    storage = build_in_memory_external_storage(root=tmp_path / "derive_no_coord")
    with pytest.raises(StorageRequirementsNotMetError, match="RunCommitCoordinator"):
        derive_runtime_requirements(
            settings=RuntimeSettings(topology=RuntimeTopology.MULTI_WORKER),
            dependencies=RuntimeDependencies(storage=storage),
        )


def test_derive_multi_worker_refuses_storage_missing_jobstore():
    # The MULTI_WORKER real-object rule: workers share JobStore rows, so a
    # Storage without a JobStore cannot serve the topology -- derive refuses it
    # rather than letting workers race on rows that have nowhere to live.
    from types import SimpleNamespace

    from linktools.ai.runtime.builder import RuntimeSettings
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.errors import StorageRequirementsNotMetError
    from linktools.ai.run.requirements import (
        RuntimeTopology,
        derive_runtime_requirements,
    )

    storage_without_jobs = SimpleNamespace(jobs=None)
    with pytest.raises(StorageRequirementsNotMetError, match="JobStore"):
        derive_runtime_requirements(
            settings=RuntimeSettings(topology=RuntimeTopology.MULTI_WORKER),
            dependencies=RuntimeDependencies(
                storage=storage_without_jobs,
                run_commit_coordinator=object(),
            ),
        )


def test_feature_consistency_rejects_declared_transactional_component_without_store():
    # : a StorageFeatures that declares a component transactional must have
    # a real wired store for it. Declaring ASSETS transactional with no AssetStore
    # is a false capability claim -- the consistency gate fails fast at build.
    from types import SimpleNamespace

    from linktools.ai.run.requirements import enforce_storage_feature_consistency
    from linktools.ai.storage.features import StorageComponent

    bogus = SimpleNamespace(
        features=_features(transactional_components=frozenset({StorageComponent.ASSETS})),
        _transaction_manager=object(),
        coordination=object(),
        artifacts=object(),
        assets=None,  # declared transactional, but no wired store
        jobs=object(),
    )
    with pytest.raises(StorageRequirementsNotMetError, match="not wired"):
        enforce_storage_feature_consistency(bogus)


def test_runtime_dependencies_carries_composition_root_fields():
    # B-C5: RuntimeDependencies carries the composition-root objects the gate
    # reads (storage + run_commit_coordinator). A spec-providers-only bundle
    # leaves them None (the common case); the build kernel populates them on
    # the effective dependencies it hands derive_runtime_requirements.
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import (
        FilesystemRunCommitCoordinator,
    )

    empty = RuntimeDependencies()
    assert empty.storage is None
    assert empty.run_commit_coordinator is None

    storage = FilesystemStorage(root="/tmp/_deps_field_check")
    coord = FilesystemRunCommitCoordinator.from_storage(storage)
    populated = RuntimeDependencies(storage=storage, run_commit_coordinator=coord)
    assert populated.storage is storage
    assert populated.run_commit_coordinator is coord


def test_multi_worker_jobs_rejects_process_local_storage(tmp_path):
    # external-adapter step 5: remove the distributed capability and a
    # multi-worker Job topology must FAIL TO CONSTRUCT against process-local
    # storage -- no silent fallback to the in-process coordinator.
    from linktools.ai.errors import StorageRequirementsNotMetError as _Err
    from linktools.ai.runtime import Runtime, build_runtime
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    rt_storage = FilesystemStorage(root=tmp_path)  # process-local coordination
    with pytest.raises(_Err, match="coordination"):
        build_runtime(
            storage=rt_storage,
            requirements=RuntimeRequirements.for_multi_worker_jobs(),
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(rt_storage),
        )


def test_feature_consistency_rejects_streaming_artifacts_without_artifact_store():
    # : declaring streaming_artifacts=True on StorageFeatures but wiring no
    # ArtifactStore must fail fast at the consistency gate, not AttributeError
    # at first use.
    from types import SimpleNamespace

    from linktools.ai.run.requirements import enforce_storage_feature_consistency

    bogus = SimpleNamespace(
        features=_features(streaming_artifacts=True),
        _transaction_manager=object(),  # present
        coordination=object(),  # present
        artifacts=None,  # MISSING despite streaming_artifacts=True
        assets=None,
        jobs=None,
    )
    with pytest.raises(StorageRequirementsNotMetError, match="ArtifactStore"):
        enforce_storage_feature_consistency(bogus)


def test_feature_consistency_rejects_artifact_store_without_streaming_flag():
    # The reverse direction: an ArtifactStore is wired (it can stream) but the
    # flag is False. The flag must agree with the wired store (the consistency requirement).
    from types import SimpleNamespace

    from linktools.ai.run.requirements import enforce_storage_feature_consistency

    bogus = SimpleNamespace(
        features=_features(streaming_artifacts=False),
        _transaction_manager=object(),
        coordination=object(),
        artifacts=object(),  # WIRED despite streaming_artifacts=False
        assets=None,
        jobs=None,
    )
    with pytest.raises(StorageRequirementsNotMetError, match="streaming_artifacts=False"):
        enforce_storage_feature_consistency(bogus)


def test_feature_consistency_rejects_distributed_coordination_without_coordinator():
    # : declaring coordination=DISTRIBUTED but wiring no coordinator fails.
    from types import SimpleNamespace

    from linktools.ai.run.requirements import enforce_storage_feature_consistency

    bogus = SimpleNamespace(
        features=_features(coordination_scope=CoordinationScope.DISTRIBUTED),
        _transaction_manager=object(),
        coordination=None,  # MISSING despite DISTRIBUTED
        artifacts=object(),
        assets=object(),
        jobs=object(),
    )
    with pytest.raises(StorageRequirementsNotMetError, match="LeaseCoordinator"):
        enforce_storage_feature_consistency(bogus)


def test_feature_consistency_accepts_an_honest_inrepo_storage(tmp_path):
    # The in-repo FilesystemStorage wires every object its features declare, so
    # the consistency gate admits it.
    from linktools.ai.run.requirements import enforce_storage_feature_consistency
    from linktools.ai.storage.facade import FilesystemStorage

    enforce_storage_feature_consistency(FilesystemStorage(root=tmp_path))
