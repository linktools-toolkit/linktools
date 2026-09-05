-- linktools-ai persistence v1. Provisioning is intentionally explicit.
-- This file is manually reviewed DBA input, not an application bootstrap.

CREATE TABLE ai_state_records (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the persisted runtime record.',
    partition_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 partition for records of the same tenant, runtime domain, and record kind.',
    scope_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 grouping key used by the record kind''s primary list query.',
    parent_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 identity of the logical parent used by hierarchical list queries.',
    kind VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Stable persisted record kind such as session, execution, task_node, or step_run.',
    sort_key VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
    storage_version BIGINT NOT NULL COMMENT 'Internal optimistic-concurrency version incremented by every physical record mutation.',
    lease_owner TEXT NULL COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
    lease_fence BIGINT NOT NULL DEFAULT 0 COMMENT 'Monotonic fencing token for the current durable record lease.',
    lease_expires_at TIMESTAMP(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical domain payload not needed by SQL identity, query, CAS, or lease predicates.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest (key_digest),
    KEY ix_partition_digest_sort_key (partition_digest, sort_key),
    KEY ix_scope_digest_sort_key (scope_digest, sort_key),
    KEY ix_scope_digest_state_sort_key (scope_digest, state, sort_key),
    KEY ix_parent_digest_sort_key (parent_digest, sort_key),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Current durable state for runtime and step resources persisted as versioned records.';

CREATE TABLE ai_state_aliases (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    alias_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of a secondary unique runtime lookup key.',
    record_key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the runtime record resolved by this alias.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_alias_digest (alias_digest),
    KEY ix_record_key_digest (record_key_digest), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Secondary unique lookup identities that resolve to canonical runtime records.';

CREATE TABLE ai_state_facts (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    stream_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the append-only fact stream.',
    sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the fact within its stream.',
    owner_key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the runtime record that owns this fact.',
    kind VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Fact kind such as execution_event, step_event, step_snapshot, or step_effect.',
    subject_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 grouping identity for multiple facts of one logical subject such as a tool call.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Queryable fact state such as snapshot completeness or tool-effect lifecycle state.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical immutable fact payload.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_stream_digest_sequence (stream_digest, sequence),
    KEY ix_owner_key_digest (owner_key_digest), KEY ix_stream_digest_subject_digest_sequence (stream_digest, subject_digest, sequence),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable ordered runtime facts including events, snapshots, and effects.';

CREATE TABLE ai_state_sequences (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the monotonic sequence counter.',
    value BIGINT NOT NULL COMMENT 'Last committed value allocated by the sequence.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest (key_digest), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Durable monotonic counters used to allocate ordered runtime sequence numbers.';

CREATE TABLE ai_state_operations (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the durable operation.',
    stream_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the ordered operation stream.',
    sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the operation within its stream.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Current durable operation state used by replay and completion logic.',
    compactable BOOLEAN NOT NULL COMMENT 'Whether a terminal operation may be removed by ledger compaction.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical operation request, result, error, resource identity, and semantic timestamps.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest (key_digest), UNIQUE KEY uk_stream_digest_sequence (stream_digest, sequence),
    KEY ix_stream_digest_state_sequence (stream_digest, state, sequence), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Ordered durable operation ledger for replay, result recovery, status transition, and compaction.';

CREATE TABLE ai_asset_heads (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the AssetStore namespace.',
    store_revision BIGINT NOT NULL COMMENT 'Last committed namespace-wide AssetStore revision.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_namespace_digest (namespace_digest), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Namespace-level optimistic revision head that serializes AssetStore mutations.';

CREATE TABLE ai_asset_entries (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the asset key including namespace identity.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the AssetStore namespace.',
    entry_revision BIGINT NOT NULL COMMENT 'Latest committed per-asset revision.',
    store_revision BIGINT NOT NULL COMMENT 'Namespace-wide revision that produced this current projection.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical current AssetInfo payload including status, metadata, content reference, and semantic timestamps.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest (key_digest), KEY ix_namespace_digest_store_revision (namespace_digest, store_revision),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Current AssetStore projection for the latest committed revision of each asset key.';

CREATE TABLE ai_asset_changes (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the asset key including namespace identity.',
    entry_revision BIGINT NOT NULL COMMENT 'Immutable per-asset revision number represented by this history row.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the AssetStore namespace.',
    store_revision BIGINT NOT NULL COMMENT 'Namespace-wide revision in which this asset change committed.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical immutable AssetInfo history payload.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest_entry_revision (key_digest, entry_revision),
    UNIQUE KEY uk_namespace_digest_store_revision_key_digest (namespace_digest, store_revision, key_digest),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable AssetStore history containing every committed per-asset revision.';

CREATE TABLE ai_objects (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the ObjectStore store identifier and object key.',
    store_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical ObjectStore identifier that namespaces object keys.',
    object_key TEXT NOT NULL COMMENT 'Original opaque object key exposed by the ObjectStore API.',
    content_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 digest of the immutable object bytes.',
    size BIGINT NOT NULL COMMENT 'Exact immutable object size in bytes.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest (key_digest), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable ObjectStore headers containing canonical object identity, content digest, and size.';

CREATE TABLE ai_object_chunks (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the immutable ObjectStore object owning this chunk.',
    chunk_index BIGINT NOT NULL COMMENT 'Zero-based ordered chunk position within the immutable object.',
    content LONGBLOB NOT NULL COMMENT 'Binary content bytes for this immutable object chunk.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_key_digest_chunk_index (key_digest, chunk_index), KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Ordered binary chunks that compose immutable ObjectStore content.';

CREATE TABLE ai_metric_definitions (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 partition identity of the Metrics namespace.',
    metric_name VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Versioned metric definition name.',
    revision BIGINT NOT NULL COMMENT 'Metric semantic revision.',
    definition_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the normalized semantic metric definition.',
    observation_kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical observation kind consumed by the metric.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical MetricDefinitionEnvelope payload.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id),
    UNIQUE KEY uk_namespace_digest_metric_name_revision (namespace_digest, metric_name, revision),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Custom metric definitions.';

CREATE TABLE ai_metric_observations (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 partition identity of the Metrics namespace.',
    observation_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 identity of namespace plus observation_id.',
    payload_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the canonical ObservationEnvelope payload.',
    kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical observation kind query projection.',
    occurred_at DATETIME(6) NOT NULL COMMENT 'Canonical UTC observation occurrence time.',
    payload_json JSON NOT NULL COMMENT 'Versioned canonical immutable ObservationEnvelope payload.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id),
    UNIQUE KEY uk_observation_digest (observation_digest),
    KEY ix_namespace_digest_kind_occurred_at (namespace_digest, kind, occurred_at),
    KEY ix_namespace_digest_occurred_at (namespace_digest, occurred_at),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable metric observations.';
