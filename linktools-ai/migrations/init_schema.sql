-- linktools-ai persistence v1. Provisioning is intentionally explicit.
-- This file is manually reviewed DBA input, not an application bootstrap.

CREATE TABLE ai_state_records (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 store scope for one namespace, tenant, and Runtime domain.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the persisted runtime record.',
    scope_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 grouping key used by the record kind''s primary list query.',
    subject_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 subject identity used for exact lookups within a record scope.',
    parent_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 identity of the logical parent used by hierarchical list queries.',
    kind VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Stable persisted record kind such as session, execution, task_node, or agent_run.',
    sort_key LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
    storage_revision BIGINT NOT NULL COMMENT 'Physical compare-and-swap revision incremented by every record mutation.',
    lease_owner TEXT NULL COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
    lease_fence BIGINT NOT NULL DEFAULT 0 COMMENT 'Monotonic fencing token for the current durable record lease.',
    lease_expires_at TIMESTAMP(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
    payload_json JSON NOT NULL COMMENT 'Canonical domain JSON encoded as a JSON string preserving numeric representations.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_store_digest_key_digest (store_digest, key_digest),
    KEY ix_store_digest_kind_sort_key (store_digest, kind, sort_key(128)),
    KEY ix_scope_digest_sort_key (scope_digest, sort_key(128)),
    KEY ix_scope_digest_subject_digest (scope_digest, subject_digest),
    KEY ix_parent_digest_sort_key (parent_digest, sort_key(128)),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Current durable state for runtime and step resources persisted as versioned records.';

CREATE TABLE ai_state_aliases (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 store scope for one namespace, tenant, and Runtime domain.',
    alias_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of a secondary unique runtime lookup key.',
    record_key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the runtime record resolved by this alias.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_store_digest_alias_digest (store_digest, alias_digest),
    KEY ix_record_key_digest (record_key_digest),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Secondary unique lookup identities that resolve to canonical runtime records.';

CREATE TABLE ai_state_facts (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 store scope for one namespace, tenant, and Runtime domain.',
    stream_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the append-only fact stream.',
    sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the fact within its stream.',
    owner_key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the runtime record that owns this fact.',
    kind VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Fact kind such as execution_event, step_event, step_checkpoint, or step_effect.',
    subject_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical SHA-256 grouping identity for multiple facts of one logical subject such as a tool call.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Queryable fact state such as checkpoint completeness or tool-effect lifecycle state.',
    payload_json JSON NOT NULL COMMENT 'Canonical immutable fact JSON encoded as a JSON string preserving numeric representations.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_store_digest_stream_digest_sequence (store_digest, stream_digest, sequence),
    KEY ix_owner_key_digest (owner_key_digest), KEY ix_stream_digest_subject_digest_sequence (stream_digest, subject_digest, sequence),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable ordered runtime facts including events, checkpoints, and effects.';

CREATE TABLE ai_state_sequences (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 store scope for one namespace, tenant, and Runtime domain.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the monotonic sequence counter.',
    value BIGINT NOT NULL COMMENT 'Last committed value allocated by the sequence.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_store_digest_key_digest (store_digest, key_digest),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Durable monotonic counters used to allocate ordered runtime sequence numbers.';

CREATE TABLE ai_state_operations (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 store scope for one namespace, tenant, and Runtime domain.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the durable operation.',
    stream_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the ordered operation stream.',
    sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the operation within its stream.',
    state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Current durable operation state used by replay and completion logic.',
    compactable BOOLEAN NOT NULL COMMENT 'Whether a terminal operation may be removed by ledger compaction.',
    payload_json JSON NOT NULL COMMENT 'Canonical operation JSON encoded as a JSON string preserving numeric representations.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_store_digest_key_digest (store_digest, key_digest), UNIQUE KEY uk_store_digest_stream_digest_sequence (store_digest, stream_digest, sequence),
    KEY ix_stream_digest_state_sequence (stream_digest, state, sequence),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
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

CREATE TABLE ai_asset_batch_receipts (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the AssetStore namespace.',
    idempotency_key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 digest of the caller batch idempotency key.',
    payload_json JSON NOT NULL COMMENT 'Versioned committed Asset batch receipt without input file bytes.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id), UNIQUE KEY uk_namespace_digest_idempotency_key_digest (namespace_digest, idempotency_key_digest),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Durable idempotency receipts for committed atomic AssetStore batches.';

CREATE TABLE ai_objects (
    id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate row identifier used only by the SQL backend.',
    key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical SHA-256 identity of the ObjectStore store identifier and object key.',
    store_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical ObjectStore identifier that namespaces object keys.',
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
    namespace VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original Metrics namespace retained after pruning.',
    observation_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original observation identity retained after pruning.',
    namespace_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 partition identity of the Metrics namespace.',
    observation_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 identity of namespace plus observation_id.',
    payload_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the canonical ObservationEnvelope payload.',
    kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical observation kind query projection.',
    occurred_at DATETIME(6) NOT NULL COMMENT 'Canonical UTC observation occurrence time.',
    payload_json JSON NULL COMMENT 'Versioned canonical immutable ObservationEnvelope payload.',
    active_occurred_at DATETIME(6) NULL COMMENT 'Derived occurrence time for active observations only.',
    pruned_at DATETIME(6) NULL COMMENT 'Backend time of the first payload pruning transition.',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    PRIMARY KEY (id),
    UNIQUE KEY uk_observation_digest (observation_digest),
    KEY ix_namespace_digest_kind_active_occurred_at (namespace_digest, kind, active_occurred_at),
    KEY ix_namespace_digest_active_occurred_at (namespace_digest, active_occurred_at),
    KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Immutable metric observations and retained pruning receipts.';

-- Official StepStore physical mapping.

CREATE TABLE ai_step_events (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical namespace and tenant partition.',
	run_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical lookup digest of the original run identifier.',
	run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	kind VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream boundary event kind.',
	outcome VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Owned lifecycle outcome, absent for ordinary upstream events.',
	step_index BIGINT NOT NULL COMMENT 'Upstream graph step index.',
	timestamp VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 event time.',
	conversation_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream conversation identifier.',
	parent_run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream parent run identifier.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream agent name.',
	tool_call_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream tool call identifier.',
	tool_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream tool name.',
	error LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream boundary error text.',
	metadata JSON COMMENT 'Upstream string metadata encoded as JSON, unless referenced from the native run.',
	run_fields SMALLINT UNSIGNED NOT NULL COMMENT 'Native run field references: conversation 1, parent 2, agent 4, metadata 8.' DEFAULT 0,
	body_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Integrity digest of the managed lifecycle source body.',
	fields_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Integrity digest of managed lifecycle fields excluding metadata.',
	idempotency_key LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream boundary replay key.',
	key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 identity of the scoped run and replay key or append occurrence.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_store_digest_run_digest (store_digest, run_digest),
	UNIQUE KEY uk_key_digest (key_digest),
	KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Append-only upstream agent execution boundary events.' COLLATE utf8mb4_bin;

CREATE TABLE ai_step_messages (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical namespace and tenant partition.',
	sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the exact stored upstream message JSON.',
	message LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'One upstream ModelMessage JSON body.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	UNIQUE KEY uk_store_digest_sha256 (store_digest, sha256),
	KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Deduplicated upstream ModelMessage JSON bodies.' COLLATE utf8mb4_bin;

CREATE TABLE ai_step_runs (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical namespace and tenant partition.',
	run_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical lookup digest of the original run identifier.',
	run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	conversation_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream conversation identifier.',
	conversation_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Physical conversation lookup digest.',
	parent_run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream parent run identifier.',
	parent_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Physical parent run lookup digest.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream agent name.',
	metadata LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream string metadata encoded as JSON.',
	started_at VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 run start time.',
	registration_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream registration invocation identifier.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_store_digest_conversation_digest (store_digest, conversation_digest),
	KEY ix_store_digest_parent_digest (store_digest, parent_digest),
	UNIQUE KEY uk_store_digest_run_digest (store_digest, run_digest),
	KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Upstream agent run lineage and registration records.' COLLATE utf8mb4_bin;

CREATE TABLE ai_step_sequences (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical namespace and tenant partition.',
	sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the immutable sequence node.',
	message_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the referenced upstream message JSON.',
	tree_count BIGINT NOT NULL COMMENT 'Number of ordered message occurrences in this tree.',
	left_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Immutable left subtree root.',
	right_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Immutable right subtree root.',
	next_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Next immutable tree in the sequence forest.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	UNIQUE KEY uk_store_digest_sha256 (store_digest, sha256),
	KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Shared immutable message sequence nodes for native snapshots.' COLLATE utf8mb4_bin;

CREATE TABLE ai_step_snapshots (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical namespace and tenant partition.',
	run_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Physical lookup digest of the original run identifier.',
	run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	step_index BIGINT NOT NULL COMMENT 'Upstream graph step index.',
	conversation_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream conversation identifier.',
	parent_run_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Original upstream parent run identifier.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream agent name.',
	timestamp VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 snapshot time.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream complete or interrupted state.',
	messages VARCHAR(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Bounded message descriptor and native checkpoint state without message bodies.',
	sequence_root CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Immutable message sequence root retained by this snapshot.',
	idempotency_key LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Upstream boundary replay key.',
	key_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 identity of the scoped run and replay key or append occurrence.',
	pruned BOOL NOT NULL COMMENT 'Retired snapshot identity without retained message references.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_store_digest_run_digest (store_digest, run_digest),
	UNIQUE KEY uk_key_digest (key_digest),
	KEY ix_updated_at (updated_at), KEY ix_created_at (created_at)
)ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Upstream complete or interrupted message-history snapshots.' COLLATE utf8mb4_bin;
