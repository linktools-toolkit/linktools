-- linktools-ai persistence v1. Provisioning is intentionally explicit.
-- This file is manually reviewed DBA input, not an application bootstrap.

-- Runtime control, facts, counters and operation receipts

CREATE TABLE ai_state_records (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	`key` VARBINARY(768) NOT NULL COMMENT 'Exact bounded binary identity of the persisted runtime record.',
	scope_key VARBINARY(768) NULL COMMENT 'Bounded binary grouping identity used by the record kind''s primary list query.',
	subject_id VARBINARY(128) NULL COMMENT 'Exact UTF-8 subject identifier used for lookups within a record scope.',
	parent_key VARBINARY(768) NULL COMMENT 'Bounded binary parent identity used by hierarchical list queries.',
	kind VARCHAR(32) NOT NULL COMMENT 'Stable persisted record kind such as session, execution, task_node, or agent_run.',
	sort_key LONGBLOB NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
	storage_revision BIGINT NOT NULL COMMENT 'Physical compare-and-swap revision incremented by every record mutation.',
	lease_owner VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
	lease_fence BIGINT NOT NULL COMMENT 'Monotonic fencing token for the current durable record lease.' DEFAULT 0,
	lease_expires_at DATETIME(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
	payload_json JSON NOT NULL COMMENT 'Canonical domain JSON object.',
	payload_bytes LONGBLOB NULL COMMENT 'Optional binary content body owned by this record; ordinary domain state remains JSON.',
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete key; raw identity must also match.',
	scope_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Single SHA-256 lookup projection of the complete scope_key; raw identity must also match.',
	parent_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Single SHA-256 lookup projection of the complete parent_key; raw identity must also match.',
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 bytes of the full sort key; non-unique query projection only.',
	scan_kind VARCHAR(34) NOT NULL COMMENT 'Write-through kind and execution recovery candidate projection; never domain authority.',
	strict_turn_execution_id VARBINARY(128) NULL COMMENT 'Unpublished strict conversation turn owner, distinct from business session state.',
	strict_turn_fence BIGINT NULL COMMENT 'Fencing generation for a same-group strict conversation; NULL has no such ownership.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_parent_locator_sort_prefix (parent_locator, sort_prefix),
	KEY ix_scope_locator_sort_prefix (scope_locator, sort_prefix),
	KEY ix_scope_locator_subject_id (scope_locator, subject_id),
	KEY ix_store_key_scan_kind_sort_prefix (store_key, scan_kind, sort_prefix),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_key_locator (store_key, key_locator)
)ENGINE=InnoDB COMMENT='Current durable state for runtime and step resources persisted as versioned records.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_execution_records (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	`key` VARBINARY(488) NOT NULL COMMENT 'Exact bounded binary identity of the persisted runtime record.',
	scope_key VARBINARY(428) NULL COMMENT 'Bounded binary grouping identity used by the record kind''s primary list query.',
	subject_id VARBINARY(128) NULL COMMENT 'Exact UTF-8 subject identifier used for lookups within a record scope.',
	parent_key VARBINARY(367) NULL COMMENT 'Bounded binary parent identity used by hierarchical list queries.',
	kind VARCHAR(32) NOT NULL COMMENT 'Stable persisted record kind such as session, execution, task_node, or agent_run.',
	sort_key VARBINARY(776) NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
	storage_revision BIGINT NOT NULL COMMENT 'Physical compare-and-swap revision incremented by every record mutation.',
	lease_owner VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
	lease_fence BIGINT NOT NULL COMMENT 'Monotonic fencing token for the current durable record lease.' DEFAULT 0,
	lease_expires_at DATETIME(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
	payload_json JSON NOT NULL COMMENT 'Domain JSON object; execution reference digests are stored in typed columns.',
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete key; raw identity must also match.',
	scope_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Single SHA-256 lookup projection of the complete scope_key; raw identity must also match.',
	parent_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Single SHA-256 lookup projection of the complete parent_key; raw identity must also match.',
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 bytes of the full sort key; non-unique query projection only.',
	policy_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	binding_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	input_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	result_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	output_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	input_first_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	output_first_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Canonical execution reference SHA-256.',
	scan_kind VARCHAR(34) NOT NULL COMMENT 'Write-through kind and execution recovery candidate projection; never domain authority.',
	turn_slot_kind SMALLINT COMMENT 'Runtime slot family of the exact source owner.',
	turn_slot_key VARBINARY(64) COMMENT 'Raw runtime slot identity, separate from its family.',
	turn_token VARBINARY(32) COMMENT 'Exact generated runtime lease token.',
	turn_fence BIGINT COMMENT 'Session fencing generation of the coordinated execution.',
	turn_expiry DATETIME(6) COMMENT 'Last granted source deadline; active only with a lease token.',
	turn_revision BIGINT NOT NULL COMMENT 'Independent writable revision for source lease renewal.' DEFAULT 0,
	turn_request_route VARBINARY(64) NOT NULL COMMENT 'Unique live request, parked standalone request, or row-local unoccupied address.',
	turn_occupied_route VARBINARY(64) NOT NULL COMMENT 'Unique active or parked split-store session or row-local unoccupied address.',
	turn_session_owner_id BIGINT COMMENT 'Exact same-store strict SessionRecord row; never a portable address.',
	turn_receipt JSON COMMENT 'Compact retained coordination receipt, independent of the execution payload.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_parent_locator_sort_prefix (parent_locator, sort_prefix),
	KEY ix_scope_locator_sort_prefix (scope_locator, sort_prefix),
	KEY ix_scope_locator_subject_id (scope_locator, subject_id),
	KEY ix_store_key_scan_kind_sort_prefix (store_key, scan_kind, sort_prefix),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_key_locator (store_key, key_locator),
	UNIQUE KEY uk_store_key_turn_request_route (store_key, turn_request_route),
	UNIQUE KEY uk_store_key_turn_occupied_route (store_key, turn_occupied_route),
	KEY ix_turn_session_owner_id_turn_fence (turn_session_owner_id, turn_fence)
)ENGINE=InnoDB COMMENT='Authoritative execution lifecycle, request receipts, and content references.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_task_graphs (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	`key` VARBINARY(352) NOT NULL COMMENT 'Exact bounded binary identity of the persisted runtime record.',
	subject_id VARBINARY(128) NULL COMMENT 'Exact UTF-8 subject identifier used for lookups within a record scope.',
	sort_key LONGBLOB NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
	storage_revision BIGINT NOT NULL COMMENT 'Physical compare-and-swap revision incremented by every record mutation.',
	lease_owner VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
	lease_fence BIGINT NOT NULL COMMENT 'Monotonic fencing token for the current durable record lease.' DEFAULT 0,
	lease_expires_at DATETIME(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
	payload_json JSON NOT NULL COMMENT 'Canonical domain JSON object.',
	payload_bytes LONGBLOB NULL COMMENT 'Optional binary content body owned by this record; ordinary domain state remains JSON.',
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete key; raw identity must also match.',
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 bytes of the full sort key; non-unique query projection only.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_store_key_state_sort_prefix (store_key, state, sort_prefix),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_key_locator (store_key, key_locator)
)ENGINE=InnoDB COMMENT='Authoritative TaskGraph state and recovery ordering.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_state_facts (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	stream_key VARBINARY(768) NOT NULL COMMENT 'Exact bounded binary identity of the append-only fact stream.',
	sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the fact within its stream.',
	owner_key VARBINARY(768) NOT NULL COMMENT 'Exact bounded binary identity of the runtime record that owns this fact.',
	kind VARCHAR(32) NOT NULL COMMENT 'Fact kind such as execution_event, step_event, step_checkpoint, or step_effect.',
	subject_id VARBINARY(128) NULL COMMENT 'Exact UTF-8 subject identifier grouping facts such as one tool call.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Queryable fact state such as checkpoint completeness or tool-effect lifecycle state.',
	payload_json JSON NOT NULL COMMENT 'Canonical immutable fact JSON object.',
	stream_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete stream_key; raw identity must also match.',
	owner_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete owner_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_owner_locator (owner_locator),
	KEY ix_stream_locator_subject_id_sequence (stream_locator, subject_id, sequence),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_stream_locator_sequence (store_key, stream_locator, sequence)
)ENGINE=InnoDB COMMENT='Immutable ordered runtime facts including events, checkpoints, and effects.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_state_counters (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	stream_key VARBINARY(768) NOT NULL COMMENT 'Exact bounded binary identity of the monotonic sequence counter.',
	value BIGINT NOT NULL COMMENT 'Last committed value allocated by the sequence.',
	stream_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete stream_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_stream_locator (store_key, stream_locator)
)ENGINE=InnoDB COMMENT='Durable monotonic counters used to allocate ordered runtime sequence numbers.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_state_operations (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	operation_key VARBINARY(768) NOT NULL COMMENT 'Binary operation address, at most 768 bytes.',
	stream_key VARBINARY(768) NOT NULL COMMENT 'Exact bounded binary identity of the ordered operation stream.',
	sequence BIGINT NOT NULL COMMENT 'Strictly increasing position of the operation within its stream.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Current durable operation state used by replay and completion logic.',
	compactable BOOL NOT NULL COMMENT 'Whether a terminal operation may be removed by ledger compaction.',
	payload_json JSON NOT NULL COMMENT 'Canonical operation JSON object.',
	operation_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete operation_key; raw identity must also match.',
	stream_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete stream_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_stream_locator_state_sequence (stream_locator, state, sequence),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_operation_locator (store_key, operation_locator),
	UNIQUE KEY uk_store_key_stream_locator_sequence (store_key, stream_locator, sequence)
)ENGINE=InnoDB COMMENT='Ordered durable operation ledger for replay, result recovery, status transition, and compaction.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

-- Step runs, tool effects and immutable message content

CREATE TABLE ai_step_runs (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(274) NOT NULL COMMENT 'Exact bounded binary native namespace and tenant partition.',
	run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	conversation_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream conversation identifier.',
	parent_run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream parent run identifier.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream agent name.',
	metadata LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream string metadata encoded as JSON.',
	started_at VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 run start time.',
	registration_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream registration invocation identifier.',
	turn_source_generation BIGINT COMMENT 'Adopted Conversation generation; absent for independent Step runs.',
	turn_source_phase VARCHAR(12) COMMENT 'Subordinate split handoff phase; never an independent lease.',
	turn_source_receipt VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Exact source receipt identity matched by Conversation reconciliation.',
	turn_source_pin JSON COMMENT 'Retained immutable source checkpoint and transcript references.',
	managed_start_key VARBINARY(128) COMMENT 'Exact first managed start identity; NULL means metadata admission only.',
	store_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete store_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_store_locator_conversation_id (store_locator, conversation_id),
	KEY ix_store_locator_parent_run_id (store_locator, parent_run_id),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_locator_run_id (store_locator, run_id)
)ENGINE=InnoDB COMMENT='Upstream agent run lineage and registration records.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_step_run_admissions (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact namespace, tenant and domain partition admitting this run.',
	`key` VARBINARY(359) NOT NULL COMMENT 'Canonical local run admission identity, independent of the registration owner.',
	scope_key VARBINARY(436) COMMENT 'Conversation lookup identity derived from the validated domain projection.',
	parent_key VARBINARY(367) COMMENT 'Parent-run lookup identity derived from the validated domain projection.',
	sort_key LONGBLOB NOT NULL COMMENT 'Stable ordering by projected start time and complete admission identity.',
	storage_revision BIGINT NOT NULL COMMENT 'Optimistic local publication and release fence, independent of canonical registration.',
	payload_json JSON NOT NULL COMMENT 'Typed canonical registration reference and actual domain projection differences; no copied run body.',
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 lookup projection of the admission key; complete identity must also match.',
	scope_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'SHA-256 lookup projection of the projected conversation scope.',
	parent_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'SHA-256 lookup projection of the projected parent-run identity.',
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 sort-key bytes used only for ordered lookup.',
	event_high_water BIGINT COMMENT 'Last allocated local event sequence; NULL means no event counter.',
	checkpoint_high_water BIGINT COMMENT 'Last allocated local checkpoint sequence; NULL means no checkpoint counter.',
	interaction_high_water BIGINT COMMENT 'Last allocated local interaction sequence; NULL means no interaction counter.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_key_locator (store_key, key_locator)
) ENGINE=InnoDB CHARSET=utf8mb4 COMMENT='Step-owned domain run admission; canonical run metadata remains with its unique registration owner.' ROW_FORMAT=DYNAMIC  COLLATE utf8mb4_bin;

CREATE TABLE ai_step_tool_operations (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact bounded binary identity of the physical State store partition.',
	`key` VARBINARY(362) NOT NULL COMMENT 'Exact bounded binary identity of the persisted runtime record.',
	scope_key VARBINARY(436) COMMENT 'Bounded binary grouping identity used by the record kind''s primary list query.',
	subject_id VARBINARY(128) COMMENT 'Exact UTF-8 subject identifier used for lookups within a record scope.',
	parent_key VARBINARY(437) COMMENT 'Bounded binary parent identity used by hierarchical list queries.',
	sort_key LONGBLOB NOT NULL COMMENT 'Stable canonical ordering token used for deterministic keyset pagination.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Current SQL-queryable state for record kinds with persisted state-machine semantics.',
	storage_revision BIGINT NOT NULL COMMENT 'Physical compare-and-swap revision incremented by every record mutation.',
	lease_owner VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Current durable lease owner for record kinds using claim and fencing semantics.',
	lease_fence BIGINT NOT NULL COMMENT 'Monotonic fencing token for the current durable record lease.' DEFAULT 0,
	lease_expires_at DATETIME(6) NULL COMMENT 'Database-authoritative expiry time of the current durable lease.',
	payload_json JSON NOT NULL COMMENT 'Canonical domain JSON object.',
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete key; raw identity must also match.',
	scope_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Single SHA-256 lookup projection of the complete scope_key; raw identity must also match.',
	parent_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Single SHA-256 lookup projection of the complete parent_key; raw identity must also match.',
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 bytes of the full sort key; non-unique query projection only.',
	tool_arguments_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Derived immutable content lookup SHA-256.',
	tool_result_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Derived immutable content lookup SHA-256.',
	tool_error_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Derived immutable content lookup SHA-256.',
	tool_arguments_first_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Derived immutable content lookup SHA-256.',
	tool_result_first_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Derived immutable content lookup SHA-256.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_parent_locator_sort_prefix (parent_locator, sort_prefix),
	KEY ix_scope_locator_subject_id (scope_locator, subject_id),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_key_locator (store_key, key_locator)
)ENGINE=InnoDB COMMENT='Authoritative Step tool effect ledger, including dispatch fencing and durable receipts.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_messages (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(274) NOT NULL COMMENT 'Exact bounded binary native namespace and tenant partition.',
	sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the canonical uncompressed bytes for the partition content type.',
	message LONGBLOB NULL COMMENT 'Raw or compressed canonical content bytes, absent for native attachment manifests.',
	compression VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Message body compression codec: raw or zlib.',
	raw_size BIGINT NOT NULL COMMENT 'Canonical uncompressed content byte count for bounded decompression.',
	payload_json JSON NULL COMMENT 'Typed attachment manifest without attachment bytes, absent for ordinary message bodies.',
	gc_revision BIGINT NOT NULL COMMENT 'Optimistic content revision used by the owning partition.' DEFAULT 0,
	store_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete store_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_locator_sha256 (store_locator, sha256)
)ENGINE=InnoDB COMMENT='Immutable typed content bytes: native messages and accepted execution payloads in disjoint partitions.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_message_sequences (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(274) NOT NULL COMMENT 'Exact bounded binary native namespace and tenant partition.',
	sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the immutable sequence node.',
	message_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the referenced upstream message JSON.',
	tree_count BIGINT NOT NULL COMMENT 'Number of ordered message occurrences in this tree.',
	left_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Immutable left subtree root.',
	right_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Immutable right subtree root.',
	next_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Next immutable tree in the sequence forest.',
	store_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete store_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_locator_sha256 (store_locator, sha256)
)ENGINE=InnoDB COMMENT='Shared immutable message sequence nodes for native snapshots.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_step_snapshots (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(274) NOT NULL COMMENT 'Exact bounded binary native namespace and tenant partition.',
	run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	step_index BIGINT NOT NULL COMMENT 'Upstream graph step index.',
	conversation_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream conversation identifier.',
	parent_run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream parent run identifier.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream agent name.',
	timestamp VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 snapshot time.',
	state VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream complete or interrupted state.',
	messages VARCHAR(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Bounded message descriptor and native checkpoint state without message bodies.',
	sequence_root CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Immutable message sequence root retained by this snapshot.',
	idempotency_key VARBINARY(128) NOT NULL COMMENT 'Exact UTF-8 replay key or a reserved unique occurrence key for an unkeyed append.',
	pruned BOOL NOT NULL COMMENT 'Retired snapshot identity without retained message references.',
	store_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete store_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_store_locator_run_id_pruned (store_locator, run_id, pruned),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_locator_run_id_idempotency_key (store_locator, run_id, idempotency_key)
)ENGINE=InnoDB COMMENT='Upstream complete or interrupted message-history snapshots.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_step_events (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(274) NOT NULL COMMENT 'Exact bounded binary native namespace and tenant partition.',
	run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original upstream run identifier.',
	kind VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Upstream boundary event kind.',
	outcome VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Owned lifecycle outcome, absent for ordinary upstream events.',
	step_index BIGINT NOT NULL COMMENT 'Upstream graph step index.',
	timestamp VARCHAR(64) NOT NULL COMMENT 'Original ISO 8601 event time.',
	conversation_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream conversation identifier.',
	parent_run_id VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Original upstream parent run identifier.',
	agent_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream agent name.',
	tool_call_id LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream tool call identifier.',
	tool_name LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream tool name.',
	error LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Upstream boundary error text.',
	metadata JSON NULL COMMENT 'Upstream string metadata encoded as JSON, unless referenced from the native run.',
	run_fields SMALLINT UNSIGNED NOT NULL COMMENT 'Native run field references: conversation 1, parent 2, agent 4, metadata 8.' DEFAULT 0,
	body_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Integrity digest of the managed lifecycle source body.',
	fields_sha256 CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Integrity digest of managed lifecycle fields excluding metadata.',
	idempotency_key VARBINARY(128) NOT NULL COMMENT 'Exact UTF-8 replay key or a reserved unique occurrence key for an unkeyed append.',
	store_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete store_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_store_locator_run_id (store_locator, run_id),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_locator_run_id_idempotency_key (store_locator, run_id, idempotency_key)
)ENGINE=InnoDB COMMENT='Append-only upstream agent execution boundary events.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

-- Versioned assets and change receipts

CREATE TABLE ai_asset_heads (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original AssetStore namespace.',
	store_revision BIGINT NOT NULL COMMENT 'Last committed namespace-wide AssetStore revision.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_namespace (namespace)
)ENGINE=InnoDB COMMENT='Namespace-level optimistic revision head that serializes AssetStore mutations.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_asset_entries (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original AssetStore namespace.',
	kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original Asset kind namespace.',
	asset_id VARBINARY(512) NOT NULL COMMENT 'Exact UTF-8 bytes of the opaque Asset identifier.',
	entry_revision BIGINT NOT NULL COMMENT 'Latest committed per-asset revision.',
	store_revision BIGINT NOT NULL COMMENT 'Namespace-wide revision that produced this current projection.',
	payload_json JSON NOT NULL COMMENT 'Versioned canonical current AssetInfo payload including status, metadata, content reference, and semantic timestamps.',
	asset_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete asset_id; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_namespace_store_revision (namespace, store_revision),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_namespace_kind_asset_locator (namespace, kind, asset_locator)
)ENGINE=InnoDB COMMENT='Current AssetStore projection for the latest committed revision of each asset key.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_asset_changes (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	entry_id BIGINT NOT NULL COMMENT 'Surrogate identifier of the Asset entry owning this immutable revision.',
	entry_revision BIGINT NOT NULL COMMENT 'Immutable per-asset revision number represented by this history row.',
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original AssetStore namespace.',
	store_revision BIGINT NOT NULL COMMENT 'Namespace-wide revision in which this asset change committed.',
	payload_json JSON NOT NULL COMMENT 'Versioned canonical immutable AssetInfo history payload.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_entry_id_entry_revision (entry_id, entry_revision),
	UNIQUE KEY uk_namespace_store_revision_entry_id (namespace, store_revision, entry_id)
)ENGINE=InnoDB COMMENT='Immutable AssetStore history containing every committed per-asset revision.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_asset_batch_receipts (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original AssetStore namespace.',
	idempotency_key VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original caller batch idempotency key.',
	payload_json JSON NOT NULL COMMENT 'Versioned committed Asset batch receipt without input file bytes.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_namespace_idempotency_key (namespace, idempotency_key)
)ENGINE=InnoDB COMMENT='Durable idempotency receipts for committed atomic AssetStore batches.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

-- Object bodies and chunks

CREATE TABLE ai_objects (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical ObjectStore identifier that namespaces object keys.',
	object_key VARBINARY(1024) NOT NULL COMMENT 'Exact UTF-8 bytes of the opaque ObjectStore key.',
	content_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 digest of the immutable object bytes.',
	size BIGINT NOT NULL COMMENT 'Exact immutable object size in bytes.',
	object_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Single SHA-256 lookup projection of the complete object_key; raw identity must also match.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_id_object_locator (store_id, object_locator)
)ENGINE=InnoDB COMMENT='Immutable ObjectStore headers containing canonical object identity, content digest, and size.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_object_chunks (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	object_id BIGINT NOT NULL COMMENT 'Surrogate identifier of the immutable ObjectStore object owning this chunk.',
	chunk_index BIGINT NOT NULL COMMENT 'Zero-based ordered chunk position within the immutable object.',
	content LONGBLOB NOT NULL COMMENT 'Binary content bytes for this immutable object chunk.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_object_id_chunk_index (object_id, chunk_index)
)ENGINE=InnoDB COMMENT='Ordered binary chunks that compose immutable ObjectStore content.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

-- Tracing and metric observations

CREATE TABLE ai_trace_events (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	store_key VARBINARY(64) NOT NULL COMMENT 'Exact Trace store partition identity.',
	run_id VARBINARY(128) NOT NULL COMMENT 'Original UTF-8 agent run identifier.',
	agent_conversation_id VARBINARY(128) NULL COMMENT 'Inline event conversation identity; Native references retain source ownership.',
	parent_agent_run_id VARBINARY(128) NULL COMMENT 'Inline event parent run identity; Native references retain source ownership.',
	agent_id LONGBLOB NULL COMMENT 'Untruncated inline agent identifier as UTF-8 bytes; null for Native references.',
	sequence BIGINT NOT NULL COMMENT 'Committed position within the run event stream.',
	event_index BIGINT NOT NULL COMMENT 'Original event index reported by its producer.',
	idempotency_key LONGBLOB NULL COMMENT 'Untruncated inline event replay key as UTF-8 bytes; null for Native references.',
	kind VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Trace event type.',
	step_index BIGINT NOT NULL COMMENT 'Original agent graph step index.',
	timestamp VARCHAR(64) NOT NULL COMMENT 'Original event timestamp encoded as ISO 8601.',
	order_key VARBINARY(16) NOT NULL COMMENT 'UTC event timestamp and sequence for stable chronological paging.',
	request_identity VARBINARY(128) NULL COMMENT 'Original model request correlation string; leading zeros remain significant.',
	request_number VARBINARY(54) NULL COMMENT 'Unsigned binary request number used only for numeric filtering.',
	message_identity VARBINARY(128) NULL COMMENT 'Successful response message identity; other message metadata remains in the body.',
	tool_call_id VARBINARY(128) NULL COMMENT 'Original UTF-8 tool call identifier.',
	tool_name LONGBLOB NULL COMMENT 'Untruncated inline tool name as UTF-8 bytes; null for Native references.',
	history_identity VARBINARY(128) NULL COMMENT 'Eligible raw request or tool identity for exact history role lookup.',
	error_code VARBINARY(128) NULL COMMENT 'Tool error code used to distinguish unresolved external effects.',
	error LONGBLOB NULL COMMENT 'Untruncated inline error text as UTF-8 bytes; null for Native references.',
	payload_json JSON NOT NULL COMMENT 'Residual event metadata and extensions, or a Native source reference.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_store_key_run_id_history_identity (store_key, run_id, history_identity),
	KEY ix_store_key_run_id_message_identity (store_key, run_id, message_identity),
	KEY ix_store_key_run_id_order_key (store_key, run_id, order_key),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_store_key_run_id_sequence (store_key, run_id, sequence)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin ROW_FORMAT=DYNAMIC COMMENT='Immutable Trace events with direct history correlation and chronological queries.';

CREATE TABLE ai_metric_definitions (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Metrics namespace owning the definition.',
	metric_name VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Versioned metric definition name.',
	revision BIGINT NOT NULL COMMENT 'Metric semantic revision.',
	observation_kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical observation kind consumed by the metric.',
	payload_json JSON NOT NULL COMMENT 'Versioned canonical MetricDefinitionEnvelope payload.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_namespace_metric_name_revision (namespace, metric_name, revision)
)ENGINE=InnoDB COMMENT='Custom metric definitions.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

CREATE TABLE ai_metric_observations (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	namespace VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original Metrics namespace retained after pruning.',
	observation_id VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Original observation identity retained after pruning.',
	payload_digest CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 of the canonical ObservationEnvelope payload.',
	kind VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Canonical observation kind query projection.',
	occurred_at DATETIME(6) NOT NULL COMMENT 'Canonical UTC observation occurrence time.',
	payload_json JSON NULL COMMENT 'Versioned canonical immutable ObservationEnvelope payload.',
	active_occurred_at DATETIME(6) NULL COMMENT 'Derived occurrence time for active observations only.',
	pruned_at DATETIME(6) NULL COMMENT 'Backend time of the first payload pruning transition.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	KEY ix_created_at (created_at),
	KEY ix_namespace_active_occurred_at (namespace, active_occurred_at),
	KEY ix_namespace_kind_active_occurred_at (namespace, kind, active_occurred_at),
	KEY ix_updated_at (updated_at),
	UNIQUE KEY uk_namespace_observation_id (namespace, observation_id)
)ENGINE=InnoDB COMMENT='Immutable metric observations and retained pruning receipts.' CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COLLATE utf8mb4_bin;

-- Official StepStore physical mapping.



-- Regenerable conversation authority

CREATE TABLE ai_sessions (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	scope_key VARBINARY(32) NOT NULL COMMENT 'Partition of one tenant in the configured session store.',
	session_key VARBINARY(64) NOT NULL COMMENT 'Exact session identity.',
	status VARCHAR(8) NOT NULL COMMENT 'Session lifecycle status.',
	revision BIGINT NOT NULL COMMENT 'Compare-and-swap revision of the session.',
	active_turn_id VARBINARY(128) COMMENT 'Accepted turn occupying the session.',
	last_turn_no BIGINT NOT NULL COMMENT 'Last finalized turn number.',
	config_json JSON NOT NULL COMMENT 'Session metadata, workspace path and instruction scopes.',
	lineage_json JSON COMMENT 'Immutable referenced turn ranges for a fork.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	INDEX ix_created_at (created_at),
	INDEX ix_updated_at (updated_at),
	UNIQUE INDEX uk_scope_key_session_key (scope_key, session_key)
)ENGINE=InnoDB ROW_FORMAT=DYNAMIC COMMENT='Identity, configuration and publication head of a regenerable conversation.' CHARSET=utf8mb4 COLLATE utf8mb4_bin;

CREATE TABLE ai_session_turns (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT,
	scope_key VARBINARY(32) NOT NULL COMMENT 'Partition of one tenant in the configured session store.',
	session_key VARBINARY(64) NOT NULL COMMENT 'Session owning the turn.',
	turn_id VARBINARY(128) NOT NULL COMMENT 'Exact public turn identity.',
	request_key VARBINARY(64) NOT NULL COMMENT 'Idempotency key scoped to the session.',
	turn_no BIGINT NOT NULL COMMENT 'Sequential turn position within the session.',
	context_from BIGINT NOT NULL COMMENT 'First selected context position, inclusive.',
	context_through BIGINT NOT NULL COMMENT 'Finalized context boundary at admission.',
	state VARCHAR(10) NOT NULL COMMENT 'Admission, cancellation or terminal state.',
	generation BIGINT NOT NULL COMMENT 'Fencing generation of the accepted producer.',
	lease_until BIGINT COMMENT 'Expiry in database-clock seconds since epoch.',
	binding_key VARBINARY(32) NOT NULL COMMENT 'Frozen Agent binding in the same scope.',
	request_json JSON NOT NULL COMMENT 'Accepted principal and canonical request.',
	split_source_receipt VARBINARY(64) COMMENT 'Exact separately committed Step receipt; absent for same-group publication.',
	result_json JSON COMMENT 'Canonical terminal output and continuation material.',
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (id),
	INDEX ix_created_at (created_at),
	INDEX ix_updated_at (updated_at),
	UNIQUE INDEX uk_scope_key_session_key_request_key (scope_key, session_key, request_key),
	UNIQUE INDEX uk_scope_key_session_key_turn_no (scope_key, session_key, turn_no),
	UNIQUE INDEX uk_scope_key_turn_id (scope_key, turn_id)
)ENGINE=InnoDB ROW_FORMAT=DYNAMIC COMMENT='Authoritative admission and publication of one regenerable conversation turn.' CHARSET=utf8mb4 COLLATE utf8mb4_bin;
