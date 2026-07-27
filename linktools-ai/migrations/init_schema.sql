-- linktools-ai init schema (MySQL 5.7+/8.0)
-- Design + rationale: .superpowers/specs/2026-07-27-enterprise-db-schema-conformance-design.md
-- Conventions: ai_ table prefix; surrogate BIGINT id PK; natural keys UNIQUE (wide ones via
-- BINARY(32) hash columns); created_at/updated_at built-ins on every table; uk_/ix_ index names.

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ===== Storage kernel (Base) =====

CREATE TABLE `ai_storage_objects` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `key` VARCHAR(1024) NOT NULL COMMENT 'Object key',
  `key_hash` BINARY(32) NOT NULL COMMENT 'Object key hash',
  `etag` VARCHAR(64) NOT NULL COMMENT 'ETag',
  `version` INT NOT NULL COMMENT 'Object version',
  `content_type` VARCHAR(255) NULL COMMENT 'Content-Type',
  `size` INT NOT NULL COMMENT 'Size (bytes)',
  `content` LONGBLOB NOT NULL COMMENT 'Object content',
  `modified_at` DATETIME NOT NULL COMMENT 'Modified at',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `tombstone` TINYINT(1) NOT NULL COMMENT 'Tombstone flag',
  `commit_revision` INT NOT NULL COMMENT 'Commit revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_key_hash` (`key_hash`),
  KEY `ix_key` (`key`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Object current state (incl. tombstones)';

CREATE TABLE `ai_storage_object_versions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `key` VARCHAR(1024) NOT NULL COMMENT 'Object key',
  `key_hash` BINARY(32) NOT NULL COMMENT 'Object key hash',
  `version` INT NOT NULL COMMENT 'Object version',
  `etag` VARCHAR(64) NOT NULL COMMENT 'ETag',
  `content_type` VARCHAR(255) NULL COMMENT 'Content-Type',
  `size` INT NOT NULL COMMENT 'Size (bytes)',
  `content` LONGBLOB NULL COMMENT 'Version content',
  `modified_at` DATETIME NOT NULL COMMENT 'Modified at',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `tombstone` TINYINT(1) NOT NULL COMMENT 'Tombstone flag',
  `commit_revision` INT NOT NULL COMMENT 'Commit revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_key_hash_version` (`key_hash`, `version`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Object version history (append-only)';

CREATE TABLE `ai_storage_object_idempotency` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `key_hash` BINARY(32) NOT NULL COMMENT 'Idempotency key hash',
  `key` VARCHAR(1024) NOT NULL COMMENT 'Idempotency key',
  `request_hash` VARCHAR(64) NOT NULL COMMENT 'Request hash',
  `operation` VARCHAR(32) NOT NULL COMMENT 'Operation',
  `result_key_hash` BINARY(32) NULL COMMENT 'Result key hash',
  `result_key` VARCHAR(1024) NULL COMMENT 'Result key',
  `result_version` INT NULL COMMENT 'Result version',
  `commit_revision` INT NOT NULL DEFAULT 0 COMMENT 'Commit revision',
  `result_json` TEXT NULL COMMENT 'Result (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_key_hash` (`key_hash`),
  KEY `ix_key` (`key`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Object-store idempotency records';

CREATE TABLE `ai_storage_object_revision` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `value` INT NOT NULL COMMENT 'Current revision number',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  KEY `ix_value` (`value`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Global object revision counter';

CREATE TABLE `ai_storage_schema_version` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `component` VARCHAR(128) NOT NULL COMMENT 'Schema component',
  `version` INT NOT NULL COMMENT 'Component version',
  `checksum` VARCHAR(128) NOT NULL COMMENT 'Schema checksum',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_component` (`component`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Storage schema version';

-- ===== AI domain (DomainBase) =====

CREATE TABLE `ai_idempotency` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `idempotency_id` VARCHAR(128) NOT NULL COMMENT 'Idempotency record id',
  `scope` VARCHAR(128) NOT NULL COMMENT 'Scope',
  `key` VARCHAR(512) NOT NULL COMMENT 'Idempotency key',
  `scope_key_hash` BINARY(32) NOT NULL COMMENT 'Scope+key hash',
  `request_hash` VARCHAR(64) NOT NULL COMMENT 'Request hash',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `result_json` TEXT NULL COMMENT 'Result (JSON)',
  `error_message` TEXT NULL COMMENT 'Error message',
  `completed_at` DATETIME NULL COMMENT 'Completed at',
  `expires_at` DATETIME NULL COMMENT 'Expires at',
  `owner_id` VARCHAR(128) NULL COMMENT 'Owner id',
  `generation` INT NOT NULL DEFAULT 0 COMMENT 'Fencing generation',
  `claimed_at` DATETIME NULL COMMENT 'Claimed at',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `receipt_artifact_id` VARCHAR(256) NULL COMMENT 'Receipt artifact id',
  `binding_fingerprint` VARCHAR(128) NULL COMMENT 'Binding fingerprint',
  `result_processor_revision` VARCHAR(128) NULL COMMENT 'Result processor revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_idempotency_id` (`idempotency_id`),
  UNIQUE KEY `uk_scope_key_hash` (`scope_key_hash`),
  KEY `ix_scope` (`scope`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Tool-call idempotency records';

CREATE TABLE `ai_runs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `root_run_id` VARCHAR(128) NOT NULL COMMENT 'Root run id',
  `parent_run_id` VARCHAR(128) NULL COMMENT 'Parent run id',
  `session_id` VARCHAR(128) NOT NULL COMMENT 'Session id',
  `runnable_id` VARCHAR(255) NOT NULL COMMENT 'Runnable id',
  `runnable_type` VARCHAR(32) NOT NULL COMMENT 'Runnable type',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `started_at` DATETIME NULL COMMENT 'Started at',
  `finished_at` DATETIME NULL COMMENT 'Finished at',
  `cancel_requested_at` DATETIME NULL COMMENT 'Cancel requested at',
  `cancel_requested_by` VARCHAR(128) NULL COMMENT 'Cancel requested by',
  `worker_id` VARCHAR(128) NULL COMMENT 'Worker id',
  `execution_token` VARCHAR(256) NULL COMMENT 'Execution token',
  `heartbeat_at` DATETIME NULL COMMENT 'Last heartbeat',
  `manifest_id` VARCHAR(256) NULL COMMENT 'Manifest id',
  `resumability` VARCHAR(32) NULL COMMENT 'Resumability',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_run_id` (`run_id`),
  KEY `ix_root_run_id` (`root_run_id`),
  KEY `ix_parent_run_id` (`parent_run_id`),
  KEY `ix_session_id` (`session_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Run lifecycle';

CREATE TABLE `ai_run_checkpoints` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `checkpoint_id` VARCHAR(128) NOT NULL COMMENT 'Checkpoint id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `format` VARCHAR(32) NOT NULL COMMENT 'Payload format',
  `schema_version` INT NOT NULL COMMENT 'Payload schema version',
  `payload` LONGBLOB NOT NULL COMMENT 'Checkpoint payload',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_checkpoint_id` (`checkpoint_id`),
  UNIQUE KEY `uk_run_id_sequence` (`run_id`, `sequence`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Run checkpoint';

CREATE TABLE `ai_run_checkpoint_counters` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `last_sequence` INT NOT NULL COMMENT 'Last sequence',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Run checkpoint sequence counter';

CREATE TABLE `ai_run_definitions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `runnable_type` VARCHAR(32) NOT NULL COMMENT 'Runnable type',
  `runnable_id` VARCHAR(255) NOT NULL COMMENT 'Runnable id',
  `serialized_spec_json` TEXT NOT NULL COMMENT 'Serialized spec (JSON)',
  `spec_fingerprint` VARCHAR(64) NOT NULL COMMENT 'Spec fingerprint',
  `user_id` VARCHAR(128) NULL COMMENT 'User id',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `workspace` VARCHAR(128) NULL COMMENT 'Workspace',
  `manifest_json` TEXT NULL COMMENT 'Manifest (JSON)',
  `resumability` VARCHAR(32) NULL COMMENT 'Resumability',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Run definition snapshot';

CREATE TABLE `ai_sessions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `session_id` VARCHAR(128) NOT NULL COMMENT 'Session id',
  `parent_id` VARCHAR(128) NULL COMMENT 'Parent session id',
  `user_id` VARCHAR(128) NULL COMMENT 'User id',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_session_id` (`session_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Session';

CREATE TABLE `ai_session_messages` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `message_id` VARCHAR(128) NOT NULL COMMENT 'Message id',
  `session_id` VARCHAR(128) NOT NULL COMMENT 'Session id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `role` VARCHAR(32) NOT NULL COMMENT 'Role',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `run_id` VARCHAR(128) NULL COMMENT 'Run id',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `commit_id` VARCHAR(200) NULL COMMENT 'Commit id',
  `commit_hash` BINARY(32) NOT NULL COMMENT 'Commit hash',
  `batch_index` INT NOT NULL DEFAULT 0 COMMENT 'Commit batch index',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_message_id` (`message_id`),
  UNIQUE KEY `uk_session_id_sequence` (`session_id`, `sequence`),
  UNIQUE KEY `uk_session_id_commit_hash_batch_index` (`session_id`, `commit_hash`, `batch_index`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Session message';

CREATE TABLE `ai_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `event_id` VARCHAR(128) NOT NULL COMMENT 'Event id',
  `stream_id` VARCHAR(64) NOT NULL COMMENT 'Stream id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `occurred_at` DATETIME NOT NULL COMMENT 'Occurred at',
  `root_run_id` VARCHAR(128) NOT NULL COMMENT 'Root run id',
  `parent_run_id` VARCHAR(128) NULL COMMENT 'Parent run id',
  `session_id` VARCHAR(128) NOT NULL COMMENT 'Session id',
  `runnable_id` VARCHAR(255) NOT NULL COMMENT 'Runnable id',
  `event_type` VARCHAR(32) NOT NULL COMMENT 'Event type',
  `schema_version` INT NOT NULL DEFAULT 1 COMMENT 'Event schema version',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `metadata_json` TEXT NULL COMMENT 'Metadata (JSON)',
  `commit_id` VARCHAR(200) NULL COMMENT 'Commit id',
  `commit_hash` BINARY(32) NOT NULL COMMENT 'Commit hash',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_event_id` (`event_id`),
  UNIQUE KEY `uk_stream_id_sequence` (`stream_id`, `sequence`),
  UNIQUE KEY `uk_stream_id_commit_hash_event_type` (`stream_id`, `commit_hash`, `event_type`),
  KEY `ix_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Event stream';

CREATE TABLE `ai_swarm_runs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `swarm_run_id` VARCHAR(128) NOT NULL COMMENT 'Swarm run id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Driving run id',
  `round` INT NOT NULL COMMENT 'Swarm round',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `input_tokens` INT NOT NULL COMMENT 'Input tokens',
  `output_tokens` INT NOT NULL COMMENT 'Output tokens',
  `total_cost` TEXT NOT NULL COMMENT 'Total cost',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `execution_token` VARCHAR(256) NULL COMMENT 'Execution token',
  `execution_owner_id` VARCHAR(256) NULL COMMENT 'Execution owner id',
  `execution_generation` INT NOT NULL DEFAULT 0 COMMENT 'Execution generation',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_swarm_run_id` (`swarm_run_id`),
  KEY `ix_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Swarm run';

CREATE TABLE `ai_swarm_tasks` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `task_id` VARCHAR(128) NOT NULL COMMENT 'Task id',
  `swarm_run_id` VARCHAR(128) NOT NULL COMMENT 'Swarm run id',
  `parent_task_id` VARCHAR(128) NULL COMMENT 'Parent task id',
  `assigned_agent_id` VARCHAR(128) NULL COMMENT 'Assigned agent id',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `attempts` INT NOT NULL COMMENT 'Attempt count',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `claimed_at` DATETIME NULL COMMENT 'Claimed at',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `active_run_id` VARCHAR(128) NULL COMMENT 'Active run id',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_id` (`task_id`),
  KEY `ix_swarm_run_id` (`swarm_run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Swarm task';

CREATE TABLE `ai_swarm_task_attempts` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `attempt_id` VARCHAR(128) NOT NULL COMMENT 'Attempt id',
  `task_id` VARCHAR(128) NOT NULL COMMENT 'Task id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `agent_id` VARCHAR(128) NOT NULL COMMENT 'Agent id',
  `attempt` INT NOT NULL COMMENT 'Attempt number',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `started_at` DATETIME NOT NULL COMMENT 'Started at',
  `finished_at` DATETIME NULL COMMENT 'Finished at',
  `error_json` TEXT NULL COMMENT 'Error (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_attempt_id` (`attempt_id`),
  KEY `ix_task_id` (`task_id`),
  KEY `ix_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Swarm task attempt';

CREATE TABLE `ai_memories` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `memory_id` VARCHAR(128) NOT NULL COMMENT 'Memory id',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `owner_id` VARCHAR(128) NOT NULL COMMENT 'Owner id',
  `content` TEXT NOT NULL COMMENT 'Memory content',
  `category` VARCHAR(64) NULL COMMENT 'Category',
  `confidence` DECIMAL(5,4) NULL COMMENT 'Confidence [0,1]',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `user_id` VARCHAR(128) NULL COMMENT 'User id',
  `workspace_id` VARCHAR(128) NULL COMMENT 'Workspace id',
  `session_id` VARCHAR(128) NULL COMMENT 'Session id',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_memory_id` (`memory_id`),
  KEY `ix_tenant_id` (`tenant_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Memory entry';

CREATE TABLE `ai_approvals` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `approval_id` VARCHAR(128) NOT NULL COMMENT 'Approval id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `tool_call_id` VARCHAR(128) NOT NULL COMMENT 'Tool call id',
  `tool_name` VARCHAR(255) NOT NULL COMMENT 'Tool name',
  `arguments_hash` VARCHAR(128) NOT NULL COMMENT 'Arguments hash',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `resolved_at` DATETIME NULL COMMENT 'Resolved at',
  `resolved_by` VARCHAR(128) NULL COMMENT 'Resolved by',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `descriptor_fingerprint` VARCHAR(128) NULL COMMENT 'Descriptor fingerprint',
  `handler_revision` VARCHAR(256) NULL COMMENT 'Handler revision',
  `provider_revision` VARCHAR(256) NULL COMMENT 'Provider revision',
  `policy_revision` VARCHAR(256) NULL COMMENT 'Policy revision',
  `capability_revision` VARCHAR(256) NULL COMMENT 'Capability revision',
  `schema_version` INT NOT NULL DEFAULT 1 COMMENT 'Schema version',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_approval_id` (`approval_id`),
  UNIQUE KEY `uk_run_id_tool_call_id` (`run_id`, `tool_call_id`),
  KEY `ix_tenant_id` (`tenant_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Tool-call approval';

CREATE TABLE `ai_jobs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `job_id` VARCHAR(128) NOT NULL COMMENT 'Job id',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `tenant_id` VARCHAR(128) NOT NULL COMMENT 'Tenant id',
  `root_task_id` VARCHAR(128) NULL COMMENT 'Root task id',
  `input_artifact_id` VARCHAR(128) NULL COMMENT 'Input artifact id',
  `output_artifact_id` VARCHAR(128) NULL COMMENT 'Output artifact id',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `started_at` DATETIME NULL COMMENT 'Started at',
  `finished_at` DATETIME NULL COMMENT 'Finished at',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_job_id` (`job_id`),
  KEY `ix_status_created_at` (`status`, `created_at`),
  KEY `ix_tenant_id_status` (`tenant_id`, `status`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Reliable-task job';

CREATE TABLE `ai_tasks` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `task_id` VARCHAR(128) NOT NULL COMMENT 'Task id',
  `job_id` VARCHAR(128) NOT NULL COMMENT 'Job id',
  `parent_task_id` VARCHAR(128) NULL COMMENT 'Parent task id',
  `key` VARCHAR(255) NOT NULL COMMENT 'Task key',
  `job_key_hash` BINARY(32) NOT NULL COMMENT 'Job+key hash',
  `handler` VARCHAR(255) NOT NULL COMMENT 'Handler',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `input_artifact_id` VARCHAR(128) NULL COMMENT 'Input artifact id',
  `output_artifact_id` VARCHAR(128) NULL COMMENT 'Output artifact id',
  `attempt_count` INT NOT NULL COMMENT 'Attempt count',
  `available_at` DATETIME NOT NULL COMMENT 'Available at',
  `lease_owner` VARCHAR(128) NULL COMMENT 'Lease owner',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `fencing_token` INT NOT NULL COMMENT 'Fencing token',
  `active_attempt_id` VARCHAR(128) NULL COMMENT 'Active attempt id',
  `timeout_ms` INT NULL COMMENT 'Timeout (ms)',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_id` (`task_id`),
  UNIQUE KEY `uk_job_key_hash` (`job_key_hash`),
  KEY `ix_job_id_status` (`job_id`, `status`),
  KEY `ix_status_available_at` (`status`, `available_at`),
  KEY `ix_handler_status_available_at` (`handler`(64), `status`, `available_at`),
  KEY `ix_lease_expires_at` (`lease_expires_at`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Reliable task';

CREATE TABLE `ai_task_attempts` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `attempt_id` VARCHAR(128) NOT NULL COMMENT 'Attempt id',
  `task_id` VARCHAR(128) NOT NULL COMMENT 'Task id',
  `job_id` VARCHAR(128) NOT NULL COMMENT 'Job id',
  `attempt` INT NOT NULL COMMENT 'Attempt number',
  `worker_id` VARCHAR(128) NOT NULL COMMENT 'Worker id',
  `fencing_token` INT NOT NULL COMMENT 'Fencing token',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `run_id` VARCHAR(128) NULL COMMENT 'Run id',
  `started_at` DATETIME NOT NULL COMMENT 'Started at',
  `finished_at` DATETIME NULL COMMENT 'Finished at',
  `failure_kind` VARCHAR(64) NULL COMMENT 'Failure kind',
  `error_type` VARCHAR(255) NULL COMMENT 'Error type',
  `error_message` TEXT NULL COMMENT 'Error message',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_attempt_id` (`attempt_id`),
  UNIQUE KEY `uk_task_id_attempt` (`task_id`, `attempt`),
  KEY `ix_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Reliable-task attempt';

CREATE TABLE `ai_task_transitions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `job_id` VARCHAR(128) NOT NULL COMMENT 'Job id',
  `task_id` VARCHAR(128) NULL COMMENT 'Task id',
  `attempt_id` VARCHAR(128) NULL COMMENT 'Attempt id',
  `from_status` VARCHAR(32) NULL COMMENT 'From status',
  `to_status` VARCHAR(32) NOT NULL COMMENT 'To status',
  `reason` VARCHAR(64) NOT NULL COMMENT 'Reason',
  `occurred_at` DATETIME NOT NULL COMMENT 'Occurred at',
  `data_json` TEXT NOT NULL COMMENT 'Context (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  KEY `ix_job_id` (`job_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Task state-transition audit';

CREATE TABLE `ai_task_signals` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `signal_id` VARCHAR(128) NOT NULL COMMENT 'Signal id',
  `job_id` VARCHAR(128) NOT NULL COMMENT 'Job id',
  `name` VARCHAR(255) NOT NULL COMMENT 'Signal name',
  `correlation_key` VARCHAR(255) NOT NULL COMMENT 'Correlation key',
  `payload_artifact_id` VARCHAR(128) NULL COMMENT 'Payload artifact id',
  `consumed_by_task_id` VARCHAR(128) NULL COMMENT 'Consumed by task id',
  `data_json` TEXT NOT NULL COMMENT 'Context (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_signal_id` (`signal_id`),
  KEY `ix_job_id_name` (`job_id`, `name`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Task signal';

CREATE TABLE `ai_eval_runs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `eval_run_id` VARCHAR(128) NOT NULL COMMENT 'Eval run id',
  `suite_id` VARCHAR(128) NOT NULL COMMENT 'Suite id',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `started_at` DATETIME NULL COMMENT 'Started at',
  `finished_at` DATETIME NULL COMMENT 'Finished at',
  `data_json` TEXT NOT NULL COMMENT 'Payload (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_eval_run_id` (`eval_run_id`),
  KEY `ix_suite_id` (`suite_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Evaluation run';

CREATE TABLE `ai_eval_results` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `result_id` VARCHAR(128) NOT NULL COMMENT 'Result id',
  `eval_run_id` VARCHAR(128) NOT NULL COMMENT 'Eval run id',
  `case_id` VARCHAR(128) NOT NULL COMMENT 'Case id',
  `run_id` VARCHAR(128) NULL COMMENT 'Run id',
  `job_id` VARCHAR(128) NULL COMMENT 'Job id',
  `task_id` VARCHAR(128) NULL COMMENT 'Task id',
  `output_artifact_id` VARCHAR(128) NULL COMMENT 'Output artifact id',
  `snapshot_artifact_id` VARCHAR(128) NULL COMMENT 'Snapshot artifact id',
  `error_type` VARCHAR(128) NULL COMMENT 'Error type',
  `error_message` TEXT NULL COMMENT 'Error message',
  `data_json` TEXT NOT NULL COMMENT 'Scores/metrics (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_result_id` (`result_id`),
  KEY `ix_eval_run_id` (`eval_run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Evaluation result';

CREATE TABLE `ai_artifact_records` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `artifact_id` VARCHAR(128) NOT NULL COMMENT 'Artifact id',
  `tenant_id` VARCHAR(128) NOT NULL COMMENT 'Tenant id',
  `content_hash` VARCHAR(64) NOT NULL COMMENT 'Content hash (SHA-256)',
  `producer_kind` VARCHAR(64) NULL COMMENT 'Producer kind',
  `producer_id` VARCHAR(255) NULL COMMENT 'Producer id',
  `run_id` VARCHAR(128) NULL COMMENT 'Run id',
  `data_json` TEXT NOT NULL COMMENT 'Record (JSON)',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_artifact_id` (`artifact_id`),
  KEY `ix_content_hash` (`content_hash`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Artifact lineage record';

CREATE TABLE `ai_run_commit_log` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `commit_id` VARCHAR(200) NOT NULL COMMENT 'Commit id',
  `commit_hash` BINARY(32) NOT NULL COMMENT 'Commit hash',
  `operation` VARCHAR(64) NOT NULL COMMENT 'Operation',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `request_hash` BINARY(32) NOT NULL COMMENT 'Request hash',
  `result_json` TEXT NOT NULL COMMENT 'Result (JSON)',
  `result_payload` LONGBLOB NULL COMMENT 'Result payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_commit_hash` (`commit_hash`),
  KEY `ix_commit_id` (`commit_id`(128)),
  KEY `ix_run_id` (`run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Run commit replay log';

CREATE TABLE `ai_swarm_commit_log` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `commit_id` VARCHAR(200) NOT NULL COMMENT 'Commit id',
  `commit_hash` BINARY(32) NOT NULL COMMENT 'Commit hash',
  `operation` VARCHAR(64) NOT NULL COMMENT 'Operation',
  `swarm_run_id` VARCHAR(128) NOT NULL COMMENT 'Swarm run id',
  `request_hash` BINARY(32) NOT NULL COMMENT 'Request hash',
  `result_json` TEXT NOT NULL COMMENT 'Result (JSON)',
  `result_payload` LONGBLOB NOT NULL COMMENT 'Result payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_commit_hash` (`commit_hash`),
  KEY `ix_commit_id` (`commit_id`(128)),
  KEY `ix_swarm_run_id` (`swarm_run_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Swarm commit replay log';

SET FOREIGN_KEY_CHECKS = 1;
