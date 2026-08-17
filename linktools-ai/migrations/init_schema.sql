-- linktools-ai canonical MySQL schema reference.
--
-- This file is manually reviewed DBA input, not an application bootstrap.
-- Provisioning is performed only by the explicit migrate/provision API.
-- 1. All business tables use the ai_ prefix.
-- 2. The active schema contains exactly 24 tables.
-- 3. There is no schema manifest table.
-- 4. Every table and column has a business COMMENT.
-- 5. Placeholder comments are prohibited.
-- 6. Every table uses an id BIGINT AUTO_INCREMENT surrogate primary key.
-- 7. updated_at is immediately before created_at.
-- 8. MySQL updated_at uses DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP.
-- 9. Every table retains ix_updated_at and ix_created_at.
-- 10. UNIQUE names are uk_<ordered_columns>.
-- 11. INDEX names are ix_<ordered_columns>.
-- 12. MySQL index names do not contain table names.
-- 13. Composite UNIQUE and INDEX keys contain at most three physical columns.
-- 14. SHA-256 values use CHAR(64) with utf8mb4_bin.
-- 15. Wide columns carrying uniqueness never use prefix UNIQUE keys.
-- 16. Ordinary non-unique wide indexes use a 128-character prefix.
-- 17. Redundant indexes fully covered by UNIQUE or PRIMARY keys are prohibited.
-- 18. Low-selectivity status single-column indexes are prohibited.
-- 19. Foreign keys are prohibited.
-- 20. FLOAT columns are prohibited.
-- 21. JSON and LONGBLOB are limited to the real fields in this schema.
-- 22. This DDL and the canonical metadata describe the same schema.

SET NAMES utf8mb4;

CREATE TABLE `ai_runtime_sessions` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `session_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Session identifier',
  `owner_principal_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Owner principal identifier',
  `binding_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Binding definition SHA-256 digest',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `resource_generation` BIGINT NOT NULL COMMENT 'Resource generation',
  `cwd` TEXT COMMENT 'Working directory',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `continuation_step_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Continuation Step run identifier',
  `active_execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
    COMMENT 'Durably admitted execution identifier',
  `closed_at` DATETIME COMMENT 'Close time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_session_id` (`namespace_digest`, `tenant_id`, `session_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime sessions';

CREATE TABLE `ai_runtime_executions` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `session_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Session identifier',
  `binding_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Binding definition SHA-256 digest',
  `parent_execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Parent execution identifier',
  `root_execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Root execution identifier',
  `source_execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution lineage kind',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `event_sequence` BIGINT NOT NULL COMMENT 'Event sequence number',
  `agent_run_sequence` BIGINT NOT NULL COMMENT 'Agent run sequence number',
  `error_code` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error code',
  `safe_error_details` JSON NOT NULL COMMENT 'Safe error details',
  `memory_scope` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Memory scope',
  `conversation_step_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Conversation Step run identifier',
  `output_schema_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Output schema identifier',
  `output_schema_revision` BIGINT COMMENT 'Output schema revision',
  `output_schema_fingerprint` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Output schema fingerprint',
  `result_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result ObjectStore identifier',
  `result_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result object key',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `result_size` BIGINT COMMENT 'Result size in bytes',
  `stop_reason` TEXT COMMENT 'Stop reason',
  `model_requests` BIGINT COMMENT 'Model request count',
  `tool_calls` BIGINT COMMENT 'Tool call count',
  `input_tokens` BIGINT COMMENT 'Input token count',
  `output_tokens` BIGINT COMMENT 'Output token count',
  `cache_read_tokens` BIGINT COMMENT 'Cache read token count',
  `cache_write_tokens` BIGINT COMMENT 'Cache write token count',
  `result_created_at` DATETIME COMMENT 'Result creation time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_execution_id` (`namespace_digest`, `tenant_id`, `execution_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime executions';

CREATE TABLE `ai_runtime_idempotency` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `runtime_domain` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Runtime domain',
  `scope` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Idempotency scope',
  `idempotency_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Idempotency key SHA-256 digest',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `request_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Request SHA-256 digest',
  `resource_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Resource identifier',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `error_code` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error code',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime idempotency records';

CREATE TABLE `ai_runtime_events` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `sequence` BIGINT NOT NULL COMMENT 'Sequence number',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Kind',
  `payload` JSON NOT NULL COMMENT 'Event payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_execution_id` (`namespace_digest`, `tenant_id`, `execution_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime execution events';

CREATE TABLE `ai_runtime_task_graphs` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `graph_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Task graph identifier',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_graph_id` (`namespace_digest`, `tenant_id`, `graph_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime task graphs';

CREATE TABLE `ai_runtime_task_nodes` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `graph_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Task graph identifier',
  `node_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Task node identifier',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `dependencies` JSON NOT NULL COMMENT 'Dependency node set',
  `input` JSON COMMENT 'Input data',
  `budget_cost` BIGINT COMMENT 'Budget cost',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `owner` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Lease owner',
  `fence` BIGINT NOT NULL COMMENT 'Lease fence value',
  `lease_expires_at` DATETIME COMMENT 'Lease expiration time',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Execution identifier',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `error_code` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error code',
  `error_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error SHA-256 digest',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_graph_id` (`namespace_digest`, `tenant_id`, `graph_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime task nodes';

CREATE TABLE `ai_runtime_evaluations` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `evaluation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Evaluation identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `dataset_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Dataset identifier',
  `dataset_revision` BIGINT NOT NULL COMMENT 'Dataset revision',
  `evaluator_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Evaluator identifier',
  `evaluator_revision` BIGINT NOT NULL COMMENT 'Evaluator revision',
  `binding_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Binding definition SHA-256 digest',
  `output_schema_fingerprint` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Output schema fingerprint',
  `artifact_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Artifact SHA-256 digest',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `metrics` JSON NOT NULL COMMENT 'Evaluation metrics',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_evaluation_id` (`namespace_digest`, `tenant_id`, `evaluation_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime evaluations';

CREATE TABLE `ai_runtime_memories` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `memory_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Memory identifier',
  `memory_scope_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Memory scope SHA-256 digest',
  `content_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content ObjectStore identifier',
  `content_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content object key',
  `content_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content SHA-256 digest',
  `content_size` BIGINT NOT NULL COMMENT 'Content size in bytes',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_memory_id` (`namespace_digest`, `tenant_id`, `memory_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime memories';

CREATE TABLE `ai_runtime_artifacts` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `artifact_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Artifact identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `producer` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Artifact producer',
  `media_type` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Media type',
  `content_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content ObjectStore identifier',
  `content_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content object key',
  `content_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content SHA-256 digest',
  `content_size` BIGINT NOT NULL COMMENT 'Content size in bytes',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_artifact_id` (`namespace_digest`, `tenant_id`, `artifact_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime artifacts';

CREATE TABLE `ai_runtime_approvals` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `approval_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Approval identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `operation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation identifier',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `idempotency_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Idempotency key SHA-256 digest',
  `decision` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Approval decision',
  `decided_by` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Decision principal',
  `decision_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Approval decision SHA-256 digest',
  `decided_at` DATETIME COMMENT 'Decision time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_approval_id` (`namespace_digest`, `tenant_id`, `approval_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime approvals';

CREATE TABLE `ai_runtime_external_calls` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `call_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'External call identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `operation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation identifier',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `idempotency_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Idempotency key SHA-256 digest',
  `result_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result ObjectStore identifier',
  `result_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result object key',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `result_size` BIGINT COMMENT 'Result size in bytes',
  `payload_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Payload SHA-256 digest',
  `supplied_at` DATETIME COMMENT 'Result supplied time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_call_id` (`namespace_digest`, `tenant_id`, `call_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime external calls';

CREATE TABLE `ai_runtime_recovery_checkpoints` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Execution identifier',
  `step_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Step run identifier',
  `agent_run_sequence` BIGINT NOT NULL COMMENT 'Agent run sequence number',
  `state` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'State data',
  `handoff_phase` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Handoff phase',
  `input` JSON NOT NULL COMMENT 'Input data',
  `terminal_handoff` JSON COMMENT 'Terminal handoff data',
  `handoff_contract_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Handoff contract SHA-256 digest',
  `pending_operation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Pending operation identifier',
  `revision` BIGINT NOT NULL COMMENT 'Resource revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_execution_id` (`namespace_digest`, `tenant_id`, `execution_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime recovery checkpoints';

CREATE TABLE `ai_runtime_tool_operations` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `tool_operation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool operation identifier',
  `step_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Step run identifier',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool call identifier',
  `call_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool call identity SHA-256 digest',
  `idempotency_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Idempotency key SHA-256 digest',
  `tool_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool name',
  `arguments_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool arguments SHA-256 digest',
  `binding_fingerprint` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Binding fingerprint',
  `replay_safe` BOOL NOT NULL COMMENT 'Replay safety flag',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `owner` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Lease owner',
  `fence` BIGINT NOT NULL COMMENT 'Lease fence value',
  `lease_expires_at` DATETIME COMMENT 'Lease expiration time',
  `result_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result ObjectStore identifier',
  `result_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result object key',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `result_size` BIGINT COMMENT 'Result size in bytes',
  `error_code` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error code',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_tool_operation_id` (`namespace_digest`, `tenant_id`, `tool_operation_id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_call_digest` (`namespace_digest`, `tenant_id`, `call_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime tool operations';

CREATE TABLE `ai_runtime_operation_counters` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `runtime_domain` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Runtime domain',
  `resource_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Resource kind',
  `resource_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Resource identifier',
  `stream_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation stream SHA-256 digest',
  `last_sequence` BIGINT NOT NULL COMMENT 'Last allocated sequence',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_stream_digest` (`namespace_digest`, `stream_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime operation sequence counters';

CREATE TABLE `ai_runtime_operations` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `runtime_domain` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Runtime domain',
  `resource_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Resource kind',
  `resource_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Resource identifier',
  `stream_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation stream SHA-256 digest',
  `operation_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation kind',
  `operation_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation identifier',
  `operation_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Operation identity SHA-256 digest',
  `sequence` BIGINT NOT NULL COMMENT 'Sequence number',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `execution_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Execution identifier',
  `request_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Request SHA-256 digest',
  `result_ref` TEXT COMMENT 'Operation result reference',
  `result_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Result SHA-256 digest',
  `error_code` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Error code',
  `compactable` BOOL NOT NULL COMMENT 'Compaction eligibility flag',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_operation_digest` (`namespace_digest`, `tenant_id`, `operation_digest`),
  UNIQUE KEY `uk_namespace_digest_stream_digest_sequence` (`namespace_digest`, `stream_digest`, `sequence`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime operation ledger';

CREATE TABLE `ai_step_runs` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `runtime_domain` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Runtime domain',
  `run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Step run identifier',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `conversation_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Parent Step run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Agent name',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `last_event_index` BIGINT NOT NULL COMMENT 'Last event index',
  `last_snapshot_index` BIGINT NOT NULL COMMENT 'Last snapshot index',
  `last_effect_index` BIGINT NOT NULL COMMENT 'Last effect index',
  `started_at` DATETIME NOT NULL COMMENT 'Start time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_conversation_id` (`namespace_digest`, `tenant_id`, `conversation_id`(128)),
  KEY `ix_namespace_digest_tenant_id_parent_run_id` (`namespace_digest`, `tenant_id`, `parent_run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Step runs';

CREATE TABLE `ai_step_events` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Step run identifier',
  `event_index` BIGINT NOT NULL COMMENT 'Event index',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `kind` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Kind',
  `step_index` BIGINT NOT NULL COMMENT 'Step index',
  `timestamp` DATETIME NOT NULL COMMENT 'Event timestamp',
  `conversation_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Parent Step run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Agent name',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Tool call identifier',
  `tool_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Tool name',
  `error` TEXT COMMENT 'Error message',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_run_id` (`namespace_digest`, `tenant_id`, `run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Step events';

CREATE TABLE `ai_step_snapshots` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `runtime_domain` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Runtime domain',
  `run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Step run identifier',
  `snapshot_index` BIGINT NOT NULL COMMENT 'Snapshot index',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `step_index` BIGINT NOT NULL COMMENT 'Step index',
  `state` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'State data',
  `conversation_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Parent Step run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Agent name',
  `timestamp` DATETIME NOT NULL COMMENT 'Event timestamp',
  `media_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Media ObjectStore identifier',
  `messages` JSON NOT NULL COMMENT 'Message snapshot',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_run_id` (`namespace_digest`, `tenant_id`, `run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Step snapshots';

CREATE TABLE `ai_step_effects` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Step run identifier',
  `effect_index` BIGINT NOT NULL COMMENT 'Effect index',
  `identity_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Logical identity SHA-256 digest',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool call identifier',
  `tool_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool name',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `started_at` DATETIME NOT NULL COMMENT 'Start time',
  `ended_at` DATETIME COMMENT 'End time',
  `idempotency_key` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Idempotency key',
  `effect_summary` TEXT COMMENT 'Effect summary',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_tenant_id_identity_digest` (`namespace_digest`, `tenant_id`, `identity_digest`),
  KEY `ix_namespace_digest_tenant_id_run_id` (`namespace_digest`, `tenant_id`, `run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Step effects';

CREATE TABLE `ai_storage_objects` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `object_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Object key',
  `object_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Object key SHA-256 digest',
  `digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Object content SHA-256 digest',
  `size` BIGINT NOT NULL COMMENT 'Size in bytes',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_object_key_digest` (`object_key_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='ObjectStore objects';

CREATE TABLE `ai_storage_object_chunks` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `object_id` BIGINT NOT NULL COMMENT 'Object row identifier',
  `chunk_index` BIGINT NOT NULL COMMENT 'Chunk index',
  `content` LONGBLOB NOT NULL COMMENT 'Chunk content',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_object_id_chunk_index` (`object_id`, `chunk_index`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='ObjectStore object chunks';

CREATE TABLE `ai_asset_heads` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `store_revision` BIGINT NOT NULL COMMENT 'Global storage revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest` (`namespace_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Asset namespace heads';

CREATE TABLE `ai_asset_entries` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `asset_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset logical key SHA-256 digest',
  `asset_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset kind',
  `asset_id` VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset identifier',
  `entry_revision` BIGINT NOT NULL COMMENT 'File revision',
  `store_revision` BIGINT NOT NULL COMMENT 'Global storage revision',
  `etag` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'File content SHA-256 ETag',
  `size` BIGINT NOT NULL COMMENT 'Size in bytes',
  `status` VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `content_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Content ObjectStore identifier',
  `content_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Content object key',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `modified_at` DATETIME NOT NULL COMMENT 'File modification time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_asset_key_digest` (`namespace_digest`, `asset_key_digest`),
  KEY `ix_namespace_digest_store_revision` (`namespace_digest`, `store_revision`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Asset entries';

CREATE TABLE `ai_asset_changes` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Auto-increment primary key',
  `namespace_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace SHA-256 digest',
  `asset_key_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset logical key SHA-256 digest',
  `asset_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset kind',
  `asset_id` VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset identifier',
  `entry_revision` BIGINT NOT NULL COMMENT 'File revision',
  `store_revision` BIGINT NOT NULL COMMENT 'Global storage revision',
  `etag` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'File content SHA-256 ETag',
  `size` BIGINT NOT NULL COMMENT 'Size in bytes',
  `status` VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Status',
  `content_store_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Content ObjectStore identifier',
  `content_key` VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'Content object key',
  `metadata` JSON NOT NULL COMMENT 'Extended metadata',
  `modified_at` DATETIME NOT NULL COMMENT 'File modification time',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_digest_asset_key_digest_entry_revision` (`namespace_digest`, `asset_key_digest`, `entry_revision`),
  KEY `ix_namespace_digest_store_revision` (`namespace_digest`, `store_revision`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Asset change history';
