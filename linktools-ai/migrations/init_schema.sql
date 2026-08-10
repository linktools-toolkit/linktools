-- linktools-ai complete schema reference (MySQL 5.7+/8.0) --
--
-- This file is a DBA review artifact, not a production bootstrap or migration
-- command. Production databases must be created and changed only through
-- reviewed, environment-specific DBA migration tooling.
--
-- This reference contains every current SQL table, grouped by its owner:
--   * Runtime tables owned by adapter.sql;
--   * Harness StepStore tables owned by adapter._step;
--   * Asset current entries, change history, content blobs, and revision
--     counter tables owned by asset.sql.
-- Each owner retains an independent runtime initialization boundary. Do not
-- register this complete reference as one metadata set in application code.
--
-- Reference DBA rules:
--   * Every table uses the ai_ prefix and every table/column carries a COMMENT.
--   * Every table uses a BIGINT AUTO_INCREMENT surrogate id, puts updated_at
--     immediately before created_at, and has both ix_updated_at and
--     ix_created_at timestamp indexes.
--   * Index and unique-key names are logical, unprefixed identifiers. Identity
--     keys use uk_namespace_key_tenant_id_record_id; timestamp keys use
--     ix_updated_at and ix_created_at. Special keys retain the exact logical
--     names declared by their columns, including uk_namespace, uk_digest,
--     uk_namespace_asset_key_hash, uk_namespace_key_tenant_id_digest,
--     uk_namespace_key_tenant_id_chunk_key,
--     uk_namespace_key_tenant_id_call_key, and
--     uk_namespace_key_tenant_id_identity_key.
--     Runtime execution lookups use
--     ix_namespace_key_tenant_id_parent_execution_id and
--     ix_namespace_key_tenant_id_session_id; Step run identity uses
--     uk_namespace_key_run_key.
--   * No composite index or unique key contains more than three physical
--     columns. Where a logical identity has more fields, its canonical
--     combination is stored in a utf8mb4 CHAR(64) identity column and indexed
--     with namespace/tenant scope.
--   * namespace_key and every digest/hash column use utf8mb4_bin, never ASCII.
--     Sessions session_id, tool run_id/tool_call_id, and Step run
--     conversation_id are NOT NULL wherever their identity requires them.
--     Step event/snapshot run_id indexes use a (128) prefix.
--   * Indexed columns wider than 128 characters use a (128) prefix.
--   * No low-selectivity status index, redundant left-prefix index,
--     floating-point column, or database-level foreign key is introduced.
--   * JSON and LONGBLOB payloads are limited to fields required by each contract.

SET NAMES utf8mb4;

CREATE TABLE `ai_runtime_approvals` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime approval records';

CREATE TABLE `ai_runtime_artifacts` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime artifact records';

CREATE TABLE `ai_runtime_blob_chunks` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content digest',
  `chunk_index` BIGINT DEFAULT 0 NOT NULL COMMENT 'Blob chunk index',
  `chunk_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Blob chunk identity digest',
  `content` LONGBLOB NOT NULL COMMENT 'Blob chunk content',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  UNIQUE KEY `uk_namespace_key_tenant_id_chunk_key` (`namespace_key`, `tenant_id`, `chunk_key`),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime content blob chunks';

CREATE TABLE `ai_runtime_blobs` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content digest',
  `size` BIGINT DEFAULT 0 NOT NULL COMMENT 'Content size in bytes',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  UNIQUE KEY `uk_namespace_key_tenant_id_digest` (`namespace_key`, `tenant_id`, `digest`),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime content blob records';

CREATE TABLE `ai_runtime_evaluations` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime evaluation records';

CREATE TABLE `ai_runtime_events` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime execution events';

CREATE TABLE `ai_runtime_executions` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `agent_run_sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Agent run sequence',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_namespace_key_tenant_id_base_execution_id` (`namespace_key`, `tenant_id`, `base_execution_id`(128)),
  KEY `ix_namespace_key_tenant_id_parent_execution_id` (`namespace_key`, `tenant_id`, `parent_execution_id`(128)),
  KEY `ix_namespace_key_tenant_id_session_id` (`namespace_key`, `tenant_id`, `session_id`(128)),
  KEY `ix_namespace_key_tenant_id_source_execution_id` (`namespace_key`, `tenant_id`, `source_execution_id`(128))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime execution records';

CREATE TABLE `ai_runtime_externals` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime external result records';

CREATE TABLE `ai_runtime_idempotency` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `scope` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Idempotency scope',
  `key_hash` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Idempotency key digest',
  `identity_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Idempotency identity digest',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  UNIQUE KEY `uk_namespace_key_tenant_id_identity_key` (`namespace_key`, `tenant_id`, `identity_key`),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime idempotency records';

CREATE TABLE `ai_runtime_memories` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime memory records';

CREATE TABLE `ai_runtime_operation_counters` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `resource_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Operation resource kind',
  `resource_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Operation resource identifier',
  `partition_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Operation partition identity digest',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  UNIQUE KEY `uk_namespace_key_tenant_id_partition_key` (`namespace_key`, `tenant_id`, `partition_key`),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime operation counters';

CREATE TABLE `ai_runtime_operations` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `resource_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Operation resource kind',
  `resource_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Operation resource identifier',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime operation ledger';

CREATE TABLE `ai_runtime_results` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime result records';

CREATE TABLE `ai_runtime_sessions` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `profile` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Session profile',
  `head_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Current session head execution',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_session_id` (`namespace_key`, `tenant_id`, `session_id`(128)),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime session records';

CREATE TABLE `ai_runtime_task_nodes` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `owner` VARCHAR(512) NULL COMMENT 'Lease owner',
  `fence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Lease fence',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expiration',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime task node records';

CREATE TABLE `ai_runtime_tasks` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime task graph records';

CREATE TABLE `ai_runtime_tools` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Namespace digest',
  `tenant_id` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tenant identifier',
  `record_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Domain record identifier',
  `session_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Session identifier',
  `parent_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent execution identifier',
  `source_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Source execution identifier',
  `base_execution_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Base execution identifier',
  `lineage_kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'RUN' NOT NULL COMMENT 'Execution lineage kind',
  `sequence` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record sequence',
  `revision` BIGINT DEFAULT 0 NOT NULL COMMENT 'Record revision',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Record status',
  `payload` JSON NOT NULL COMMENT 'Canonical record payload',
  `run_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool run identifier',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool call identifier',
  `call_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT '' NOT NULL COMMENT 'Tool call identity digest',
  `owner` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Lease owner',
  `fence` BIGINT DEFAULT 0 NULL COMMENT 'Lease fence',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expiration',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_tenant_id_call_key` (`namespace_key`, `tenant_id`, `call_key`),
  UNIQUE KEY `uk_namespace_key_tenant_id_record_id` (`namespace_key`, `tenant_id`, `record_id`(128)),
  KEY `ix_created_at` (`created_at`),
  KEY `ix_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Runtime tool operation records';

CREATE TABLE `ai_step_runs` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'StepStore namespace digest',
  `run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Agent run identifier',
  `conversation_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Agent name',
  `metadata_json` JSON NOT NULL COMMENT 'Run metadata',
  `started_at` DATETIME NOT NULL COMMENT 'Run start timestamp',
  `run_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Run identity digest',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_run_id` (`namespace_key`, `run_id`(128)),
  UNIQUE KEY `uk_namespace_key_run_key` (`namespace_key`, `run_key`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Harness StepStore run records';

CREATE TABLE `ai_step_events` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'StepStore namespace digest',
  `run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Agent run identifier',
  `kind` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Event kind',
  `step_index` INT NOT NULL COMMENT 'Step index',
  `timestamp` DATETIME NOT NULL COMMENT 'Event timestamp',
  `conversation_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Agent name',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Tool call identifier',
  `tool_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Tool name',
  `error` TEXT NULL COMMENT 'Event error details',
  `metadata_json` JSON NOT NULL COMMENT 'Event metadata',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  KEY `ix_namespace_key_run_id` (`namespace_key`, `run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Harness StepStore event records';

CREATE TABLE `ai_step_snapshots` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'StepStore namespace digest',
  `run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Agent run identifier',
  `step_index` INT NOT NULL COMMENT 'Step index',
  `conversation_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Conversation identifier',
  `parent_run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Parent run identifier',
  `agent_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Agent name',
  `timestamp` DATETIME NOT NULL COMMENT 'Snapshot timestamp',
  `state` VARCHAR(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT 'complete' NOT NULL COMMENT 'Snapshot state',
  `messages_json` JSON NOT NULL COMMENT 'Serialized model messages',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  KEY `ix_namespace_key_run_id` (`namespace_key`, `run_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Harness StepStore snapshots';

CREATE TABLE `ai_step_effects` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'StepStore namespace digest',
  `run_id` VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Agent run identifier',
  `tool_call_id` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool call identifier',
  `tool_name` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Tool name',
  `status` VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Effect status',
  `started_at` DATETIME NOT NULL COMMENT 'Effect start timestamp',
  `ended_at` DATETIME NULL COMMENT 'Effect end timestamp',
  `idempotency_key` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Idempotency key',
  `effect_summary` TEXT NULL COMMENT 'Effect summary',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_run_id_tool_call_id` (`namespace_key`, `run_id`(128), `tool_call_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Harness StepStore tool effects';

CREATE TABLE `ai_step_media` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace_key` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'StepStore namespace digest',
  `sha256` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Media content digest',
  `media_type` VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Media MIME type',
  `bytes` LONGBLOB NOT NULL COMMENT 'Media content',
  `size_bytes` BIGINT NOT NULL COMMENT 'Media content size',
  `metadata_json` JSON NOT NULL COMMENT 'Media metadata',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_key_sha256` (`namespace_key`, `sha256`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Harness StepStore media objects';

CREATE TABLE `ai_asset_entries` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset namespace',
  `asset_key_hash` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset key digest',
  `asset_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset kind',
  `asset_id` VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset file identifier',
  `entry_revision` BIGINT NOT NULL COMMENT 'File entry revision',
  `store_revision` BIGINT NOT NULL COMMENT 'Store revision at file update',
  `etag` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'File content digest',
  `size` BIGINT NOT NULL COMMENT 'File content size',
  `status` VARCHAR(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Current file status',
  `blob_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Content blob digest',
  `modified_at` DATETIME NOT NULL COMMENT 'File modification timestamp',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace_asset_key_hash` (`namespace`, `asset_key_hash`),
  KEY `ix_namespace_store_revision` (`namespace`, `store_revision`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Current Asset file entries';

CREATE TABLE `ai_asset_changes` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset namespace',
  `asset_key_hash` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset key digest',
  `asset_kind` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset kind',
  `asset_id` VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset file identifier',
  `entry_revision` BIGINT NOT NULL COMMENT 'File entry revision',
  `store_revision` BIGINT NOT NULL COMMENT 'Store revision at file update',
  `etag` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'File content digest',
  `size` BIGINT NOT NULL COMMENT 'File content size',
  `status` VARCHAR(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'File status at this history revision',
  `blob_digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL COMMENT 'Content blob digest',
  `modified_at` DATETIME NOT NULL COMMENT 'File modification timestamp',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  KEY `ix_namespace_asset_key_hash_entry_revision` (`namespace`, `asset_key_hash`, `entry_revision`),
  KEY `ix_namespace_store_revision` (`namespace`, `store_revision`),
  KEY `ix_blob_digest` (`blob_digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Asset file change history';

CREATE TABLE `ai_asset_blobs` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `digest` CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Content digest',
  `content` LONGBLOB NOT NULL COMMENT 'File content',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_digest` (`digest`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Content-addressed Asset file blobs';

CREATE TABLE `ai_asset_revision` (
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT 'Surrogate primary key',
  `namespace` VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'Asset namespace',
  `store_revision` BIGINT NOT NULL COMMENT 'Asset store revision',
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL COMMENT 'Last update timestamp',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT 'Creation timestamp',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_namespace` (`namespace`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT='Asset store revision counters';
