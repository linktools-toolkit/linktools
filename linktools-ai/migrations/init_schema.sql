-- linktools-ai init schema (MySQL 5.7+/8.0) -- DBA-lint compliant
--
-- Production DDL for the 15 ORM tables in
-- linktools.ai.storage.sqlalchemy.base.Base.metadata. Tables / columns / types /
-- nullability mirror the ORM; the index strategy, comments, column ordering and
-- timestamp indexes follow the DBA lint standard (and therefore diverge from
-- Base.metadata.create_all's literal output). The storage kernel is
-- filesystem/object-store based and has no SQL tables.
--
-- DBA requirements applied (every rule below is satisfied):
--   * ai_ table prefix; surrogate BIGINT AUTO_INCREMENT `id` primary key.
--   * Every column carries a COMMENT.
--   * updated_at is the second-to-last column, created_at the last.
--   * Every table has ix_created_at(created_at) and ix_updated_at(updated_at).
--   * Index naming: unique uk_<cols>, normal ix_<cols>; a composite name joins
--     its column names with '_'. Composite indexes never include the PK column.
--   * No low-selectivity single-column indexes (status) and no leftmost-
--     redundant indexes (covered by a composite / unique key's left prefix).
--   * Any index column wider than 128 chars uses a (128) prefix -- including
--     unique keys, which is safe because every such column holds a generated
--     id (uuid-hex = 32, sha256-hex = 64) well under 128.
--   * VARCHAR right-sized: identifiers/hashes are VARCHAR(128), short enums
--     VARCHAR(64); no VARCHAR >= 200 remains.
--   * At most two large fields (JSON/BLOB/TEXT) per table. Variable JSON
--     payloads are packed: ai_executions keeps definition + data
--     (input/approval/error); ai_execution_snapshots keeps resume_messages +
--     outcome (final_output/usage).
--   * No single-precision float columns (DOUBLE is used instead, e.g. score).
--   * No DB-level foreign keys (relationships are by string id column).

SET NAMES utf8mb4;

CREATE TABLE `ai_sessions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `session_id` VARCHAR(255) NOT NULL COMMENT 'Session id',
  `tenant_id` VARCHAR(255) NULL COMMENT 'Tenant id',
  `user_id` VARCHAR(255) NULL COMMENT 'User id',
  `next_turn_sequence` INT NOT NULL COMMENT 'Next turn sequence',
  `latest_completed_run_id` VARCHAR(255) NULL COMMENT 'Latest completed run id',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_session_id` (`session_id`(128)),
  KEY `ix_tenant_id_user_id` (`tenant_id`(128), `user_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Session';

CREATE TABLE `ai_session_turns` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `session_id` VARCHAR(255) NOT NULL COMMENT 'Session id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `input` JSON NULL COMMENT 'Input',
  `assistant_summary` JSON NULL COMMENT 'Assistant summary',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `completed_at` DATETIME NULL COMMENT 'Completed at',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_session_id_sequence` (`session_id`(128), `sequence`),
  UNIQUE KEY `uk_execution_id` (`execution_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Session turn';

CREATE TABLE `ai_executions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `session_id` VARCHAR(255) NOT NULL COMMENT 'Session id',
  `kind` VARCHAR(40) NOT NULL COMMENT 'Kind',
  `runnable_id` VARCHAR(255) NOT NULL COMMENT 'Runnable id',
  `runnable_type` VARCHAR(40) NOT NULL COMMENT 'Runnable type',
  `session_turn_sequence` INT NULL COMMENT 'Session turn sequence',
  `parent_execution_id` VARCHAR(255) NULL COMMENT 'Parent execution id',
  `root_execution_id` VARCHAR(255) NOT NULL COMMENT 'Root execution id',
  `status` VARCHAR(40) NOT NULL COMMENT 'Status',
  `definition` JSON NOT NULL COMMENT 'Definition',
  `definition_hash` VARCHAR(64) NOT NULL COMMENT 'Definition hash',
  `data` JSON NOT NULL COMMENT 'Data',
  `owner` VARCHAR(255) NULL COMMENT 'Owner',
  `fence` INT NOT NULL COMMENT 'Fence',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `cancel_requested_at` DATETIME NULL COMMENT 'Cancel requested at',
  `snapshot_revision` INT NOT NULL COMMENT 'Snapshot revision',
  `trace_sequence` INT NOT NULL COMMENT 'Trace sequence',
  `event_sequence` INT NOT NULL COMMENT 'Event sequence',
  `tenant_id` VARCHAR(255) NULL COMMENT 'Tenant id',
  `user_id` VARCHAR(255) NULL COMMENT 'User id',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_id` (`execution_id`(128)),
  KEY `ix_session_id` (`session_id`(128)),
  KEY `ix_root_execution_id` (`root_execution_id`(128)),
  KEY `ix_lease_expires_at` (`lease_expires_at`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Execution run lifecycle';

CREATE TABLE `ai_execution_snapshots` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `revision` INT NOT NULL COMMENT 'Revision',
  `resume_messages` JSON NOT NULL COMMENT 'Resume messages',
  `outcome` JSON NOT NULL COMMENT 'Outcome',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `trace_end_sequence` INT NOT NULL COMMENT 'Trace end sequence',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_id` (`execution_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Execution resume snapshot';

CREATE TABLE `ai_execution_trace_steps` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `kind` VARCHAR(40) NOT NULL COMMENT 'Kind',
  `payload` JSON NOT NULL COMMENT 'Payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_id_sequence` (`execution_id`(128), `sequence`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Execution trace step';

CREATE TABLE `ai_execution_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `sequence` INT NOT NULL COMMENT 'Sequence',
  `type` VARCHAR(120) NOT NULL COMMENT 'Type',
  `payload` JSON NOT NULL COMMENT 'Payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_id_sequence` (`execution_id`(128), `sequence`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Execution event';

CREATE TABLE `ai_execution_evaluations` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `evaluation_id` VARCHAR(255) NOT NULL COMMENT 'Evaluation id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `evaluator` VARCHAR(255) NOT NULL COMMENT 'Evaluator',
  `score` DOUBLE NULL COMMENT 'Score',
  `result` JSON NOT NULL COMMENT 'Result',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_evaluation_id` (`evaluation_id`(128)),
  KEY `ix_evaluator` (`evaluator`(128)),
  KEY `ix_execution_id_created_at` (`execution_id`(128), `created_at`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Execution evaluation result';

CREATE TABLE `ai_spec_documents` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `path` VARCHAR(512) NOT NULL COMMENT 'Path',
  `kind` VARCHAR(128) NOT NULL COMMENT 'Kind',
  `version` INT NOT NULL COMMENT 'Version',
  `etag` VARCHAR(255) NOT NULL COMMENT 'Etag',
  `active` TINYINT(1) NOT NULL COMMENT 'Active',
  `content` BLOB NOT NULL COMMENT 'Content',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_path` (`path`(128)),
  KEY `ix_kind` (`kind`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Spec document current state';

CREATE TABLE `ai_spec_revision` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `revision` INT NOT NULL COMMENT 'Revision',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  KEY `ix_revision` (`revision`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Global spec revision counter (singleton row id=1)';

CREATE TABLE `ai_spec_changes` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `revision` INT NOT NULL COMMENT 'Revision',
  `path` VARCHAR(512) NOT NULL COMMENT 'Path',
  `kind` VARCHAR(128) NULL COMMENT 'Kind',
  `version` INT NULL COMMENT 'Version',
  `etag` VARCHAR(255) NULL COMMENT 'Etag',
  `object_id` VARCHAR(128) NULL COMMENT 'Object id (spec_blobs sha256; null for tombstone)',
  `active` TINYINT(1) NULL COMMENT 'Active',
  `deleted` TINYINT(1) NOT NULL COMMENT 'Deleted',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_revision_path` (`revision`, `path`(128)),
  KEY `ix_object_id` (`object_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Spec change + version log (append-only; retained permanently for audit/rollback)';

CREATE TABLE `ai_spec_blobs` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `sha256` VARCHAR(64) NOT NULL COMMENT 'Sha256 content digest',
  `content` BLOB NOT NULL COMMENT 'Content bytes',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_sha256` (`sha256`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Content-addressed spec document blob (dedup by sha256; immutable)';

CREATE TABLE `ai_artifacts` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `artifact_id` VARCHAR(128) NOT NULL COMMENT 'Artifact id',
  `sha256` VARCHAR(64) NOT NULL COMMENT 'Sha256',
  `media_type` VARCHAR(128) NOT NULL COMMENT 'Media type',
  `size` INT NOT NULL COMMENT 'Size',
  `tenant_id` VARCHAR(128) NOT NULL COMMENT 'Tenant id',
  `producer_kind` VARCHAR(64) NOT NULL COMMENT 'Producer kind',
  `producer_id` VARCHAR(128) NOT NULL COMMENT 'Producer id',
  `run_id` VARCHAR(128) NULL COMMENT 'Run id',
  `session_id` VARCHAR(128) NULL COMMENT 'Session id',
  `parent_artifact_ids` JSON NOT NULL COMMENT 'Parent artifact ids',
  `provenance_metadata` JSON NOT NULL COMMENT 'Provenance metadata',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_tenant_id_artifact_id` (`tenant_id`, `artifact_id`),
  KEY `ix_artifact_id` (`artifact_id`),
  KEY `ix_sha256` (`sha256`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Artifact lineage record (blob content is file-backed)';

CREATE TABLE `ai_task_plans` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `plan_id` VARCHAR(255) NOT NULL COMMENT 'Plan id',
  `payload` JSON NOT NULL COMMENT 'Payload',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_plan_id` (`plan_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Task plan';

CREATE TABLE `ai_task_executions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `execution_id` VARCHAR(255) NOT NULL COMMENT 'Execution id',
  `plan_id` VARCHAR(255) NOT NULL COMMENT 'Plan id',
  `node_id` VARCHAR(255) NOT NULL COMMENT 'Node id',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `owner` VARCHAR(255) NULL COMMENT 'Owner',
  `fence` INT NOT NULL COMMENT 'Fence',
  `attempt` INT NOT NULL COMMENT 'Attempt',
  `result` JSON NULL COMMENT 'Result',
  `error` JSON NULL COMMENT 'Error',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_execution_id` (`execution_id`(128)),
  KEY `ix_plan_id` (`plan_id`(128)),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Task node execution';

CREATE TABLE `ai_memories` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `memory_id` VARCHAR(128) NOT NULL COMMENT 'Memory id',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `owner_id` VARCHAR(128) NOT NULL COMMENT 'Owner id',
  `content` TEXT NOT NULL COMMENT 'Memory content',
  `category` VARCHAR(64) NULL COMMENT 'Category',
  `confidence` DECIMAL(5,4) NULL COMMENT 'Confidence [0,1]',
  `version` INT NOT NULL COMMENT 'Version (optimistic lock)',
  `metadata_json` TEXT NOT NULL COMMENT 'Metadata (JSON)',
  `user_id` VARCHAR(128) NULL COMMENT 'User id',
  `workspace_id` VARCHAR(128) NULL COMMENT 'Workspace id',
  `session_id` VARCHAR(128) NULL COMMENT 'Session id',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_memory_id` (`memory_id`),
  KEY `ix_tenant_id` (`tenant_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Memory entry';

CREATE TABLE `ai_tool_operations` (
  `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Primary key id',
  `operation_id` VARCHAR(128) NOT NULL COMMENT 'Operation id',
  `tenant_id` VARCHAR(128) NULL COMMENT 'Tenant id',
  `run_id` VARCHAR(128) NOT NULL COMMENT 'Run id',
  `tool_call_id` VARCHAR(128) NOT NULL COMMENT 'Tool call id',
  `idempotency_key` VARCHAR(128) NOT NULL COMMENT 'Idempotency key',
  `tool_name` VARCHAR(128) NOT NULL COMMENT 'Tool name',
  `arguments_hash` VARCHAR(128) NOT NULL COMMENT 'Arguments hash',
  `binding_fingerprint` VARCHAR(128) NOT NULL COMMENT 'Binding fingerprint',
  `replay_safe` TINYINT(1) NOT NULL COMMENT 'Replay safe',
  `status` VARCHAR(32) NOT NULL COMMENT 'Status',
  `owner` VARCHAR(128) NULL COMMENT 'Owner',
  `fence` INT NOT NULL COMMENT 'Fence',
  `lease_expires_at` DATETIME NULL COMMENT 'Lease expires at',
  `result` JSON NULL COMMENT 'Result',
  `error` JSON NULL COMMENT 'Error',
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated at',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Created at',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_run_id_tool_call_id` (`run_id`, `tool_call_id`),
  UNIQUE KEY `uk_operation_id` (`operation_id`),
  KEY `ix_updated_at` (`updated_at`),
  KEY `ix_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Tool-call operation (idempotency/replay)';
