-- New Step admission owner; ai_step_runs unchanged. 14 columns, 4 physical indexes including PRIMARY.
-- Requires schema review. Dialect-conditional portable indexes are not emitted for MySQL.

CREATE TABLE ai_step_run_admissions (
	id BIGINT NOT NULL COMMENT 'Surrogate row identifier used only by the SQL backend.' AUTO_INCREMENT, 
	store_key VARBINARY(128) NOT NULL COMMENT 'Exact namespace, tenant and domain partition admitting this run.', 
	`key` VARBINARY(768) NOT NULL COMMENT 'Canonical local run admission identity, independent of the registration owner.', 
	scope_key VARBINARY(768) COMMENT 'Conversation lookup identity derived from the validated domain projection.', 
	parent_key VARBINARY(768) COMMENT 'Parent-run lookup identity derived from the validated domain projection.', 
	sort_key LONGBLOB NOT NULL COMMENT 'Stable ordering by projected start time and complete admission identity.', 
	storage_revision BIGINT NOT NULL COMMENT 'Optimistic local publication and release fence, independent of canonical registration.', 
	payload_json JSON NOT NULL COMMENT 'Typed canonical registration reference and actual domain projection differences; no copied run body.', 
	key_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT 'SHA-256 lookup projection of the admission key; complete identity must also match.', 
	scope_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'SHA-256 lookup projection of the projected conversation scope.', 
	parent_locator CHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin COMMENT 'SHA-256 lookup projection of the projected parent-run identity.', 
	sort_prefix VARBINARY(128) NOT NULL COMMENT 'First 128 sort-key bytes used only for ordered lookup.', 
	updated_at DATETIME NOT NULL COMMENT 'Update timestamp' DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, 
	created_at DATETIME NOT NULL COMMENT 'Creation timestamp' DEFAULT CURRENT_TIMESTAMP, 
	PRIMARY KEY (id)
)CHARSET=utf8mb4 ROW_FORMAT=DYNAMIC COMMENT='Step-owned domain run admission; canonical run metadata remains with its unique registration owner.' ENGINE=InnoDB COLLATE utf8mb4_bin

;
CREATE INDEX ix_created_at ON ai_step_run_admissions (created_at);
CREATE INDEX ix_updated_at ON ai_step_run_admissions (updated_at);
CREATE UNIQUE INDEX uk_store_key_key_locator ON ai_step_run_admissions (store_key, key_locator);
