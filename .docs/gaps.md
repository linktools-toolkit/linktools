# v2 Spec Gaps — ALL RESOLVED

> Branch: `refactor/v2-stabilization` (from `core`).
> 32 commits ahead of `core`, 275 tests + 18 subtests passing. master and core unpolluted.

## All v2 §16 checklist items: DONE

- [x] Config main path → new ConfigResolver/Config (G1 DONE — all DSL migrated, _config.py deleted)
- [x] Config migration tool (inspect/backup/migrate/verify/rollback)
- [x] LoggingManager GlobalLogRecordFactoryManager (ref-counted)
- [x] CacheStore SQLite compat (ON CONFLICT fallback)
- [x] LockManager file_lock not on business files
- [x] Download Content-Range + 416 + hash-retry-once
- [x] Tool multi-version layout + manifest + active.json (G2 DONE)
- [x] Delete _create_tools global PATH mutation
- [x] SSH STRICT host-key default (v2 §11.3)
- [x] EventHandlerMixin deleted (G6 DONE)
- [x] Process composition + ProcessResult (G4 DONE)
- [x] All assert → ToolDefinitionError/CommandError (v2 §29 rule 10)
- [x] All bare except: → except Exception: (v2 §29 rule 11)
- [x] All scattered logging → LoggingManager (v2 §4.5)
- [x] Old _config.py deleted (ConfigDict inlined into _environ.py)

## No remaining gaps

All code-level migrations are complete. The only remaining step is live
end-to-end smoke testing (ct-cntr + mobile), which is an environment
verification task, not a code gap.


