# Final frozen PR #308 measurement

Measured localb7a6796117b40dd3821e1d3ae4184b1c5db428a3, tree4850ac69e24f691e89bce1e2e5bda23ba6fd46b4. Parent verified identical published tree at remote83d6a262f18e293080d5fb61c70f6783835e118c. This final candidate was actually rerun; values are not inherited from f731. Baseline is original PR basebd5e38ab9da0f8b5c487917693c812723ecfb058, not intermediate6e.

Strict comparison uses existing bd5 **entry-attribution** run and newly measured final entry run. It does not mix earlier original-v3 raw byte/page results with this sidecar. Script/input/helpers/Python/full dependencies are identical. Incidental generated timestamps/IDs can slightly change bytes/pages between otherwise equivalent runs.

Both small32 and large65536: runtime318 reads(311SELECT+7WITH) vs base304; cold44(42+2) vs46; warm41(39+2) vs43. Runtime DML254 vs463; commits124 vs144. Final runtime still+14, despite cold/warm improvements. Full lifecycle reads615 vs533, DML267 vs464, commits319 vs286; candidate performs2 offline-GC passes to reach sampled table fixed point versus base1, so lifecycle is not runtime-only.

## Full business-table storage, all columns including BLOB/archive

|Case|Base preGC/steady bytes|Final preGC/steady bytes|Steady reduction|Base/final pages|Base/final file bytes|
|---|---:|---:|---:|---:|---:|
|small|168888/168888|146968/143787|14.8625%|164/152|671744/622592|
|large|343417/343417|146812/143639|58.1736%|207/152|847872/622592|

Base dynamically provisioned19 business tables; candidate20. No table is omitted to manufacture savings. Byte metric sums length(CAST(column AS BLOB)) across all columns, not physical SQLite encoding; DB pages/files include table/index overhead. sqlite_sequence excluded from logical business total but included in full database file/pages. Peak snapshots are phase-sampled, not transaction-level continuous peaks. Final WAL/SHM are0. No VACUUM.

Cold14 API calls: result6, history26, tool execution-list/get/agent-run-list6 total, Native snapshot4, agent-run2 =44. Each API type called twice. History13 per invocation, all other call budgets unchanged from base.

Warm/cold/post-offline-GC results, history, Native snapshots, agent run and tool observations are equal within each run; exactly two file side effects and four model requests. Both cases exit0. No production edits, CI dispatch or publication performed by measurement worker.

Limits: SQLite/offline FunctionModel only; two turns with one tool call each; large text highly compressible. Does not measure multi-tool batch scaling, live provider latency, PostgreSQL/MySQL performance, all errors/attachments, concurrency or arbitrary payload distributions. SQL statement counts are not physical I/O or latency. Diagnostic stack overhead prohibits elapsed-time claims. Do not add historic27.7%/23.6% figures.

Raw baseline/final DBs, SQL traces and metadata retained in evidence package. Machine full results: final-b7-comparison.json.
