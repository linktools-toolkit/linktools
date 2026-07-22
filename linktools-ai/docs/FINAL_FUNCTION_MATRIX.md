# Final Function Matrix — `linktools-ai`

> Source of truth for the holistic closure (see
> `.docs/linktools-ai-holistic-closure-review-and-junior-remediation-guide.md`).
> Each row is a function the runtime must support; each column is a test
> dimension that must be covered. "Status" tracks the closure WPs that close
> the row. A row is not done until every set cell is backed by a test.

Status legend: **pending** (not yet implemented this closure), **partial**
(exists but incomplete/inert), **done** (implemented + tested this closure).

## 1. Capability surface

| Function | Normal | Exception | File | SQL | Concurrency | Failure-injection | Status |
|---|---|---|---|---|---|---|---|
| Agent run (Runtime.run) | y | y | y | y | – | – | partial (WP-02/06) |
| Agent run_stream (Runtime.run_stream) | y | y | y | y | – | y | partial (WP-16) |
| Approval pause (tool/model) | y | y | y | y | y | y | partial (WP-02/03) |
| Approval approve / reject | y | y | y | y | – | – | partial (WP-03) |
| Resume (Agent) | y | y | y | y | y | y | pending (WP-04) |
| Resume (Swarm) | y | y | y | y | y | y | pending (WP-05) |
| Cancel (Agent / Swarm) | y | y | y | y | – | – | partial |
| Session history (multi-turn) | y | y | y | y | y | – | partial (WP-06) |
| Checkpoint history | y | y | y | y | y | y | pending (WP-01) |
| Tool timeout / retry / idempotency | y | y | y | y | y | y | partial (WP-07) |
| Security pipeline tool hooks | y | y | y | y | – | y | partial |
| Security pipeline model hooks | y | y | y | y | – | y | pending (WP-13) |
| Model budget | y | y | y | y | – | – | pending (WP-14) |
| Swarm cost (max_total_cost) | y | y | y | y | – | – | pending (WP-14/15) |
| Swarm context policy | y | y | – | – | – | – | pending (WP-15) |
| Swarm middleware | y | y | – | – | – | – | pending (WP-15) |

## 2. MCP

| Function | Normal | Exception | File | SQL | Concurrency | Failure-injection | Status |
|---|---|---|---|---|---|---|---|
| MCP strict discovery | y | y | n/a | n/a | – | y | done |
| MCP best-effort discovery | y | y | n/a | n/a | – | – | done |
| MCP allowlist / denylist / prefix | y | y | n/a | n/a | – | – | done |
| MCP connection concurrency / cleanup | y | y | n/a | n/a | y | y | pending (WP-12) |
| MCP fingerprint (canonical) | y | y | n/a | n/a | – | – | pending (WP-12) |

## 3. Storage / Registry / Infra

| Function | Normal | Exception | File | SQL | Concurrency | Failure-injection | Status |
|---|---|---|---|---|---|---|---|
| File Storage | y | y | y | – | y | y | partial (WP-02) |
| SQLAlchemy Storage | y | y | – | y | y | y | partial (WP-02) |
| Asset Registry refresh | y | y | y | y | y | – | partial (WP-11) |
| Filesystem Registry refresh | y | y | y | n/a | y | – | partial (WP-11) |
| Registry null / presence parsing | y | y | n/a | n/a | – | – | done (prior closure) |
| Domain validation / deep-freeze | y | y | n/a | n/a | – | – | partial (WP-09) |
| Canonical JSON | y | y | n/a | n/a | – | – | pending (WP-08) |
| Optional dependency (SQLAlchemy) | y | – | – | – | – | – | done |

## 4. Public surface

| Function | Normal | Exception | Docs | Status |
|---|---|---|---|---|
| Public API (`linktools.ai`) | y | y | y | done |
| Runtime.inspect | y | y | y | done |
| CLI verifier | y | y | y | pending (WP-18) |
| README / architecture docs | – | – | y | pending (WP-18) |

## 5. Definitions

- **Normal** — happy-path test exists.
- **Exception** — the documented error/edge path is tested.
- **File / SQL** — the contract runs against the File / SQLAlchemy backend.
- **Concurrency** — a test that would fail without the lock/CAS/fence.
- **Failure-injection** — a test that simulates a crash/failure at a defined
  step and asserts only legal end states remain (no half-committed state).
