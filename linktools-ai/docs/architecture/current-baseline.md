# Current Baseline (Phase 0)

> Frozen at branch point `bd07cce8` + Phase 0 baseline-repair commit.
> Captured: 2026-07-19. Python 3.13.5, SQLAlchemy 2.0.51, pydantic_ai 1.107.0.

## 1. Test baseline

The chosen baseline (`bd07cce8`) was **not green**: 10 deterministic failures.
Each was a real pre-existing bug (not environmental), repaired before the
refactor began so the refactor starts from a reproducible green baseline
(plan §5 阶段0 失败处理; AC-01). Run command:

```bash
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
```

Result after repair: **2006 passed, 0 failed** (was 1996 passed, 10 failed).

### Baseline repairs

| # | Bug | Fix |
|---|-----|-----|
| 1 | `TaskWorker._heartbeat` read `claim.task.lease_expires_at`; `TaskClaim` has no `.task`. The `AttributeError` killed the heartbeat on its first line, so lease renewal never ran and the renew-failure counter stayed 0. | Pass the just-claimed `task.lease_expires_at` from `_execute` (matches the documented "initialized to now + lease" intent). |
| 2 | The runtime task handler called `bind_runnable` unconditionally despite its comment stating it is best-effort for stores that lack it. | Guard with `getattr`; a store without `bind_runnable` skips drift protection as documented. |
| 3 | Storage commit tests synthesised an `approval_request` missing the 7-key execution binding the approval store enforces as a security invariant. | Storage-level fixtures carry a fixed binding (production fills it via the governed tool executor). |

## 2. Top-level package layout (current)

`linktools/ai/`: `_runtime/`, `agent/`, `artifact/`, `capability/`,
`evaluation/`, `events/`, `execution/`, `knowledge/`, `mcp/`, `memory/`,
`middleware/`, `model/`, `observability/`, `package/`, `policy/`, `prompt/`,
`providers/`, `registry/`, `run/`, `security/`, `session/`, `skill/`,
`storage/`, `subagent/`, `swarm/`, `task/`, `tool/`, plus `clock.py`,
`errors.py`, `runnable.py`.

## 3. Known structural debts to retire

These drive the refactor phases; see
`tests/ai/architecture/test_legacy_name_inventory.py` for the live ratchet.

- `storage/resource/` + `ResourceStore` back the public storage surface and
  `ArtifactStore`; the Asset/Artifact split (Phase 3) retires `Resource`.
- Top-level `providers/` (`ProviderBundle`) and `registry/` mix static spec
  collections with runtime wiring; Phase 4 collapses them into per-domain
  Catalogs + one `CapabilityProviderRegistry`.
- `package/` expresses runtime extension units (not Python distributions);
  Phase 8 renames it to `extension/`.
- `task/` carries the mature Job/Task/Attempt runtime; Phase 7 does a pure
  `task → jobs` semantic rename after the reliability snapshots are frozen.
- Identity value types live in `security/principal.py`; `task/models.py`
  imports them (`task → security` coupling). Phase 1 moves them into a
  standalone `identity/` package that depends on nothing.

## 4. Branches at branch point

- `master` @ `bd07cce8` — refactor branch point.
- `fix/ai-production-hardening` (1 commit ahead, `64347022`) — restructures the
  approval model further (`redacted_arguments`, per-field binding). **Not
  merged**; master holds an earlier enforcement shape. The refactor works from
  master; that branch's model changes are out of scope here.
- `fix/ai-task-reliability-closure` — now **behind** master (stale).

## 5. How to reproduce

```bash
git checkout bd07cce8              # + the Phase 0 repair commit
PYTHONPATH="linktools-ai/src:linktools/src" python -m pytest tests/ai/ -q
python -m pytest tests/ai/architecture/ -q   # dependency + legacy guards
```
