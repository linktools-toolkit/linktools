# AGENTS.md

Repository-wide instructions for coding agents.

`Required Rules` are mandatory. `Guidance` is informational. When changing a package, read this file and that package's `AGENTS.md`; package rules may add stricter constraints but do not weaken repository rules.

## Required Rules

### Repository workflow

- Use `manage.py` as the repository entry point for install, check, build, verify, clean, and release gates. Do not add another task runner for the same responsibilities.
- Repository build/release helpers live under root `scripts/`; package directories must not add their own release-script trees.
- `check` is read-only; `verify` validates built artifacts. Do not publish or tag after either gate fails.

### Python

- `linktools-ai` requires Python >= 3.10; all other packages support Python >= 3.6.
- Every `.py` file starts with the standard shebang and UTF-8 header.
- Public functions and methods have parameter and return annotations. Use `Any` only for genuinely untyped values.
- In Python 3.6-compatible packages, quote annotations containing `|` or `[...]`. Annotation-only imports belong under `TYPE_CHECKING`; runtime-required names do not.

### Architecture

- Use public APIs across module/package boundaries. Do not access private state through reflection or another module's private members. Add a public operation when required and implement it across affected backends/Protocol implementations.
- Test code is exempt from the public-only boundary rule: tests may import or access private modules and members when needed to verify internal behavior, regression invariants, or implementation-specific failure modes. This repository-wide exception applies unless a package `AGENTS.md` explicitly opts tests out; production/runtime code is never exempt, and tests of public contracts should still prefer public APIs.
- Keep files, classes, and packages cohesive. Behavior that belongs to a class stays on the class; prefer `classmethod` over `staticmethod` when class ownership matters.
- `_`-prefixed modules are private and re-exported through the package surface. Externally addressed modules (entry points, dotted paths, directory scanning) remain public.
- Runtime dependencies must remain acyclic at module and package level; `TYPE_CHECKING`-only references are allowed.

### Logging and comments

- Use `environ.get_logger(...)` or `environ.logger`; direct `logging.*` is reserved for the core logging manager and CLI root-logger setup.
- Log meaningful state transitions and decisions, not noise. Guard only expensive debug formatting with `if environ.debug:`.
- Comments explain real intent or constraints that naming cannot express. Do not add review/process/history narration, issue/PR references, boilerplate, or code restatement. Fix comments made stale by the current change, but avoid unrelated cleanup.

### Git commits

- Commit subjects use `type(scope)`, for example `fix(ai)`, `refactor(core)`, `feat(mobile)`, `docs(common)`, or `test(cntr)`.
- Keep subjects short and describe the actual change; avoid process-oriented subjects such as `cleanup`, `final fixes`, or `address review comments`.

## Guidance

### Packages

| Package | Responsibility | Commands |
| --- | --- | --- |
| `linktools/` | Core framework: CLI, environ, config, tool management | base |
| `linktools-common/` | Common tools | `ct-*` |
| `linktools-mobile/` | Android/iOS/Frida tools | `at-*`, `it-*` |
| `linktools-cntr/` | Docker / Docker Compose management | `ct-cntr` |
| `linktools-ai/` | AI agent runtime | - |

Each package lives under `{name}/src/linktools/`. Package entry points and capabilities are declared under `scripts:` in its `linktools.yml`. Project-specific repository gates currently live under `scripts/check/<project>/` when needed.

### Common commands

```bash
python manage.py install --editable
python manage.py check [package...]
python manage.py build [package...]
python manage.py verify [package...]
python manage.py clean [package...]
```
