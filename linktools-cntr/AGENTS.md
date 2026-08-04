# AGENTS.md (linktools-cntr)

Architecture guidance for the container-management sub-package. Shared concerns
(monorepo structure, `manage.py`, config system, code style) live in the
[repo-root AGENTS.md](../AGENTS.md). User-facing usage is in
[README.md](README.md).

## Container Sub-package (`linktools-cntr/src/linktools/cntr/`)

Docker/Podman container lifecycle management (prefix `ct-cntr`). Built on the core
CLI framework.

- **`commands/`** — CLI surface: `root.py` (the `ct-cntr` command group + help-order
  contract), `config.py` (config get/set/unset/edit/reload), `repo.py` (external
  repo add/update/remove), `compose.py` (render the resolved Docker Compose model),
  `status.py` (live container state via `docker inspect`), `plan.py`/`exec_.py`.
- **`container.py` / `context.py`** — `ContainerManager` + per-container `Container`
  abstraction; resolves dependencies, renders Dockerfile/Compose templates, drives
  the lifecycle event hooks.
- **`lifecycle/`** — event dispatcher + hooks (`on_init`, `on_prepare`, `on_check`,
  `on_starting`/`on_started`, `on_stopping`/`on_stopped`, `on_removed`). `up` /
  `restart` (= down + up) / `down` run hooks in dependency order, then hand off to
  `docker compose`. See the sequence diagram in `README.md`.
- **`_container/`** — low-level Docker/Compose interaction: `compose.py` (Compose
  model render + `--check` validation), `template.py` (Jinja2 template engine for
  Dockerfile/Compose/env), `expose.py` (service-link extraction), `actions.py`.
- **`doctor.py`** — diagnostics: config sanity, requirement checks, optional runtime
  Compose validation (`--runtime`).
- **`artifacts.py`** — built-in container definitions (nginx, lldap, authelia,
  safeline, portainer).

## `.linktools.json` (project profile)

`.linktools.json` / `linktools.json` is a generic project manifest
(`linktools.core.ProjectProfile`), not a cntr-specific format. It layers user-level
(`~/.linktools/linktools.json`) and local (`<root>/.linktools.json`) files into the
existing ConfigResolver. A container repo can declare its `linktools-cntr` version
requirement under `requires` plus per-container default env values. cntr enforces
`requires.linktools-cntr` at `repo add` / `repo update` / load time (before the
repo's `container.py` is imported); other `requires` keys it ignores.

## No pessimistic coupling

cntr does not hardcode a runtime — it drives Docker/Podman via the configured
container engine and Compose. `requires` declaring `runtime:docker-engine` /
`runtime:docker-compose` versions actually gates `up` / `restart` / `compose`
(`down` / `status` / `doctor` are read-only and unaffected).
