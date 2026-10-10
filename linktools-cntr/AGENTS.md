# AGENTS.md (linktools-cntr)

Package instructions for `linktools-cntr`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply. User-facing usage lives in [README.md](README.md).

## Required Rules

- `linktools-cntr` targets Docker / Docker Compose only. Do not add Podman compatibility paths.
- Validate an external repository's declared requirements before importing or executing its Python code.
- Lifecycle operations must preserve dependency ordering and use one orchestration path; container hooks must not bypass that ordering or invoke a parallel Compose execution path.
- Planning and execution must share the same Docker/Compose command-construction semantics.

## Guidance

`linktools-cntr/src/linktools/cntr/` is currently organized around commands, manager/container/context, lifecycle hooks, Compose rendering, runtime process construction, diagnostics, and built-in artifacts.

`.linktools.json` / `linktools.json` currently uses the generic `linktools.core.ProjectProfile`. `requires.linktools-cntr` is checked before repo code is imported. Runtime requirements for `docker-engine` / `docker-compose` currently gate `up`, `restart`, and `compose`, while `down`, `status`, and `doctor` stay outside those version gates.

Project-profile values layer user `~/.linktools/linktools.json` and local `<root>/.linktools.json` through the core resolver.

- Integration declarations are command-local, lazy and side-effect free. Read
  resolved nginx sites through the shared snapshot; navigation never registers
  proxy configuration or mutates ACL/OIDC state.
- Keep native nginx business templates single-pass with explicit namespaces and
  context. Preserve application routing, capture and authentication behavior.
- Generated configuration must be validated before publication or explicit
  restart stops. Do not write derived files directly into an active generation,
  rotate persistent secrets, or treat a successful reload command as proof that
  a configuration was loaded.

See [docs/integrations.md](docs/integrations.md) for the integration protocol.
