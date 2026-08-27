# AGENTS.md (linktools-cntr)

Package instructions for `linktools-cntr`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply. User-facing usage lives in [README.md](README.md).

## Required Rules

- `linktools-cntr` targets Docker / Docker Compose only. Do not add Podman compatibility paths.
- `.linktools.json` / `linktools.json` is the generic `linktools.core.ProjectProfile`, not a cntr-specific format.
- Enforce `requires.linktools-cntr` before an external repo's `container.py` is imported. `runtime:docker-engine` and `runtime:docker-compose` gate `up`, `restart`, and `compose`; `down`, `status`, and `doctor` stay outside those runtime-version gates.
- Keep lifecycle ordering in the lifecycle/manager layer. Container definitions contribute hooks but do not bypass dependency ordering or create a parallel Compose execution path.
- Docker/Compose command construction stays behind the runtime/process abstraction so planning and execution share the same command semantics.

## Guidance

`linktools-cntr/src/linktools/cntr/` is organized around:

- `commands/`: `ct-cntr` CLI surface.
- manager/container/context: dependency resolution and lifecycle orchestration.
- `lifecycle/`: lifecycle hooks and ordering.
- `_container/`: Compose rendering/templates/actions.
- `runtime/`: Docker/Compose process construction.
- `doctor.py`: diagnostics; `artifacts.py`: built-in containers.

Project-profile values layer user `~/.linktools/linktools.json` and local `<root>/.linktools.json` configuration through the core resolver.
