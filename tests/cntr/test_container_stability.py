#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BaseContainer public surface: imports, identity, MRO, descriptors, caching
and subclass override dispatch. These must all keep holding after internal
implementation is split into ``_container/*`` modules.
"""
from linktools.cntr.container import (
    AbstractMetaClass,
    BaseContainer,
    ContainerError,
    ContainerTemplateError,
    ExposeCategory,
    ExposeLink,
    ExposeMixin,
    NginxMixin,
    SimpleContainer,
    SourceContainer,
)
from linktools.decorator import _CachedProperty


def test_public_names_importable_from_container_module():
    for cls in (
        AbstractMetaClass, BaseContainer, ContainerError, ContainerTemplateError,
        ExposeCategory, ExposeLink, ExposeMixin, NginxMixin,
        SimpleContainer, SourceContainer,
    ):
        assert cls is not None


def test_container_class_module_identity_unchanged():
    assert BaseContainer.__module__ == "linktools.cntr.container"
    assert SourceContainer.__module__ == "linktools.cntr.container"
    assert SimpleContainer.__module__ == "linktools.cntr.container"


def test_base_container_mro_unchanged():
    assert BaseContainer.__bases__ == (ExposeMixin, NginxMixin)
    assert type(BaseContainer) is AbstractMetaClass


_DESCRIPTOR_TYPES = {
    "description": _CachedProperty,
    "docker_compose": _CachedProperty,
    "docker_file": _CachedProperty,
    "services": _CachedProperty,
    "start_hooks": _CachedProperty,
    "stop_hooks": _CachedProperty,
    "_rendered_hook_keys": _CachedProperty,
}


def test_descriptors_remain_on_base_container():
    for name, expected_type in _DESCRIPTOR_TYPES.items():
        descriptor = BaseContainer.__dict__.get(name)
        assert isinstance(descriptor, expected_type), f"{name} moved off BaseContainer"


def _pick_container(fresh_manager):
    # Any installed builtin container works; nginx is always present.
    return fresh_manager.containers["nginx"]


def test_cached_properties_store_on_the_container_instance(fresh_manager):
    container = _pick_container(fresh_manager)
    for name in ("docker_compose", "docker_file", "services", "start_hooks", "stop_hooks", "_rendered_hook_keys"):
        first = getattr(container, name)
        second = getattr(container, name)
        assert first is second
        assert container.__dict__[name] is first


def test_subcommand_methods_stay_in_base_container_dict():
    for name in (
        "on_exec_up", "on_exec_restart", "on_exec_down", "on_exec_config",
        "on_exec_shell", "on_exec_logs", "on_mount", "on_unmount_file",
    ):
        assert name in BaseContainer.__dict__, f"{name} moved off BaseContainer"


def test_configs_override_is_used_by_manager_prepare(fresh_manager, tmp_path):
    calls = []

    class _Custom(BaseContainer):
        @property
        def configs(self):
            calls.append(1)
            return super().configs

    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    fresh_manager.env_config.update_defaults(**container.configs)
    assert calls == [1]


def test_get_service_name_override_is_used(fresh_manager, tmp_path):
    class _Custom(BaseContainer):
        def get_service_name(self, key):
            return f"custom-{key}"

    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    assert container.get_service_name("web") == "custom-web"


def test_get_source_path_override_is_used_by_docker_compose_lookup(fresh_manager, tmp_path):
    calls = []

    class _Custom(BaseContainer):
        def get_source_path(self, *paths):
            calls.append(paths)
            return super().get_source_path(*paths)

    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    assert container.docker_compose is None
    assert calls  # get_source_path was consulted for each docker_compose_names candidate


def test_docker_context_and_file_path_overrides_are_used_by_docker_compose(fresh_manager, tmp_path):
    calls = []

    class _Custom(BaseContainer):
        def get_docker_context_path(self):
            calls.append("context")
            return super().get_docker_context_path()

        def get_docker_file_path(self):
            calls.append("file")
            return tmp_path / "Dockerfile"

    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n    restart: 'no'\n")

    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    compose = container.docker_compose
    assert compose is not None
    assert "context" in calls
    assert "file" in calls


def test_render_template_override_is_used_by_docker_file(fresh_manager, tmp_path):
    calls = []

    class _Custom(BaseContainer):
        def render_template(self, source, destination=None, **kwargs):
            calls.append(source)
            return "rendered"

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    assert container.docker_file == "rendered"
    assert calls


def test_source_container_context_path_uses_overridden_source_properties(fresh_manager, tmp_path):
    class _Custom(SourceContainer):
        @property
        def _source_url(self):
            return "https://example.invalid/archive.zip"

        @property
        def _source_path(self):
            return "unpacked"

        def _handle_source_file(self, source, destination):
            pass

    container = _Custom(fresh_manager, tmp_path, name="999-custom")
    assert container.get_docker_context_path() == container._context_path
    assert container._context_path.endswith("unpacked")
