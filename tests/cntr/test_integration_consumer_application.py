#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Running integration consumers apply only their changed generated inputs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr import OperationContext
from linktools.cntr.artifacts import AppliedServiceModels, bind_prepared_files
from test_lifecycle_rebuild import setup_case


def consumer_case(tmp_path, running=True, changed=True, historical=False):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("consumer", {"consumer": {"image": "consumer:new", "environment": {"PENDING": "new"}}}),
    ], running=("consumer",) if running else ())
    owner = manager.containers["consumer"]
    owner.runtime = SimpleNamespace(chmod=lambda path, mode: path.chmod(mode))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="consumer"),)
    root = owner.get_app_path("generated")
    manager.model["services"]["consumer"]["volumes"] = [
        {"type": "bind", "source": str(root / "current/config.yml"), "target": "/config.yml", "read_only": True}]
    previous = deepcopy(manager.model)
    previous["services"]["consumer"].update(image="consumer:old", environment={"PENDING": "old"})
    context = OperationContext(project_containers=list(manager.containers.values()), target_services=("consumer",))
    context.write_files(owner, {"config.yml": "old"})
    previous = bind_prepared_files(context, previous)
    if running:
        AppliedServiceModels(manager, previous).record(("consumer",))
    if historical:
        manager.integration_snapshot["app"] = ()
    def integrate(ctx):
        manager.events.append(("integrate", "consumer"))
        ctx.write_files(owner, {"config.yml": "new" if changed else "old"})
    owner.on_integrating = integrate
    owner.on_starting = lambda ctx: pytest.fail("collateral consumer initialization")
    old_image_id = manager.image_preparer.image_id
    manager.image_preparer.image_id = lambda image: image if image.startswith("sha256:") else old_image_id(image)
    manager.applied_models = []
    apply = manager.compose_runner.apply_services
    def capture(ctx, services):
        manager.applied_models.append(deepcopy(ctx.compose_model))
        apply(ctx, services)
    manager.compose_runner.apply_services = capture
    return manager


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("action", ["up", "restart"])
def test_changed_running_consumer_is_applied_without_initialization_or_image_refresh(tmp_path, historical, action):
    manager = consumer_case(tmp_path, historical=historical)
    getattr(manager.compose_operations, action)(["app"], pull=True)
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
    spec = manager.applied_models[-1]["services"]["consumer"]
    assert spec["environment"] == {"PENDING": "old"}
    assert spec["image"] == "sha256:old-consumer"
    assert ("check-callback", "consumer") in manager.events
    assert ("after-callback", "consumer") in manager.events
    assert all("consumer" not in event[1] for event in manager.events if event[0] == "stop")


@pytest.mark.parametrize("running,changed", [(False, True), (True, False)])
def test_stopped_or_unchanged_consumer_is_not_applied_or_notified(tmp_path, running, changed):
    manager = consumer_case(tmp_path, running=running, changed=changed)
    manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert ("check-callback", "consumer") not in manager.events
    assert ("after-callback", "consumer") not in manager.events
    assert (("integrate", "consumer") in manager.events) is running


def test_missing_consumer_baseline_requires_explicit_selection_before_mutation(tmp_path):
    from linktools.cntr.errors import ContainerError
    manager = consumer_case(tmp_path)
    for path in (tmp_path / 'compose/applied/services').glob('*.yml'):
        path.unlink()
    with pytest.raises(ContainerError, match='explicit'):
        manager.compose_operations.up(['app'])
    assert not any(event[0] in ('images', 'apply', 'stop') for event in manager.events)


def test_failed_consumer_integration_does_not_publish_or_apply(tmp_path):
    from linktools.cntr.errors import ContainerError
    manager = consumer_case(tmp_path)
    previous = dict(AppliedServiceModels(manager, manager.model).previous)
    def fail(context):
        raise ContainerError('invalid consumer input')
    manager.containers['consumer'].on_integrating = fail
    with pytest.raises(ContainerError, match='invalid consumer input'):
        manager.compose_operations.restart(['app'])
    assert not any(event[0] in ('images', 'apply', 'stop') for event in manager.events)
    assert dict(AppliedServiceModels(manager, manager.model).previous) == previous


def test_consumer_partial_failure_does_not_record_unapplied_generation(tmp_path):
    from linktools.cntr.errors import ContainerError
    manager = consumer_case(tmp_path)
    before = AppliedServiceModels(manager, manager.model).previous['consumer']
    manager.compose_runner.fail = 'consumer'
    with pytest.raises(ContainerError, match='apply failed consumer'):
        manager.compose_operations.up(['app'])
    assert AppliedServiceModels(manager, manager.model).previous['consumer'] == before
    assert ('after-callback', 'consumer') not in manager.events


@pytest.mark.parametrize("changed", [False, True])
def test_pending_consumer_support_only_blocks_a_real_generated_change(tmp_path, changed):
    from linktools.cntr.errors import ContainerError
    manager = consumer_case(tmp_path, changed=changed)
    previous = AppliedServiceModels(manager, manager.model).previous_model('consumer')
    previous['networks'] = {'default': {'name': 'applied-network'}}
    AppliedServiceModels(manager, previous).record(('consumer',))
    manager.model['networks'] = {'default': {'name': 'pending-network'}}
    if changed:
        with pytest.raises(ContainerError, match='conflicts.*consumer'):
            manager.compose_operations.up(['app'])
        assert not any(event[0] in ('images', 'apply', 'stop') for event in manager.events)
    else:
        manager.compose_operations.up(['app'])
        assert [event[1] for event in manager.events if event[0] == 'apply'] == ['app']


def test_pending_consumer_dependency_cycle_is_not_applied_or_used_for_ordering(tmp_path):
    manager = consumer_case(tmp_path)
    manager.model['services']['consumer']['depends_on'] = {'consumer': {'condition': 'service_started'}}
    manager.compose_operations.up(['app'])
    assert 'depends_on' not in manager.applied_models[-1]['services']['consumer']
    assert {event[1] for event in manager.events if event[0] == 'apply'} == {'app', 'consumer'}


def test_changed_consumer_keeps_dependency_order_for_all_application_callbacks(tmp_path):
    manager = consumer_case(tmp_path)
    manager.model['services']['app']['depends_on'] = {'consumer': {'condition': 'service_started'}}
    manager.compose_operations.up(['app'])
    for phase in ('before-start', 'check-callback'):
        assert [event[1] for event in manager.events if event[0] == phase and event[1] != 'manager'] == ['consumer', 'app']
    assert [event[1] for event in manager.events if event[0] == 'after-callback'] == ['app', 'consumer']


def test_pending_consumer_generated_mount_is_not_validated_or_applied(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers['consumer']
    manager.model['services']['consumer']['volumes'].append({
        'type': 'bind', 'source': str(owner.get_app_path('generated/current/pending.yml')), 'target': '/pending.yml'})
    manager.compose_operations.up(['app'])
    mounts = manager.applied_models[-1]['services']['consumer']['volumes']
    assert [item['target'] for item in mounts] == ['/config.yml']


@pytest.mark.parametrize('other_state', ['exited', 'running'])
def test_collateral_image_comes_only_from_unambiguous_running_replicas(tmp_path, other_state):
    from linktools.cntr.errors import ContainerError
    from linktools.cntr.runtime.inspect import ProjectRuntimeState
    manager = consumer_case(tmp_path)
    original = manager.docker_inspector.get_project_state
    actual = original(None)
    extra = SimpleNamespace(service='consumer', state=other_state, image_id='sha256:other', labels={})
    mixed = ProjectRuntimeState('test', actual.services + (extra,), 'docker')
    manager.docker_inspector.get_project_state = lambda containers, **kwargs: mixed
    if other_state == 'running':
        with pytest.raises(ContainerError, match='one running image'):
            manager.compose_operations.up(['app'])
        assert not any(event[0] in ('images', 'apply', 'stop') for event in manager.events)
    else:
        manager.compose_operations.up(['app'])
        assert manager.applied_models[-1]['services']['consumer']['image'] == 'sha256:old-consumer'


def test_partial_authelia_generations_never_validate_an_unselected_pending_image(tmp_path):
    from types import MethodType
    from linktools.cntr.errors import ContainerError
    from test_builtin_generated_config import builtin_container_type
    manager = setup_case(tmp_path, [
        ('app', {'app': {'image': 'app:new'}}),
        ('authelia', {'authelia': {'image': 'authelia:pending'},
                      'authelia-admin': {'image': 'admin:pending', 'depends_on': ['authelia']}}),
    ], running=('authelia', 'authelia-admin'))
    owner = manager.containers['authelia']
    owner.runtime = SimpleNamespace(chmod=lambda path, mode: path.chmod(mode))
    manager.integration_snapshot['app'] = (SimpleNamespace(consumer='authelia'),)
    for service in owner.services:
        manager.model['services'][service]['volumes'] = [{
            'type': 'bind', 'source': str(owner.get_app_path('generated/current/configuration.yml')),
            'target': '/generated/configuration.yml'}]
    for service, content in (('authelia', 'new'), ('authelia-admin', 'old')):
        context = OperationContext(project_containers=list(manager.containers.values()), target_services=(service,))
        context.write_files(owner, {'configuration.yml': content})
        previous = deepcopy(manager.model)
        previous['services'][service]['image'] = service + ':old'
        AppliedServiceModels(manager, bind_prepared_files(context, previous)).record((service,))
    owner.on_integrating = lambda ctx: ctx.write_files(owner, {'configuration.yml': 'new'})
    builtin = builtin_container_type('102-authelia')
    owner._config_services, owner.config_files = builtin._config_services, builtin.config_files
    owner.get_config = lambda key: True
    owner.on_check = MethodType(builtin.on_check, owner)
    manager.compose_runner.validate_service = lambda *args, **kwargs: pytest.fail('unselected validator image')
    old_image_id = manager.image_preparer.image_id
    manager.image_preparer.image_id = lambda image: image if image.startswith('sha256:') else old_image_id(image)
    with pytest.raises(ContainerError, match='explicitly select the authelia'):
        manager.compose_operations.up(['app'])
    assert not any(event[0] in ('apply', 'stop') for event in manager.events)


@pytest.mark.parametrize('changed', [False, True])
def test_shared_contributor_requirement_applies_only_to_changed_consumer(tmp_path, changed):
    from linktools.cntr.errors import ContainerError
    from test_lifecycle_rebuild import Owner
    manager = consumer_case(tmp_path, changed=changed)
    manager.containers['contributor'] = Owner('contributor', manager.containers['consumer'].services, manager)
    manager.integration_snapshot['contributor'] = ()
    manager.load_installed_config_metadata = lambda: list(manager.containers.values())
    native_model = manager.compose_runner.final_model
    manager.compose_runner.final_model = lambda context, **kwargs: native_model(context)
    if changed:
        with pytest.raises(ContainerError, match='Shared integration consumer contributors.*contributor'):
            manager.compose_operations.up(['app'])
        assert not any(event[0] in ('images', 'apply', 'stop') for event in manager.events)
    else:
        manager.compose_operations.up(['app'])
        assert [event[1] for event in manager.events if event[0] == 'apply'] == ['app']


def test_consumer_only_network_preserves_applied_definition_without_false_conflict(tmp_path):
    manager = consumer_case(tmp_path)
    previous = AppliedServiceModels(manager, manager.model).previous_model('consumer')
    previous['services']['consumer']['networks'] = {'private': {}}
    previous['networks'] = {'private': {'name': 'applied-private'}}
    AppliedServiceModels(manager, previous).record(('consumer',))
    manager.model['services']['consumer']['networks'] = {'private': {}}
    manager.model['networks'] = {'private': {'name': 'pending-private'}}
    manager.compose_operations.up(['app'])
    assert manager.applied_models[-1]['networks']['private'] == {'name': 'applied-private'}
    assert {event[1] for event in manager.events if event[0] == 'apply'} == {'app', 'consumer'}


def test_consumer_snapshot_can_predate_new_explicit_producer(tmp_path):
    manager = consumer_case(tmp_path)
    previous = AppliedServiceModels(manager, manager.model).previous_model('consumer')
    previous['services'].pop('app')
    AppliedServiceModels(manager, previous).record(('consumer',))
    manager.compose_operations.up(['app'])
    assert {event[1] for event in manager.events if event[0] == 'apply'} == {'app', 'consumer'}
