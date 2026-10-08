#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container-provided consumers shared by planning and execution."""
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._base import IntegrationConsumer
from ..container import ContainerError

if TYPE_CHECKING:
    from typing import AbstractSet, Iterable, Mapping, Optional, Type
    from ..container import BaseContainer
    from ..manager import ContainerManager


def _get_consumer(container: "BaseContainer") -> "Optional[IntegrationConsumer]":
    consumer = container.integration_consumer
    if consumer is not None and (not isinstance(consumer, IntegrationConsumer) or consumer.container is not container):
        raise ContainerError("Invalid integration consumer in " + container.name)
    return consumer


def consumer_type(container: "BaseContainer") -> "Type[IntegrationConsumer]":
    consumer = _get_consumer(container)
    return type(consumer) if consumer is not None else IntegrationConsumer


def consumer_for_service(manager: "ContainerManager", service: str) -> "Type[IntegrationConsumer]":
    for consumer in manager.integration_consumers.values():
        if service in consumer.container.services:
            return type(consumer)
    return IntegrationConsumer


def create_consumers(manager: "ContainerManager") -> "Mapping[str, IntegrationConsumer]":
    result = {}
    for name in manager.integration_snapshot:
        container = manager.containers[name]
        consumer = _get_consumer(container)
        if consumer is not None:
            result[name] = consumer
    return MappingProxyType(result)


def create_generations(manager: "ContainerManager") -> "Mapping[str, IntegrationConsumer]":
    return MappingProxyType({name: consumer for name, consumer in manager.integration_consumers.items()
                             if consumer.generated})


def runtime_requirements(manager: "ContainerManager",
                         required: "AbstractSet[str]") -> "Mapping[str, set[str]]":
    result = {}
    for consumer in manager.integration_consumers.values():
        for provider, services in consumer.runtime_requirements(manager, required).items():
            result.setdefault(provider, set()).update(services)
    return result


def order_services(containers: "Iterable[BaseContainer]", services: "Iterable[str]") -> "tuple[str, ...]":
    priorities = {service: consumer_type(container).application_order
                  for container in containers for service in container.services}
    return tuple(sorted(services, key=priorities.__getitem__))
