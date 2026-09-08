#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install Task v2 attachment integration with instance-owned coordination."""

import base64
import binascii
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from pydantic_ai.messages import BinaryContent, UserContent

import linktools.ai.runtime._planner as planner_runtime
import linktools.ai.runtime._runtime_service as runtime_service
import linktools.ai.runtime._task_attachment as task_attachment
import linktools.ai.runtime.state._codec as codec_runtime

from ..errors import AIError, ErrorCode
from ..task import DefaultTaskService, TaskGraph, TaskGraphLaunch, TaskNode
from ._attachment import DefaultAttachmentService
from ._input import _decode_user_content, managed_user_prompt_draft, task_prompt_draft
from .state import RuntimeDomain
from .state._attachment_codec import _entry
from .state._codec import _decode_domain
from .state._contracts import ExecutionStartReservation
from .state._repositories import ExecutionRepositoryImpl

_installed = False
_original_runtime_init: Any = None
_original_admit_graph: Any = None
_original_task_for_agent: Any = None
_original_arm_graph: Any = None
_original_runner_handler: Any = None
_original_agent_normalize: Any = None
_original_agent_validate_recovery: Any = None
_original_agent_prepare_request: Any = None
_original_agent_run_node: Any = None
_original_agent_cancel_node: Any = None
_original_reserve_start: Any = None
_original_iter_object_refs: Any = None


def _decode_task_prompt(body: Mapping[str, Any]) -> tuple[Any, tuple[str, ...]]:
    prompt = body.get("prompt")
    attachments = body.get("attachments")
    if not isinstance(attachments, list) or any(
        not isinstance(path, str) or not path for path in attachments
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if isinstance(prompt, Mapping):
        if (
            set(prompt) != {"kind", "text"}
            or prompt.get("kind") != "text"
            or not isinstance(prompt.get("text"), str)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cast(str, prompt["text"]), tuple(attachments)
    if not isinstance(prompt, list) or not prompt:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    items: list[UserContent] = []
    for raw in prompt:
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        kind = raw.get("kind")
        if kind == "text":
            if set(raw) != {"kind", "text"} or not isinstance(raw.get("text"), str):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            items.append(cast(str, raw["text"]))
            continue
        if kind == "binary":
            if set(raw) != {
                "kind",
                "data_b64",
                "media_type",
                "identifier",
                "vendor_metadata",
            }:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            data_b64 = raw.get("data_b64")
            media_type = raw.get("media_type")
            identifier = raw.get("identifier")
            metadata = raw.get("vendor_metadata")
            if (
                not isinstance(data_b64, str)
                or not isinstance(media_type, str)
                or not media_type
                or identifier is not None and not isinstance(identifier, str)
                or metadata is not None and not isinstance(metadata, Mapping)
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                data = base64.b64decode(data_b64, validate=True)
            except (ValueError, binascii.Error) as error:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            if not data or base64.b64encode(data).decode("ascii") != data_b64:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            items.append(
                BinaryContent(
                    data,
                    media_type=media_type,
                    identifier=cast(str | None, identifier),
                    vendor_metadata=(
                        None if metadata is None else cast(dict[str, Any], dict(metadata))
                    ),
                )
            )
            continue
        if kind == "native":
            if (
                set(raw) != {"kind", "codec", "value"}
                or raw.get("codec") != "pydantic-user-content-v1"
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            value = raw.get("value")
            if not isinstance(value, Mapping):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            decoded = _decode_user_content(dict(value))
            if len(decoded) != 1:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            items.append(decoded[0])
            continue
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return tuple(items), tuple(attachments)


async def _runtime_admit_graph(
    self: runtime_service.Runtime,
    graph: TaskGraph,
    *,
    principal: Any,
    idempotency_key: str,
    limits: Any,
    correlation: Any,
):
    coordinator = self.__dict__.get("_task_attachment_coordinator")
    if coordinator is None or not any(
        task_attachment._is_v2_draft(node) for node in graph.nodes
    ):
        return await _original_admit_graph(
            self,
            graph,
            principal=principal,
            idempotency_key=idempotency_key,
            limits=limits,
            correlation=correlation,
        )
    resolved = self._resolve_principal(principal)
    prepared = await coordinator.prepare_graph(
        graph,
        principal=resolved,
        idempotency_key=idempotency_key,
    )
    return await _original_admit_graph(
        self,
        prepared,
        principal=resolved,
        idempotency_key=idempotency_key,
        limits=limits,
        correlation=correlation,
    )


def _runtime_task_for_agent(self: runtime_service.Runtime, *args: Any, **kwargs: Any):
    node = _original_task_for_agent(self, *args, **kwargs)
    user_prompt = args[2] if len(args) > 2 else kwargs.get("user_prompt")
    attachments = kwargs.get("attachments", ())
    if attachments or user_prompt is None:
        return node
    try:
        draft = managed_user_prompt_draft(user_prompt)
    except TypeError:
        return node
    if draft is None:
        return node
    value = node.input
    if (
        not isinstance(value, Mapping)
        or value.get("type") != task_attachment._AGENT_TASK_TYPE
        or value.get("version") != 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    required = {"binding", "mode", "planning", "thinking"}
    if not required.issubset(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return replace(
        node,
        input={
            "type": task_attachment._AGENT_TASK_TYPE,
            "version": 2,
            "stage": "draft",
            "binding": value["binding"],
            "prompt": task_prompt_draft(draft),
            "attachments": [],
            "mode": value["mode"],
            "planning": value["planning"],
            "thinking": value["thinking"],
        },
    )


async def _task_arm_graph(
    self: DefaultTaskService,
    launch: TaskGraphLaunch,
) -> None:
    coordinator = self.__dict__.get("_task_attachment_coordinator")
    if coordinator is not None:
        await coordinator.confirm_launch(launch)
    await _original_arm_graph(self, launch)


def _runtime_init(self: runtime_service.Runtime, *args: Any, **kwargs: Any) -> None:
    _original_runtime_init(self, *args, **kwargs)
    attachment_service = self.attachments
    task_runtime = self._task_node_runtime
    if not isinstance(attachment_service, DefaultAttachmentService) or not isinstance(
        task_runtime,
        planner_runtime.RuntimeTaskNodeRunner,
    ):
        return
    coordinator = task_attachment._TaskAttachmentCoordinator(
        self.task,
        task_runtime,
        attachment_service,
        self.workspace,
    )
    self.__dict__["_task_attachment_coordinator"] = coordinator
    self.task.__dict__["_task_attachment_coordinator"] = coordinator


def _runner_handler(
    self: planner_runtime.RuntimeTaskNodeRunner,
    task_type: str,
    task_version: int,
):
    if task_type == task_attachment._AGENT_TASK_TYPE and task_version == 2:
        return self._agent
    return _original_runner_handler(self, task_type, task_version)


def _agent_normalize(self: Any, body: Mapping[str, Any]) -> dict[str, Any]:
    if body.get("stage") is None:
        return _original_agent_normalize(self, body)
    if body.get("stage") != "prepared":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return task_attachment._normalized_prepared_body(body)


def _agent_validate_recovery(
    self: Any,
    body: Mapping[str, Any],
    *,
    graph_id: str,
    node_id: str,
) -> dict[str, Any]:
    if body.get("stage") is None:
        return _original_agent_validate_recovery(
            self,
            body,
            graph_id=graph_id,
            node_id=node_id,
        )
    del graph_id, node_id
    return task_attachment._normalized_prepared_body(body)


def _agent_prepare_request(self: Any, node: Any, **kwargs: Any):
    if not isinstance(node, TaskNode) or not task_attachment._is_v2_prepared(node):
        return _original_agent_prepare_request(self, node, **kwargs)
    binding, request, _prepared, _target = task_attachment._derived_execution_prepared(
        self,
        node,
        graph_id=kwargs["graph_id"],
        principal=kwargs["principal"],
        dependencies=kwargs["dependencies"],
    )
    return binding, replace(request, correlation=kwargs["correlation"])


async def _agent_run_node(self: Any, node: Any, **kwargs: Any):
    if not isinstance(node, TaskNode) or not task_attachment._is_v2_prepared(node):
        return await _original_agent_run_node(self, node, **kwargs)
    _binding, request, prepared, target = task_attachment._derived_execution_prepared(
        self,
        node,
        graph_id=kwargs["graph_id"],
        principal=kwargs["principal"],
        dependencies=kwargs["dependencies"],
    )
    with task_attachment.admission_scope(
        task_attachment.ManagedAdmission(
            "execution.run",
            request.idempotency_key,
            prepared,
            source_target=target,
        )
    ):
        return await _original_agent_run_node(self, node, **kwargs)


async def _agent_cancel_node(self: Any, node: Any, **kwargs: Any) -> None:
    if not isinstance(node, TaskNode) or not task_attachment._is_v2_prepared(node):
        await _original_agent_cancel_node(self, node, **kwargs)
        return
    _binding, request, prepared, target = task_attachment._derived_execution_prepared(
        self,
        node,
        graph_id=kwargs["graph_id"],
        principal=kwargs["principal"],
        dependencies=kwargs["dependencies"],
    )
    with task_attachment.admission_scope(
        task_attachment.ManagedAdmission(
            "execution.run",
            request.idempotency_key,
            prepared,
            source_target=target,
        )
    ):
        await _original_agent_cancel_node(self, node, **kwargs)


async def _reserve_start(
    self: ExecutionRepositoryImpl,
    reservation: ExecutionStartReservation,
):
    return await task_attachment._reserve_start_from_adopted_source(self, reservation)


def _iter_object_refs(value: object, domain: RuntimeDomain, codec: Any):
    if isinstance(value, Mapping) and value.get("$dataclass") == "task_node":
        fields = value.get("fields")
        if isinstance(fields, Mapping):
            input_value = fields.get("input")
            try:
                decoded = _decode_domain(input_value, Any, codec, persisted=True)
            except AIError:
                decoded = None
            if (
                isinstance(decoded, Mapping)
                and decoded.get("type") == task_attachment._AGENT_TASK_TYPE
                and decoded.get("version") == 2
                and decoded.get("stage") == "prepared"
            ):
                manifest = decoded.get("attachment_manifest")
                if isinstance(manifest, list):
                    for raw in manifest:
                        entry = _entry(raw)
                        try:
                            yield RuntimeDomain(entry.content.domain), entry.content.object
                        except ValueError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    yield from _original_iter_object_refs(value, domain, codec)


def install_task_attachments() -> None:
    """Install Task v2 attachment integration exactly once."""
    global _installed
    global _original_runtime_init
    global _original_admit_graph
    global _original_task_for_agent
    global _original_arm_graph
    global _original_runner_handler
    global _original_agent_normalize
    global _original_agent_validate_recovery
    global _original_agent_prepare_request
    global _original_agent_run_node
    global _original_agent_cancel_node
    global _original_reserve_start
    global _original_iter_object_refs
    if _installed:
        return

    _original_runtime_init = runtime_service.Runtime.__init__
    _original_admit_graph = runtime_service.Runtime._admit_graph
    _original_task_for_agent = runtime_service.Runtime._task_for_agent
    _original_arm_graph = DefaultTaskService._arm_graph
    _original_runner_handler = planner_runtime.RuntimeTaskNodeRunner._handler
    _original_agent_normalize = planner_runtime._AgentTaskNodeHandler.normalize
    _original_agent_validate_recovery = planner_runtime._AgentTaskNodeHandler.validate_recovery
    _original_agent_prepare_request = planner_runtime._AgentTaskNodeHandler._prepare_request
    _original_agent_run_node = planner_runtime._AgentTaskNodeHandler.run_node
    _original_agent_cancel_node = planner_runtime._AgentTaskNodeHandler.cancel_node
    _original_reserve_start = ExecutionRepositoryImpl.reserve_start
    _original_iter_object_refs = codec_runtime._iter_runtime_object_refs

    task_attachment._decode_draft_prompt = _decode_task_prompt
    task_attachment._original_reserve_start = _original_reserve_start
    task_attachment._original_runtime_init = _original_runtime_init
    task_attachment._original_admit_graph = _original_admit_graph
    task_attachment._original_arm_graph = _original_arm_graph
    task_attachment._original_runner_handler = _original_runner_handler
    task_attachment._original_agent_normalize = _original_agent_normalize
    task_attachment._original_agent_validate_recovery = _original_agent_validate_recovery
    task_attachment._original_agent_prepare_request = _original_agent_prepare_request
    task_attachment._original_agent_run_node = _original_agent_run_node
    task_attachment._original_agent_cancel_node = _original_agent_cancel_node

    runtime_service.Runtime.__init__ = _runtime_init
    runtime_service.Runtime._admit_graph = _runtime_admit_graph
    runtime_service.Runtime._task_for_agent = _runtime_task_for_agent
    DefaultTaskService._arm_graph = _task_arm_graph
    planner_runtime.RuntimeTaskNodeRunner._handler = _runner_handler
    planner_runtime._AgentTaskNodeHandler.normalize = _agent_normalize
    planner_runtime._AgentTaskNodeHandler.validate_recovery = _agent_validate_recovery
    planner_runtime._AgentTaskNodeHandler._prepare_request = _agent_prepare_request
    planner_runtime._AgentTaskNodeHandler.run_node = _agent_run_node
    planner_runtime._AgentTaskNodeHandler.cancel_node = _agent_cancel_node
    ExecutionRepositoryImpl.reserve_start = _reserve_start
    codec_runtime._iter_runtime_object_refs = _iter_object_refs
    _installed = True


__all__ = ["install_task_attachments"]