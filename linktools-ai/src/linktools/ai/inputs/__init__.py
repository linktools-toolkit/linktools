#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted input, immutable captures, and replay context shared by producers."""

from ._input import (
    InputFileSource,
    encode_user_content,
    captured_input_files,
    captured_input_prompt,
    CanonicalUserInput,
    ExecutionInputMaterializer,
    InputIntent,
    decode_task_prompt_draft,
    decode_user_content_payload,
    stored_input_attachment_views,
    input_intent,
    merge_captured_input_files,
    stored_user_input_view,
    task_prompt_draft,
    validate_user_input,
)
from ._input_capture import (
    AgentInputCapture,
    CaptureInputRequest,
    CaptureGraphRequest,
    ExecutionInputCaptureRef,
    InputCaptures,
    InputCaptureExecutionPort,
)
from ._context_archive import (
    ExecutionContextArchive,
    PreparedExecutionContext,
)
from ._agent_task_input import (
    AgentTaskContextError,
    AgentTaskInput,
    AgentTaskInputBuilder,
    AgentTaskInputContext,
)
from ._task_graph_binding_capture import (
    builtin_task_declaration,
    TaskGraphBindingCapture,
    TaskGraphBindingCaptureStore,
    task_declaration_semantics,
)
from ..model import (
    UserPromptInput,
)

__all__ = ['AgentInputCapture', 'AgentTaskContextError', 'AgentTaskInput', 'AgentTaskInputBuilder', 'AgentTaskInputContext', 'CanonicalUserInput', 'CaptureGraphRequest', 'CaptureInputRequest', 'ExecutionContextArchive', 'ExecutionInputCaptureRef', 'ExecutionInputMaterializer', 'InputCaptureExecutionPort', 'InputCaptures', 'InputFileSource', 'InputIntent', 'PreparedExecutionContext', 'TaskGraphBindingCapture', 'TaskGraphBindingCaptureStore', 'UserPromptInput', 'builtin_task_declaration', 'captured_input_files', 'captured_input_prompt', 'decode_task_prompt_draft', 'decode_user_content_payload', 'encode_user_content', 'input_intent', 'merge_captured_input_files', 'stored_input_attachment_views', 'stored_user_input_view', 'task_declaration_semantics', 'task_prompt_draft', 'validate_user_input']
