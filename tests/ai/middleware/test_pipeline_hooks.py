#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

from linktools.ai.middleware.base import Middleware
from linktools.ai.middleware.pipeline import MiddlewarePipeline


class _Recorder(Middleware):
    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self.log = log

    async def before_model(self, context, request):
        self.log.append(f"{self.name}.before_model:{request}")
        return f"{request}->bm{self.name}"

    async def after_model(self, context, response):
        self.log.append(f"{self.name}.after_model:{response}")
        return f"{response}->am{self.name}"

    async def before_tool(self, context, request):
        self.log.append(f"{self.name}.before_tool:{request}")
        return f"{request}->bt{self.name}"

    async def after_tool(self, context, request, result):
        self.log.append(f"{self.name}.after_tool:{result}")
        return f"{result}->at{self.name}"


def _pipeline():
    log: "list[str]" = []
    pipeline = MiddlewarePipeline(middlewares=(
        _Recorder("M1", log), _Recorder("M2", log), _Recorder("M3", log),
    ))
    return pipeline, log


def test_run_before_model_threads_request_in_registration_order():
    pipeline, log = _pipeline()
    async def _run():
        return await pipeline.run_before_model(context=None, request="req")
    out = asyncio.run(_run())
    assert log == ["M1.before_model:req", "M2.before_model:req->bmM1", "M3.before_model:req->bmM1->bmM2"]
    assert out == "req->bmM1->bmM2->bmM3"


def test_run_after_model_threads_response_in_reverse_order():
    pipeline, log = _pipeline()
    async def _run():
        return await pipeline.run_after_model(context=None, response="resp")
    out = asyncio.run(_run())
    assert log == ["M3.after_model:resp", "M2.after_model:resp->amM3", "M1.after_model:resp->amM3->amM2"]
    assert out == "resp->amM3->amM2->amM1"


def test_run_before_tool_threads_in_registration_order():
    pipeline, log = _pipeline()
    async def _run():
        return await pipeline.run_before_tool(context=None, request="toolreq")
    out = asyncio.run(_run())
    assert log == ["M1.before_tool:toolreq", "M2.before_tool:toolreq->btM1", "M3.before_tool:toolreq->btM1->btM2"]
    assert out == "toolreq->btM1->btM2->btM3"


def test_run_after_tool_threads_result_in_reverse_order_and_passes_request_through():
    pipeline, log = _pipeline()
    async def _run():
        return await pipeline.run_after_tool(context=None, request="toolreq", result="res")
    out = asyncio.run(_run())
    assert log == ["M3.after_tool:res", "M2.after_tool:res->atM3", "M1.after_tool:res->atM3->atM2"]
    assert out == "res->atM3->atM2->atM1"


def test_run_after_tool_passes_same_request_to_every_middleware():
    seen_requests: "list[str]" = []
    class _ReqCapture(Middleware):
        async def after_tool(self, context, request, result):
            seen_requests.append(request)
            return result
    pipeline = MiddlewarePipeline(middlewares=(_ReqCapture(), _ReqCapture(), _ReqCapture()))
    async def _run():
        return await pipeline.run_after_tool(context=None, request="the-request", result="r")
    asyncio.run(_run())
    assert seen_requests == ["the-request", "the-request", "the-request"]


def test_empty_pipeline_returns_inputs_unchanged():
    pipeline = MiddlewarePipeline(middlewares=())
    async def _run():
        bm = await pipeline.run_before_model(context=None, request="r1")
        am = await pipeline.run_after_model(context=None, response="r2")
        bt = await pipeline.run_before_tool(context=None, request="r3")
        at = await pipeline.run_after_tool(context=None, request="r3", result="r4")
        return (bm, am, bt, at)
    out = asyncio.run(_run())
    assert out == ("r1", "r2", "r3", "r4")
