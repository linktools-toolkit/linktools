from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from linktools.ai.budget.hook import BudgetCapability
from linktools.ai.budget.tracker import BudgetExceededError, BudgetTracker


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("ok")],
        usage=RequestUsage(input_tokens=1000, output_tokens=1000),
    )


def test_accrues_cost_after_each_model_call():
    tracker = BudgetTracker(budget_usd=10.0, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.03)
    agent = Agent(FunctionModel(_model_fn), capabilities=[BudgetCapability(tracker)])
    agent.run_sync("hi")
    assert tracker.spent_usd > 0.0


def test_raises_before_the_call_that_would_exceed_budget():
    tracker = BudgetTracker(budget_usd=0.001, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.03)
    agent = Agent(FunctionModel(_model_fn), capabilities=[BudgetCapability(tracker)])
    # First call succeeds (tracker starts at 0, check() passes), accruing cost past budget.
    agent.run_sync("hi")
    assert tracker.spent_usd > tracker.budget_usd
    # A second top-level run against the same tracker must be blocked before any model call.
    try:
        agent.run_sync("hi again")
        assert False, "expected BudgetExceededError"
    except BudgetExceededError:
        pass


def test_capability_satisfies_protocol():
    assert isinstance(BudgetCapability(BudgetTracker(budget_usd=1.0)), AbstractCapability)
