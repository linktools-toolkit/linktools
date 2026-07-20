"""Stable action names used by authorization call sites."""

class SecurityAction:
    RUN_CANCEL = "run.cancel"
    RUN_CANCEL_SELF = "run.cancel:self"
    RUN_RESUME = "run.resume"
    APPROVAL_APPROVE = "approval.approve"
    APPROVAL_REJECT = "approval.reject"
    TASK_CANCEL = "task.cancel"
