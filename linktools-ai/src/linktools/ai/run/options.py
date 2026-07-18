from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeCancellationOptions:
    grace_seconds: float = 30.0
    poll_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.grace_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError("cancellation intervals must be positive")
        if self.poll_seconds > self.grace_seconds:
            raise ValueError("poll_seconds must not exceed grace_seconds")
