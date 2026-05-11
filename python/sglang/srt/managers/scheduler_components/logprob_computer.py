from __future__ import annotations


class SchedulerLogprobComputer:
    """Pure-compute logprob accumulator helpers. Composition target on
    Scheduler (``self.logprob_computer``)."""

    def __init__(self, *, server_args, model_config) -> None:
        self.server_args = server_args
        self.model_config = model_config
