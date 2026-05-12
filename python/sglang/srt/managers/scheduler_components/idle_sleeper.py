"""``IdleSleeper`` — zmq Poller wrapper with empty-cache throttling.

Reduces system power consumption during long inactive periods.
"""

from dataclasses import dataclass, field
from typing import Any

import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.utils import empty_device_cache


@dataclass(kw_only=True, slots=True)
class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    sockets: Any
    last_empty_time: float = field(default_factory=real_time)

    def __post_init__(self) -> None:
        self.poller = zmq.Poller()
        for s in self.sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            empty_device_cache()
