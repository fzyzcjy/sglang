from __future__ import annotations

from enum import Enum


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"
