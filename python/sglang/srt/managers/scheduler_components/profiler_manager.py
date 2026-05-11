from __future__ import annotations  # noqa: F401

from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import Any, Callable, List, Optional  # noqa: F401

from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.utils.profile_utils import ProfileManager  # noqa: F401


@dataclass(kw_only=True)
class SchedulerProfilerManager:
    """torch profiler / RPD / cuda profiler lifecycle. Composition target on
    Scheduler (``self.profiler_manager``). Owns 19 mutable runtime fields."""

    ps: Any
    dp_tp_cpu_group: Any
    get_forward_ct: Callable[[], int]

    def __post_init__(self) -> None:
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager = ProfileManager(
                ps=self.ps,
                cpu_group=self.dp_tp_cpu_group,
            )
        else:
            self.torch_profiler = None
            self.torch_profiler_output_dir: Optional[Path] = None
            self.profiler_activities: Optional[List[str]] = None
            self.profile_id: Optional[str] = None

            self.profiler_start_forward_ct: Optional[int] = None
            self.profiler_target_forward_ct: Optional[int] = None

            self.profiler_prefill_ct: Optional[int] = None
            self.profiler_decode_ct: Optional[int] = None
            self.profiler_target_prefill_ct: Optional[int] = None
            self.profiler_target_decode_ct: Optional[int] = None

            self.profile_by_stage: bool = False
            self.profile_in_progress: bool = False
            self.merge_profiles = False

            # For ROCM
            self.rpd_profiler = None
