import json
import os
import random
import threading
import time
from typing import Any, Dict, Optional

_temp_log_file: Optional[object] = None
_temp_log_lock = threading.Lock()
_temp_log_initialized = False


def _get_pod_identifier() -> str:
    hostname = os.environ.get("HOSTNAME", "")
    if hostname:
        return hostname
    pod_name = os.environ.get("POD_NAME", "")
    if pod_name:
        return pod_name
    return f"pid_{os.getpid()}"


def _get_torch_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _lazy_init():
    global _temp_log_file, _temp_log_initialized
    if _temp_log_initialized:
        return
    with _temp_log_lock:
        if _temp_log_initialized:
            return
        log_name = os.environ.get("SGLANG_TEMP_LOG_NAME", "")
        if not log_name:
            _temp_log_initialized = True
            print("temp_log skip since no env var")
            return
        log_dir = f"/shared_data/{log_name}"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pod_id = _get_pod_identifier()
        torch_rank = _get_torch_rank()
        rand_int = random.randint(0, 999999)
        filename = f"{timestamp}_{pod_id}_rank{torch_rank}_{rand_int}.log"
        filepath = os.path.join(log_dir, filename)
        print(f"temp_log will write to: {filepath}")
        _temp_log_file = open(filepath, "a", encoding="utf-8")
        _temp_log_initialized = True


def temp_log(a_dict: Dict[str, Any]):
    _lazy_init()
    if _temp_log_file is None:
        return
    a_dict["_ts"] = time.time()
    line = json.dumps(a_dict, ensure_ascii=False) + "\n"
    with _temp_log_lock:
        _temp_log_file.write(line)
        _temp_log_file.flush()

