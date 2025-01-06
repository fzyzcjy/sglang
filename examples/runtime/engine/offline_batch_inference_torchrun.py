import datetime
import os
import sys

from sglang import Engine
from sglang.srt.server_args import ServerFragmentArgs


def run():
    """
    Example command:

    ```
    torchrun --nproc_per_node=4 offline_batch_inference_torchrun.py
    ```
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(
        f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}'
    )

    tp_size = world_size
    tp_rank = rank
    _log(f"{tp_rank=} {tp_size=}")

    model_name, mem_fraction_static = "meta-llama/Llama-3.2-1B-Instruct", 0.1
    # model_name, mem_fraction_static = "meta-llama/Llama-3.1-70B-Instruct", 0.9 # test large models

    for k in [
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
        "RANK",
        "ROLE_NAME",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "WORLD_SIZE",
    ]:
        del os.environ[k]

    engine = Engine(
        model_path=model_name,
        mem_fraction_static=mem_fraction_static,
        tp_size=tp_size,
        fragment=ServerFragmentArgs(
            tp_rank=tp_rank,
            nccl_port=23456,
        ),
    )
    _log(f"{engine=}")

    for i in range(5):
        output = engine.generate(
            prompt="Today is a sunny day and I like",
            sampling_params=dict(max_new_tokens=16),
        )
        _log(f"Iteration {i}: {output=}")

    engine.shutdown()


if __name__ == "__main__":
    run()
