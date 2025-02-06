import multiprocessing
import multiprocessing as mp
import os
import traceback
import unittest
from multiprocessing import Process

import torch
from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server.engine_fragment import EngineFragment
from sglang.srt.server_args import find_available_port
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.runners import check_close_model_outputs
from sglang.test.test_utils import is_in_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoModelForCausalLM

_TP_SIZE = 2
_MAX_NEW_TOKENS = 8
_PROMPTS = ["Today is a sunny day and I like", "I have a very good idea on"]

# Set to false to temporarily debug issues unrelated to weight update
_ENABLE_UPDATE_WEIGHTS = True
# _ENABLE_UPDATE_WEIGHTS = False

# TODO maybe we should add more other models? should we keep it in sync with test_generation_models.py?
CI_MODELS = [
    dict(model_path="meta-llama/Llama-3.1-8B-Instruct"),
    dict(model_path="google/gemma-2-2b"),
]
ALL_OTHER_MODELS = [
    dict(model_path="meta-llama/Llama-3.2-1B-Instruct"),
    dict(model_path="Qwen/Qwen2-1.5B"),
    dict(model_path="Qwen/Qwen2.5-14B-Instruct"),
    dict(model_path="HuggingFaceTB/SmolLM-135M-Instruct"),
    dict(model_path="allenai/OLMo-1B-0724-hf"),
    dict(model_path="THUDM/glm-4-9b-chat"),
    dict(model_path="openai-community/gpt2"),
    dict(model_path="microsoft/Phi-3-small-8k-instruct"),
    dict(model_path="allenai/OLMo-2-1124-7B-Instruct"),
    dict(model_path="ibm-granite/granite-3.0-2b-instruct"),
]


class TestFragment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn")

    def assert_fragment_e2e_execution(self, index: int, model_path: str):
        nccl_port = find_available_port(12345)
        master_port = find_available_port(23456)

        print(f'assert_fragment_e2e_execution START {index=} {model_path=}')

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, master_port, nccl_port, output_writer, model_path),
            )
            p.start()
            processes.append(p)

        for _ in range(_TP_SIZE):
            self.assertTrue(output_reader.recv(),
                            f'Subprocess has error, please see logs above. ({index=} {model_path=})')

        for p in processes:
            p.join()

    def test_ci_models(self):
        for index, model_info in enumerate(CI_MODELS):
            self.assert_fragment_e2e_execution(index=index, **model_info)

    def test_others(self):
        if is_in_ci():
            return

        for index, model_info in enumerate(ALL_OTHER_MODELS):
            self.assert_fragment_e2e_execution(index=index, **model_info)

    # def test_adhoc(self):
    #     self.assert_fragment_e2e_execution(index=0, model_path="meta-llama/Llama-3.2-1B-Instruct")


def _run_subprocess(tp_rank: int, master_port: int, nccl_port: int, output_writer, model_path: str):
    try:
        print(f"subprocess[{tp_rank=}] Start {os.environ['CUDA_VISIBLE_DEVICES']=}")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(rank=tp_rank, world_size=_TP_SIZE)

        mesh_kwargs = dict(mesh_shape=(_TP_SIZE, 1), mesh_dim_names=["tp", "pp"])
        inference_device_mesh_device = init_device_mesh("cuda", **mesh_kwargs)
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)
        print(
            f"subprocess[{tp_rank=}] {inference_device_mesh_device=} {inference_device_mesh_cpu=}"
        )

        fragment = EngineFragment(
            model_path=model_path,
            load_format='dummy' if _ENABLE_UPDATE_WEIGHTS else 'auto',
            mem_fraction_static=0.4,
            tp_size=_TP_SIZE,
            random_seed=42,
            trust_remote_code=True,
            # fragment args
            tp_rank=tp_rank,
            gpu_id=tp_rank,
            nccl_port=nccl_port,
            parallel_process_groups=ParallelProcessGroups.from_devices_meshes(
                device_mesh_device=inference_device_mesh_device,
                device_mesh_cpu=inference_device_mesh_cpu,
                dim_tp="tp",
                dim_pp="pp",
            ),
        )
        print(f"subprocess[{tp_rank=}] {fragment=}", flush=True)

        # hf model is used for comparison
        with torch.device("cuda"):
            hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            hf_model.to(torch.bfloat16)
        hf_model.cuda()
        hf_tokenizer = get_tokenizer(model_path, trust_remote_code=True)

        hf_outputs = HFRunner.forward_generation_raw(
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            base_model=hf_model,
            tokenizer=hf_tokenizer,
            lora_paths=None,
            torch_dtype=torch.float16,
            output_str_only=False,
        )

        if _ENABLE_UPDATE_WEIGHTS:
            # test update weights
            fsdp_state_dict = _get_fsdp_state_dict(hf_model=hf_model)
            print(
                f"subprocess[{tp_rank=}] call update_weights_from_tensor ({list(fsdp_state_dict.keys())=})",
                flush=True,
            )
            fragment.update_weights_from_tensor(
                [(k, v) for k, v in fsdp_state_dict.items()]
            )

        srt_outputs = SRTRunner.forward_generation_raw(
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            lora_paths=None,
            engine=fragment,
        )
        print(
            f"subprocess[{tp_rank=}] call srt.forward {srt_outputs=}",
            flush=True,
        )

        check_close_model_outputs(
            hf_outputs=hf_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=0.2,
            decode_tolerance=0.2,
            rouge_l_tolerance=1,
        )

        execution_ok = True

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    fragment.shutdown()
    print(f"subprocess[{tp_rank=}] end", flush=True)


# Adapted from https://github.com/volcengine/verl/blob/main/tests/rollout/run_fsdp_vllm.py
def _get_fsdp_state_dict(hf_model):
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(_TP_SIZE,), mesh_dim_names=["fsdp"]
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp_model = FSDP(
        hf_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    print(f"{fsdp_model=}")

    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
    )

    return fsdp_model.state_dict()


if __name__ == "__main__":
    unittest.main()
