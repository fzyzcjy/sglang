import time

from python.sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from python.sglang.srt.managers.scheduler import Scheduler
from python.sglang.srt.sampling.sampling_params import SamplingParams
from python.sglang.srt.server_args import ServerArgs, PortArgs
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer
from verl.distributed import initialize_global_process_group


def run():
    # build distributed world
    local_rank, rank, world_size = initialize_global_process_group()

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # `sync_model_weights` not in this PR
    # # build device mesh for training engine.
    # device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    # fsdp_model = FSDP(actor_model, ..., device_mesh = device_mesh)
    # FSDP.set_state_dict_type(fsdp_model,
    #                          state_dict_type=StateDictType.SHARDED_STATE_DICT,
    #                          state_dict_config=ShardedStateDictConfig())
    # # get sharded model state dict
    # state_dict = fsdp_model.state_dict()
    # # sync weights between actor and rollout, support several format: DTensor and Megatron (sharded)
    # inference_engine.sync_model_weights(actor_weights=state_dict, load_format='dtensor')

    # [Optional] build device mesh for inference engine
    dp_size, tp_size = 2, 4
    assert world_size == dp_size * tp_size
    gen_device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, tp_size), mesh_dim_names=['dp', 'tp'])
    # build inference engine
    inference_engine = Scheduler(
        server_args=ServerArgs(
            model_path=model_name,
            mem_fraction_static=0.1,
            tp_size=tp_size,
            dp_size=dp_size,
        ),
        port_args=PortArgs(
            tokenizer_ipc_name='/tmp/hack_sglang/tokenizer_ipc',
            scheduler_input_ipc_name='/tmp/hack_sglang/scheduler_input_ipc',
            detokenizer_ipc_name='/tmp/hack_sglang/detokenizer_ipc',
            nccl_port=12345,
        ),
        gpu_id=0,  # TODO
        tp_rank=TODO,
        dp_rank=TODO,
    )

    # moved to above
    # # [Optional] update parallel state in SGLang for 3D-HybridEngine
    # inference_engine.update_parallel_state(TP=device_mesh["tp"])

    input_text = "Today is a sunny day and I like"
    input_ids = hf_tokenizer(input_text)['input_ids'][0].tolist()

    def hack_send_to_detokenizer_callback(out):
        print('outputs', out)

    inference_engine.hack_send_to_detokenizer_callback = hack_send_to_detokenizer_callback

    # generate sequence, it would be better if the output is a list of Tensor not list of list[str]
    inference_engine.handle_generate_request(TokenizedGenerateReqInput(
        rid='req-0',  # TODO when multi req, handle this
        input_text=input_text,
        input_ids=input_ids,
        image_inputs={},
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        stream=True,  # TODO ?
    ))

    print('sleep')
    time.sleep(10)

    # already done in old PR, waiting for merging
    # # offload kvcache after generation
    # inference_engine.free_kvcache()  # inference_engine.init_kvcache()
    # # offload model
    # inference_engine.offload_model_weights()  # inference_engine.load_model_weights(), we can simply re-init them


if __name__ == '__main__':
    run()
