from python.sglang.srt.managers.scheduler import Scheduler
from python.sglang.srt.server_args import ServerArgs, PortArgs
from torch.distributed.device_mesh import init_device_mesh


def run():
    # build distributed world
    local_rank, rank, world_size = initialize_global_process_group()

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
    gen_device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, tp_size), mesh_dim_names=['dp', 'tp'])
    # build inference engine
    inference_engine = Scheduler(
        server_args=ServerArgs(
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            mem_fraction_static=0.1,
            tp_size=tp_size,
            dp_size=dp_size,
        ),
        port_args=PortArgs(
            tokenizer_ipc_name=TODO,
            scheduler_input_ipc_name=TODO,
            detokenizer_ipc_name=TODO,
            nccl_port=TODO,
        ),
        gpu_id=0,  # TODO
        tp_rank=TODO,
        dp_rank=TODO,
    )

    # moved to above
    # # [Optional] update parallel state in SGLang for 3D-HybridEngine
    # inference_engine.update_parallel_state(TP=device_mesh["tp"])

    # generate sequence, it would be better if the output is a list of Tensor not list of list[str]
    outputs = inference_engine.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)

    # already done in old PR, waiting for merging
    # # offload kvcache after generation
    # inference_engine.free_kvcache()  # inference_engine.init_kvcache()
    # # offload model
    # inference_engine.offload_model_weights()  # inference_engine.load_model_weights(), we can simply re-init them


if __name__ == '__main__':
    run()
