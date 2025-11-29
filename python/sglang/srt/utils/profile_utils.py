import torch


class Profiler:
    def __init__(self):
        TODO

    def step(self):
        TODO


class ProfilerCore:
    def start(self):
        stage_str = f" for {stage.name}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:  # for ROCM
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
            )

            if self.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
        elif torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                on_trace_ready=(
                    None
                    if not _is_npu
                    else torch_npu.profiler.tensorboard_trace_handler(
                        self.torch_profiler_output_dir
                    )
                ),
            )
            self.torch_profiler.start()

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)

        if "CUDA_PROFILER" in activities:
            torch.cuda.cudart().cudaProfilerStart()

        return ProfileReqOutput(success=True, message="Succeeded")

    def stop(self):
        TODO
