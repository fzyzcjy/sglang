import multiprocessing
import multiprocessing as mp
import traceback
import unittest
from multiprocessing import Process

import torch
from sglang import Engine
from sglang.srt.server.engine_fragment import EngineFragment
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from test.srt.test_update_weights_from_tensor import check_params_by_get_weights_by_name

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        multiprocessing.set_start_method("spawn")

        queue = multiprocessing.Queue()

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, queue, output_writer),
            )
            p.start()
            processes.append(p)

        output = output_reader.recv()
        print(output)
        self.assertEqual(
            output,
            [
                " to spend it outdoors. I decided to take a walk in the nearby park.",
                " how to improve the performance of my website. I've been doing some research and",
                " a new user of the platform. I am looking for a new laptop to buy",
                " I'm looking for someone to help me with a project.\nI'm a student",
                " the science of numbers and their properties. It is a vast and complex field that",
            ],
        )

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int, queue: multiprocessing.Queue, output_writer):
    try:
        print(f"subprocess[{tp_rank=}] Start")

        # Engine can be put anywhere, e.g. tp_rank=0, or other places
        if tp_rank == 0:
            engine = Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                mem_fraction_static=0.1,
                tp_size=_TP_SIZE,
                random_seed=42,
                fragment=True,
            )
            print(f"subprocess[{tp_rank=}] {engine=}", flush=True)

            for _ in range(_TP_SIZE):
                queue.put(engine.fragment_args)

        # can use e.g. torch.distributed to broadcast it; here for simplicity we do not init torch.distributed
        fragment_args = queue.get()

        fragment = EngineFragment(
            fragment_args=fragment_args,
            tp_rank=tp_rank,
            gpu_id=tp_rank,
        )
        print(f"subprocess[{tp_rank=}] {fragment=}", flush=True)

        if tp_rank == 0:
            engine.await_fragments()
            print(f"subprocess[{tp_rank=}] end wait engine launch", flush=True)

            ans = []
            for prompt in [
                ["Today is a sunny day and I like", "I have a very good idea on"],
                ["Hello, I am", "What is your name?", "Mathematics is defined as"],
            ]:
                print(f"subprocess[{tp_rank=}] Start generation", flush=True)
                outputs = engine.generate(
                    prompt=prompt,
                    sampling_params=[dict(max_new_tokens=16, temperature=0.0)]
                                    * len(prompt),
                )
                print(
                    f"subprocess[{tp_rank=}] End generation {tp_rank=} {prompt=} {outputs=}",
                    flush=True,
                )
                ans += [o["text"] for o in outputs]

            output_writer.send(ans)
            output_writer.close()

        _test_update_weights_from_tensor(tp_rank=tp_rank, fragment=fragment)

        if tp_rank == 0:
            print(f"subprocess[{tp_rank=}] engine.shutdown", flush=True)
            engine.shutdown()

            for _ in range(_TP_SIZE):
                queue.put("END")

        # Again, can use torch barrier
        assert queue.get() == "END"
        print(f"subprocess[{tp_rank=}] end", flush=True)

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        raise


def _test_update_weights_from_tensor(tp_rank: int, fragment):
    param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

    check_params_by_get_weights_by_name(fragment, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

    print(f"subprocess[{tp_rank=}] update_weights_from_tensor", flush=True)
    new_tensor = torch.full((16384, 2048), 10.0 + tp_rank)
    fragment.update_weights_from_tensor([(x, new_tensor) for x in param_names])

    for param_name in param_names[:3]:
        # TODO get subtensor
        check_params_by_get_weights_by_name(fragment, param_name, [tp_rank] * 5)


if __name__ == "__main__":
    unittest.main()
