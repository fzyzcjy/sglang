import logging
from typing import Dict, Iterable, Tuple

import torch

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors = None

    def handle(self, action: str):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        else:
            raise Exception(f"Unsupported {action=}")

    def _snapshot(self):
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=_postprocess_tensors(self._snapshot_tensors),
            actual_tensors=_postprocess_tensors(dict(self._model_state())),
        )

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _check_tensors(
        expect_tensors: Iterable[Tuple[str, torch.Tensor]], actual_tensors: Iterable[Tuple[str, torch.Tensor]]
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []

    for (expect_name, expect), (actual_name, actual) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        name = expect_name

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            error_messages.append(
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )

    logger.info(f"[check_tensors] passed: {good_names}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _postprocess_tensors(raw: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    remain_names = sorted(list(raw))

    # dequant fp8
    interest_names = [
        name for name in remain_names
        if name.endswith(".weight") and name.replace(".weight", ".weight_scale_inv") in raw
    ]
    remain_names = [x for x in remain_names if x not in interest_names]
    for name in interest_names:
        weight = raw[name]
        scale = raw[name.replace(".weight", ".weight_scale_inv")]
        TODO

    for name in remain_names:
        yield name, raw[name]
