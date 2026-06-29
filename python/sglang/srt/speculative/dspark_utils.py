from __future__ import annotations

import logging
from typing import Any, List, Optional

import msgspec

from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_GAMMA = 7
SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")


class DSparkLengthContract(msgspec.Struct, frozen=True):
    """Length contract for DSpark semi-AR static verify (plan ``§2``).

    ``gamma`` is the number of proposed draft tokens (DeepSpec ``block_size``).
    The draft block is exactly ``gamma`` slots ``[anchor, mask×(gamma-1)]`` and
    every slot is sampled (serial Markov), producing ``s_0..s_{gamma-1}``. The
    verify window is ``gamma + 1`` tokens ``[anchor, s_0..s_{gamma-1}]``.
    """

    gamma: int

    @property
    def num_draft_positions(self) -> int:
        return int(self.gamma)

    @property
    def verify_num_draft_tokens(self) -> int:
        return int(self.gamma) + 1

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self.verify_num_draft_tokens

    def validate(self) -> None:
        if int(self.gamma) < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")


def make_dspark_length_contract(*, gamma: int) -> DSparkLengthContract:
    contract = DSparkLengthContract(gamma=int(gamma))
    contract.validate()
    return contract


def dspark_gamma_from_num_draft_tokens(num_draft_tokens: int) -> int:
    gamma = int(num_draft_tokens) - 1
    if gamma < 1:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {num_draft_tokens}."
        )
    return gamma


class DSparkDraftConfig(msgspec.Struct, frozen=True):
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    gamma: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]
    markov_rank: int
    markov_head_type: Optional[str]

    def resolve_gamma(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.gamma if self.gamma is not None else default

    def require_markov(self) -> bool:
        return int(self.markov_rank) > 0


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    return config


def _get_dspark_config(config: Any) -> dict:
    cfg = _cfg_get(config, "dspark_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def parse_dspark_draft_config(*, draft_hf_config: Any) -> DSparkDraftConfig:
    """Parse DSpark draft config fields from an HF config/dict.

    Reuses the DFlash parser for the shared backbone/context fields (the DSpark
    draft backbone is DFlash-shaped) and adds the Markov-head fields. The HF
    ``block_size`` field is interpreted as ``gamma`` (number of draft positions),
    not the verify window. Confidence-head fields are intentionally ignored: the
    static-verify MVP does not build/load/use the confidence head (plan ``§0``).
    """
    base = parse_dflash_draft_config(draft_hf_config=draft_hf_config)

    dspark_cfg = _get_dspark_config(draft_hf_config)
    text_config = _get_text_config(draft_hf_config)

    raw_markov_rank = dspark_cfg.get(
        "markov_rank",
        _cfg_get(text_config, "markov_rank", _cfg_get(draft_hf_config, "markov_rank", 0)),
    )
    markov_rank = int(raw_markov_rank) if raw_markov_rank is not None else 0
    if markov_rank < 0:
        raise ValueError(f"DSpark markov_rank must be >= 0, got {markov_rank}.")

    markov_head_type = dspark_cfg.get(
        "markov_head_type",
        _cfg_get(
            text_config,
            "markov_head_type",
            _cfg_get(draft_hf_config, "markov_head_type", None),
        ),
    )
    if markov_rank > 0:
        if markov_head_type is None:
            raise ValueError(
                "DSpark requires markov_head_type when markov_rank > 0, got None."
            )
        markov_head_type = str(markov_head_type).lower()
        if markov_head_type not in SUPPORTED_DSPARK_MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unsupported DSpark markov_head_type={markov_head_type!r}. "
                f"Supported: {SUPPORTED_DSPARK_MARKOV_HEAD_TYPES}."
            )

    return DSparkDraftConfig(
        num_hidden_layers=base.num_hidden_layers,
        num_target_layers=base.num_target_layers,
        gamma=base.block_size,
        target_layer_ids=base.target_layer_ids,
        mask_token=base.mask_token,
        mask_token_id=base.mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
    )
