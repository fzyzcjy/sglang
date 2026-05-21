from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.models.qwen3 import Qwen3ForCausalLM


class _WeightHolder:
    weight: object


def _make_qwen3_model(*, tie_word_embeddings: bool) -> SimpleNamespace:
    embed_tokens = _WeightHolder()
    embed_tokens.weight = object()

    lm_head = _WeightHolder()
    if not tie_word_embeddings:
        lm_head.weight = object()

    return SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings),
        model=SimpleNamespace(embed_tokens=embed_tokens),
        lm_head=lm_head,
    )


def test_set_embed_and_head_handles_tied_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    model = _make_qwen3_model(tie_word_embeddings=True)
    embed = object()
    head = object()

    Qwen3ForCausalLM.set_embed_and_head(model, embed, head)

    assert model.model.embed_tokens.weight is embed
    assert model.lm_head.weight is head


def test_set_embed_and_head_replaces_untied_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    model = _make_qwen3_model(tie_word_embeddings=False)
    embed = object()
    head = object()

    Qwen3ForCausalLM.set_embed_and_head(model, embed, head)

    assert model.model.embed_tokens.weight is embed
    assert model.lm_head.weight is head
