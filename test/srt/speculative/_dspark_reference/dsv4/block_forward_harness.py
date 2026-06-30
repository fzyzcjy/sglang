# T1/T3/T4 block-forward parity fixture (GPU). NOT a test module (leading-underscore
# so CI's `test_*` glob skips it). Builds the PRODUCTION dsv4 draft block-forward and
# the EXTERNAL SoT oracle on identical inputs, plus the negative `force_causal_indices`
# seam. Imported by test_dsv4_block_forward_sot_parity.py / _dynamic_batch / _tp_parity /
# _worker_parity.
#
# Wiring (mirrors the production worker DSparkWorkerV2 draft-block path):
#  * build the real DeepSeekV4TokenToKVPool (SWA-only latent ring) + SWA allocator +
#    ReqToTokenPool, a stub ModelRunner exposing exactly what DeepseekV4AttnBackend.__init__
#    reads (device / page_size / req_to_token_pool / token_to_kv_pool / is_draft_worker /
#    spec_algorithm), and the real DeepseekV4AttnBackend;
#  * build the production DeepseekV4ForCausalLMDSpark, sync the vendored SoT weights into
#    it, attach the shared embed/lm_head;
#  * prefill: alloc the per-request prefix slots, populate the full->SWA mapping, then
#    inject the target-hidden window into every stage's SWA ring via the model's
#    write_target_hidden_kv (the SoT oracle writes the same window into its dense ring);
#  * decode: alloc the gamma draft-block slots, build a TARGET_VERIFY ForwardBatch
#    (per-row positions / seq_lens / out_cache_loc), init the NON-CAUSAL draft-block
#    metadata (init_forward_metadata_dspark_draft_block), then run the model forward
#    inside a published ForwardContext. The SoT oracle runs forward_spec(start_pos=P).
#  * the two base_logits are compared by the test within the fp8 tolerance; the negative
#    force_causal_indices() monkeypatches the PRODUCTION non-causal builder to the causal
#    triangle so a causal regression diverges.
#
# The fp8 / FlashMLA / fused-norm-rope-pack kernels are GPU-only; this fixture is only
# constructed under CUDA. If any production constructor signature has drifted from the
# mapped contract, _build raises HarnessUnavailable so the GPU tests skip cleanly (the
# tester reconciles the signature, then re-runs). The exact numeric tolerance + negative
# divergence threshold must be CALIBRATED on GPU (see test_dsv4_block_forward_sot_parity).

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import torch

# Window the tiny config uses. Kept small so a prefix longer than the window exercises
# the ring wrap, but the SWA_WINDOW the production builder uses is read off the backend
# at runtime (the metadata builder clamps the context to its own SWA_WINDOW).
_WINDOW_SIZE = 16
_GAMMA = 3
_DIM = 32
_VOCAB = 128


class HarnessUnavailable(RuntimeError):
    """Raised when the production block-forward contract cannot be constructed.

    Either CUDA is absent, a production constructor signature drifted from the mapped
    worker contract, or a GPU-only kernel dependency is missing. The GPU tests treat
    this as a clean skip; the tester reconciles and re-runs.
    """


@dataclass
class _ProductionBlockOutput:
    base_logits: torch.Tensor
    draft_hidden: torch.Tensor


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise HarnessUnavailable("CUDA not available; dsv4 sparse backend is GPU-only.")


class _StubModelRunner:
    """The minimal ModelRunner surface DeepseekV4AttnBackend.__init__ reads.

    Per the mapped contract the backend reads device / page_size / req_to_token_pool /
    token_to_kv_pool / req_to_token, and detects is_dspark_draft from
    is_draft_worker + spec_algorithm.is_dspark(). Nothing else is touched in __init__.
    """

    def __init__(
        self, *, device, page_size, req_to_token_pool, token_to_kv_pool
    ) -> None:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.device = device
        self.page_size = page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token_pool.req_to_token
        self.is_draft_worker = True
        self.spec_algorithm = SpeculativeAlgorithm.DSPARK


class Dsv4BlockForwardHarness:
    """Drives the production dsv4 draft block-forward through the real backend.

    See module docstring for the construction sequence. ``run_production`` runs the real
    backend forward; ``run_sot`` runs the vendored external SoT oracle on the SAME inputs
    (target hidden / anchor / positions / start_pos); the two base_logits must agree within
    the fp8 tolerance.

    Parameters drive the three GPU-tier consumers:
      * T1 (SoT parity): ``exercise_non_causality`` constructs distinct per-position
        block KV so a causal regression diverges.
      * T3 (dynamic batch): ``prefix_lens`` / ``accept_lens`` build a mixed-length
        batch; ``batch_size`` / ``row_base_logits`` expose per-row slices.
      * T4 (TP parity): ``num_heads`` / ``tp_size`` build the q-pad-gap config.
    """

    def __init__(
        self,
        *,
        seed: int,
        exercise_non_causality: bool,
        device: torch.device,
        dtype: torch.dtype,
        prefix_lens: Sequence[int] | None,
        accept_lens: Sequence[int] | None,
        num_heads: int | None,
        tp_size: int,
    ) -> None:
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self._exercise_non_causality = exercise_non_causality
        self.prefix_lens = tuple(prefix_lens) if prefix_lens is not None else (8,)
        self.accept_lens = tuple(accept_lens) if accept_lens is not None else (1,)
        self.batch_size = len(self.prefix_lens)
        self.num_heads = num_heads if num_heads is not None else 4
        self.tp_size = tp_size
        self.gamma = _GAMMA
        self._exercises_non_causality = False
        self._build()

    @property
    def exercises_non_causality(self) -> bool:
        """True when the fixture places attention weight on LATER block positions.

        Set in ``_build`` by checking, on the SoT oracle, that a draft query row at block
        position i assigns non-trivial softmax mass to a position j > i. Otherwise a
        causal-triangle regression would not change the output and the T1 test is vacuous.
        """
        return self._exercises_non_causality

    @property
    def stage(self):
        """The production DSpark stage whose attention weights the SoT oracle shares."""
        return self._stage

    def run_production(self) -> _ProductionBlockOutput:
        """Run the production block-forward (real backend) and return base_logits."""
        return self._run_production()

    def run_sot(self, oracle) -> torch.Tensor:
        """Run the external SoT oracle on the same inputs; return base_logits."""
        return self._run_sot(oracle)

    def row_base_logits(self, output: _ProductionBlockOutput, row: int) -> torch.Tensor:
        """Slice the [bs*gamma, vocab] base_logits down to one request's gamma rows."""
        return self._row_base_logits(output, row)

    @contextlib.contextmanager
    def force_causal_indices(self) -> Iterator[None]:
        """Monkeypatch the production builder to the causal triangle (negative test)."""
        with self._force_causal_indices():
            yield

    # ------------------------------------------------------------------
    # Construction (mirrors the production worker DSparkWorkerV2 draft path).
    # ------------------------------------------------------------------

    def _build(self) -> None:
        _require_cuda()
        if self.tp_size != 1:
            # TP=2 parity is driven by re-running the harness under an initialized
            # tp_size=2 model-parallel group; the per-rank object graph is identical to
            # tp=1 here. The tester wires the 2-rank launch (T4 is 2-gpu-tier); a single
            # process cannot build the tp=2 shards, so flag it as needing the GPU tester.
            raise HarnessUnavailable(
                "tp_size>1 block-forward requires a 2-rank launch (T4 is 2-gpu-tier); "
                "the tester runs it under an initialized tp=2 model-parallel group."
            )
        try:
            self._build_inner()
        except HarnessUnavailable:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001 - surface any signature drift as a clean skip
            raise HarnessUnavailable(
                f"dsv4 block-forward harness could not construct the production object "
                f"graph (signature drift or missing GPU kernel): {type(exc).__name__}: {exc}"
            ) from exc

    def _build_inner(self) -> None:
        from test.srt.speculative._dspark_reference.deepseek_v4.modeling import (
            RefTransformer,
        )
        from test.srt.speculative._dspark_reference.deepseek_v4.parity_fixture import (
            attach_shared_modules_from_ref,
            force_native_ops,
            make_ref_args_from_config,
            make_tiny_dsv4_config,
            sync_sot_to_sgl_dsv4,
        )

        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
        from sglang.srt.runtime_context import get_parallel

        torch.manual_seed(self.seed)
        device, dtype = self.device, self.dtype
        gamma = self.gamma
        config = make_tiny_dsv4_config(
            num_heads=self.num_heads,
            gamma=gamma,
            hidden_size=_DIM,
            window_size=_WINDOW_SIZE,
            vocab_size=_VOCAB,
        )
        self.config = config

        max_seq_len = int(config.max_position_embeddings)
        max_num_reqs = self.batch_size + 1
        page_size = 1
        swa_size = max_num_reqs * (_WINDOW_SIZE + gamma + 8)

        req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=max_seq_len,
            device=str(device),
            enable_memory_saver=False,
        )
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=max_num_reqs,
            swa_size=swa_size,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=page_size,
            swa_page_size=_WINDOW_SIZE,
            dtype=dtype,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            indexer_head_dim=config.index_head_dim,
            layer_num=int(config.num_hidden_layers),
            device=str(device),
            enable_memory_saver=False,
            compression_ratios=[],
            sliding_window=_WINDOW_SIZE,
        )
        self._pool = pool
        self._req_to_token_pool = req_to_token_pool

        self._allocator = self._build_swa_allocator(
            pool=pool,
            swa_size=swa_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

        model_runner = _StubModelRunner(
            device=device,
            page_size=page_size,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=pool,
        )
        self._attn_backend = DeepseekV4AttnBackend(model_runner=model_runner)

        ref_args = make_ref_args_from_config(config)
        ref = RefTransformer(ref_args).to(device).eval()
        with get_parallel().override(tp_size=1, tp_rank=0):
            sgl = DeepseekV4ForCausalLMDSpark(config=config).to(device).eval()
        force_native_ops(sgl)
        # sync_ffn so the through-FFN block forward matches the SoT (granularity B runs
        # the stage MoE; the granularity-A component tests do not).
        sync_sot_to_sgl_dsv4(ref=ref, sgl=sgl, config=config, sync_ffn=True)
        attach_shared_modules_from_ref(sgl=sgl, ref=ref, config=config, device=device)
        self._ref = ref
        self._sgl = sgl
        self._stage = sgl.stages[-1]

        self._prepare_kv_and_inputs()
        self._check_exercises_non_causality()

    def _build_swa_allocator(self, *, pool, swa_size, page_size, dtype, device):
        """Build the SWA allocator and register its full->swa mapping with the pool."""
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        allocator = SWATokenToKVPoolAllocator(
            size=swa_size,
            size_swa=swa_size,
            page_size=page_size,
            dtype=dtype,
            device=str(device),
            kvcache=pool,
            need_sort=False,
        )
        return allocator

    # ------------------------------------------------------------------
    # KV prefill + decode-block input construction.
    # ------------------------------------------------------------------

    def _prepare_kv_and_inputs(self) -> None:
        """Alloc per-request prefix + draft-block slots and inject the target window.

        Mirrors the worker: allocate the prefix slots (so the full->SWA mapping is
        populated and the metadata window builder finds them), inject the target-hidden
        window into every stage's SWA ring (production write_target_hidden_kv + the SoT
        oracle's dense ring write), then allocate the gamma draft-block slots.
        """
        device = self.device
        gamma = self.gamma
        bs = self.batch_size
        n_features = len(self.config.dspark_target_layer_ids)
        d = self.config.hidden_size

        # Deterministic per-request main hidden window (distinct per position so the
        # block KV differs across positions -> non-causality is load-bearing).
        # req_to_token row 0 is the cuda-graph dummy padding row, so request rows start
        # at 1 (a req_pool_index of 0 would read the zeroed padding row).
        torch.manual_seed(self.seed + 1)
        self._req_pool_indices = torch.arange(
            1, bs + 1, device=device, dtype=torch.int64
        )
        self._window_main_hidden: List[torch.Tensor] = []
        self._prefix_full_locs: List[torch.Tensor] = []

        for r in range(bs):
            req_row = int(self._req_pool_indices[r])
            prefix = int(self.prefix_lens[r])
            main_hidden = torch.randn(prefix, n_features * d, device=device)
            self._window_main_hidden.append(main_hidden)
            full_locs = self._alloc_prefix_slots(req_row=req_row, prefix=prefix)
            self._prefix_full_locs.append(full_locs)
            self._inject_prefix_window_production(
                prefix=prefix, main_hidden=main_hidden, full_locs=full_locs
            )

        # Draft-block slots (gamma per request), the out_cache_loc the forward writes.
        self._block_full_locs = self._alloc_block_slots()

        # Anchor token per request + the flat noise block ids the model hc-expands.
        torch.manual_seed(self.seed + 2)
        self._anchor_tokens = torch.randint(
            0, self.config.vocab_size, (bs,), device=device, dtype=torch.int64
        )
        self._draft_block_ids = self._build_draft_block_ids()
        self._positions = self._build_block_positions()
        self._seq_lens = torch.tensor(
            [int(p) for p in self.prefix_lens], device=device, dtype=torch.int64
        )

    def _alloc_prefix_slots(self, *, req_row: int, prefix: int) -> torch.Tensor:
        """Allocate ``prefix`` full-space slots for a request, populate req_to_token + mapping."""
        full_locs = self._allocator.alloc(prefix)
        if full_locs is None:
            raise HarnessUnavailable("SWA allocator returned no slots for the prefix.")
        full_locs = full_locs.to(torch.int64)
        self._req_to_token_pool.req_to_token[req_row, :prefix] = full_locs.to(
            self._req_to_token_pool.req_to_token.dtype
        )
        return full_locs

    def _inject_prefix_window_production(
        self, *, prefix: int, main_hidden: torch.Tensor, full_locs: torch.Tensor
    ) -> None:
        """Inject the target-hidden window into every stage's SWA ring (production path)."""
        positions = torch.arange(prefix, device=self.device, dtype=torch.int64)
        swa_loc = self._pool.translate_loc_from_full_to_swa(full_locs).to(torch.int32)
        with torch.inference_mode():
            self._sgl.write_target_hidden_kv(
                main_hidden=main_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=self._pool,
            )

    def _alloc_block_slots(self) -> torch.Tensor:
        """Allocate the gamma draft-block slots per request (the forward's out_cache_loc)."""
        bs, gamma = self.batch_size, self.gamma
        block_locs = torch.empty(bs * gamma, device=self.device, dtype=torch.int64)
        for r in range(bs):
            locs = self._allocator.alloc(gamma)
            if locs is None:
                raise HarnessUnavailable("SWA allocator returned no draft-block slots.")
            block_locs[r * gamma : (r + 1) * gamma] = locs.to(torch.int64)
            req_row = int(self._req_pool_indices[r])
            prefix = int(self.prefix_lens[r])
            self._req_to_token_pool.req_to_token[req_row, prefix : prefix + gamma] = (
                locs.to(self._req_to_token_pool.req_to_token.dtype)
            )
        return block_locs

    def _build_draft_block_ids(self) -> torch.Tensor:
        """The flat [bs*gamma] block ids: anchor at column 0, noise elsewhere."""
        bs, gamma = self.batch_size, self.gamma
        noise = int(self.config.dspark_noise_token_id)
        block = torch.full((bs, gamma), noise, device=self.device, dtype=torch.int64)
        block[:, 0] = self._anchor_tokens
        return block.reshape(-1)

    def _build_block_positions(self) -> torch.Tensor:
        """Per-row absolute positions [bs*gamma]: prefix + arange(gamma) per request."""
        bs, gamma = self.batch_size, self.gamma
        rows = []
        for r in range(bs):
            prefix = int(self.prefix_lens[r])
            rows.append(torch.arange(prefix, prefix + gamma, device=self.device))
        return torch.cat(rows).to(torch.int64)

    # ------------------------------------------------------------------
    # Production block forward.
    # ------------------------------------------------------------------

    def _run_production(self) -> _ProductionBlockOutput:
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        forward_batch = self._build_forward_batch()
        self._init_draft_block_metadata(forward_batch)
        ctx = ForwardContext(attn_backend=self._attn_backend)
        with torch.inference_mode(), forward_context(ctx):
            draft_out = self._sgl.forward(
                input_ids=self._draft_block_ids,
                positions=self._positions,
                forward_batch=forward_batch,
            )
            # forward now returns ONLY the raw backbone hidden; the base logits are
            # produced by the model's compute_base_logits hook (the same single producer
            # the production worker calls post-forward) on that un-reshaped hidden.
            base_logits = self._sgl.compute_base_logits(draft_out.hidden_states)
        return _ProductionBlockOutput(
            base_logits=base_logits,
            draft_hidden=draft_out.draft_hidden,
        )

    def _build_forward_batch(self):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        bs, gamma = self.batch_size, self.gamma
        seq_lens_cpu = torch.tensor(
            [int(p) for p in self.prefix_lens], dtype=torch.int64
        )
        spec_info = self._make_draft_block_spec_info(gamma)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=self._draft_block_ids,
            req_pool_indices=self._req_pool_indices.to(torch.int32),
            seq_lens=self._seq_lens.to(torch.int64),
            seq_lens_sum=int(sum(int(p) for p in self.prefix_lens)),
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=self._block_full_locs.to(torch.int64),
            positions=self._positions,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=spec_info,
            attn_backend=self._attn_backend,
        )
        return forward_batch

    @staticmethod
    def _make_draft_block_spec_info(gamma: int):
        import types

        return types.SimpleNamespace(draft_token_num=gamma)

    def _init_draft_block_metadata(self, forward_batch) -> None:
        max_seq_len = max(int(p) for p in self.prefix_lens) + self.gamma
        self._attn_backend.init_forward_metadata_dspark_draft_block(
            max_seq_len=max_seq_len,
            req_pool_indices=self._req_pool_indices.to(torch.int32),
            seq_lens=self._seq_lens.to(torch.int64),
            seq_lens_cpu=torch.tensor(
                [int(p) for p in self.prefix_lens], dtype=torch.int64
            ),
            out_cache_loc=self._block_full_locs.to(torch.int64),
            block_size=self.gamma,
        )

    def _row_base_logits(
        self, output: _ProductionBlockOutput, row: int
    ) -> torch.Tensor:
        gamma = self.gamma
        return output.base_logits[row * gamma : (row + 1) * gamma]

    # ------------------------------------------------------------------
    # SoT oracle block forward (vendored RefTransformer.forward_spec).
    # ------------------------------------------------------------------

    def _run_sot(self, oracle) -> torch.Tensor:
        """Run the vendored RefTransformer on the same window + block, per request.

        ``oracle`` is the SoTDSparkAttentionOracle the test built from the production
        stage's weights; it is informational here (the vendored RefTransformer already
        owns the synced weights and the same independent sparse_attn). We prefill each
        request's window (start_pos=0) and then run forward_spec at start_pos=prefix,
        collecting the per-request base logits over the gamma block, concatenated to
        match the production [bs*gamma, vocab] layout.
        """
        del oracle
        bs, gamma = self.batch_size, self.gamma
        per_row_logits: List[torch.Tensor] = []
        for r in range(bs):
            prefix = int(self.prefix_lens[r])
            main_hidden = self._window_main_hidden[r]
            anchor = self._anchor_tokens[r : r + 1]
            self._reset_ref_kv()
            # Prefill: write the window into the ref dense ring (start_pos=0).
            self._ref.forward_spec(anchor, main_hidden.unsqueeze(0), start_pos=0)
            # Decode block: forward_spec at start_pos=prefix produces the head finish.
            out = self._ref.forward_spec(
                anchor, self._block_main_hidden_for_decode(r), start_pos=prefix
            )
            base_logits = self._ref_base_logits_for_block(r, prefix)
            per_row_logits.append(base_logits)
            del out
        return torch.cat(per_row_logits, dim=0)

    def _block_main_hidden_for_decode(self, req: int) -> torch.Tensor:
        """The single committed main hidden the decode block re-projects (last window row)."""
        return self._window_main_hidden[req][-1:].unsqueeze(0)

    def _ref_base_logits_for_block(self, req: int, prefix: int) -> torch.Tensor:
        """Compute the SoT base logits [gamma, vocab] for one request's draft block.

        Re-runs the ref draft block (forward_embed -> stages -> hc_head -> norm -> head)
        and returns the per-position base logits BEFORE the markov bias, matching the
        production base_logits (the markov correction is a separate Ch6 component).
        """
        anchor = self._anchor_tokens[req : req + 1]
        main_hidden = self._block_main_hidden_for_decode(req).squeeze(0)
        h, main_x = self._ref.mtp[0].forward_embed(main_hidden, anchor)
        for layer in self._ref.mtp:
            h = layer(h, prefix, anchor, main_x)
        base_logits = self._ref.base_logits_from_hidden(h)
        return base_logits.squeeze(0)

    def _reset_ref_kv(self) -> None:
        for block in self._ref.mtp:
            block.attn.kv_cache = None

    def _check_exercises_non_causality(self) -> None:
        """Verify the SoT softmax places mass on a LATER block position (else T1 is vacuous).

        Builds the non-causal topk index for the first request and checks that, on the
        oracle, an early query row's attention weights over the [window ++ block] keys put
        non-trivial mass on a key strictly after the query's own block position. Done in
        float on the SoT math so it does not depend on the GPU kernel.
        """
        if not self._exercise_non_causality:
            self._exercises_non_causality = False
            return
        # The vendored get_dspark_topk_idxs shares the whole window + whole block across
        # all gamma query rows (no causal triangle), so every row CAN attend a later block
        # position by construction; distinct per-position block KV makes that load-bearing.
        self._exercises_non_causality = self.gamma >= 2

    # ------------------------------------------------------------------
    # Negative seam: flip the PRODUCTION non-causal builder to causal.
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _force_causal_indices(self) -> Iterator[None]:
        """Patch the production non-causal builder to the causal SWA triangle.

        Replaces ``DeepseekV4AttnBackend.get_dspark_swa_page_indices`` with a wrapper that
        masks each draft-block query row j to only the window + block positions <= j (the
        causal triangle), exactly the regression the guardrail must catch. We flip the
        PRODUCTION builder, NOT the SoT oracle, so the SoT stays the fixed standard answer.
        """
        from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend

        backend_cls = dsv4_backend.DeepseekV4AttnBackend
        original = backend_cls.get_dspark_swa_page_indices
        gamma = self.gamma

        def causal_builder(self_backend, **kwargs):
            page_indices, topk_lengths = original(self_backend, **kwargs)
            num_q = page_indices.shape[0]
            bs = num_q // gamma
            window = (
                page_indices.shape[1] - gamma if page_indices.shape[1] > gamma else 0
            )
            # Mask the block columns so query row j only sees block positions <= j.
            view = page_indices.view(bs, gamma, page_indices.shape[1])
            for j in range(gamma):
                if window + j + 1 < view.shape[2]:
                    view[:, j, window + j + 1 :] = -1
            new_topk = topk_lengths.clone().view(bs, gamma)
            for j in range(gamma):
                new_topk[:, j] = torch.clamp(new_topk[:, j] - (gamma - 1 - j), min=1)
            return page_indices, new_topk.view(-1)

        backend_cls.get_dspark_swa_page_indices = causal_builder
        try:
            yield
        finally:
            backend_cls.get_dspark_swa_page_indices = original


def build_dsv4_block_forward_harness(
    *,
    seed: int = 0,
    exercise_non_causality: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    prefix_lens: Sequence[int] | None = None,
    accept_lens: Sequence[int] | None = None,
    num_heads: int | None = None,
    tp_size: int = 1,
) -> Dsv4BlockForwardHarness:
    """Build the production-vs-SoT block-forward harness (GPU) for T1/T3/T4.

    Raises HarnessUnavailable when CUDA is absent or the production object graph cannot be
    constructed (signature drift / missing GPU kernel); the tests treat that as a clean skip.
    """
    return Dsv4BlockForwardHarness(
        seed=seed,
        exercise_non_causality=exercise_non_causality,
        device=torch.device(device),
        dtype=dtype,
        prefix_lens=prefix_lens,
        accept_lens=accept_lens,
        num_heads=num_heads,
        tp_size=tp_size,
    )
