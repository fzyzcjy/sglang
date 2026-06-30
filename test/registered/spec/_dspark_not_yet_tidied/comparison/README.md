# DSpark model-parity ("comparison") tests

This folder holds **exactly** the category-1 model-parity tests: every test here
pins a piece of the production DSpark draft against an **independent reference**
— for almost all of them the external **source of truth (SoT)** oracle, and for
the cuda-graph geometry test the eager forward — and fails if they disagree
numerically. Nothing else belongs here (protocol/unit and e2e tests stay in the
parent `test/registered/spec/dspark/`).

## How a comparison test works

The SoT is a vendored, independently-derived reference under
`test/manual/_dspark_reference/`:

- dsv4 (DeepSeek-V4-Flash DSpark): `deepseek_v4/` — the pure-torch non-causal
  sparse-attention oracle (`sot_attention.py`), the `RefTransformer.forward_spec`
  draft modeling + DSpark heads + mHC math (`modeling.py`), and the shared
  config + weight-sync fixture (`parity_fixture.py`).
- dense (Qwen3 / Gemma4 DSpark): `qwen3/modeling.py`, `gemma4/modeling.py`,
  `markov_head.py`, `draft_ops.py`, `sampling.py`.

Each test builds the production module **and** the SoT with the **same real
DeepSeek-V4-Flash / DeepSpec dimensions**, syncs the **same random weights** into
both, feeds the **same inputs**, and `assert_close`s the outputs. The production
path and the oracle derive the math independently, so a shared bug cannot make a
wrong production path pass. Pure-linear components compare tight (`1e-4`);
through-sparse-attention / fp8 paths compare within the fp8 noise floor
(`~5e-2`).

Two granularities:

- **A — per component.** Tap one production sub-module (a projection, the head
  collapse, the Markov correction, ...) and compare it alone.
- **B — whole block.** Run the full draft block forward through the **real
  production attention backend** (`DeepseekV4AttnBackend` / the dense MHA
  backend) and compare the final base logits. This is the load-bearing guardrail
  for the non-causal full-block sparse attention.

## dsv4 (DeepSeek-V4-Flash DSpark draft) — component → protecting test

| Component (production symbol) | Test file :: method | What it pins against the SoT |
|---|---|---|
| MLA-LoRA KV projection (`wkv` + `kv_norm`, `kv_proj_only`) | `test_deepseek_v4_component_parity.py` :: `test_kv_projection_parity` | `kv_proj_only` + `kv_norm` equals SoT `kv_norm(wkv(·))` (the MLA-LoRA weight map) |
| Target-hidden projection (`main_proj` + `main_norm`) | `test_deepseek_v4_component_parity.py` :: `test_target_hidden_projection_parity` | `project_target_hidden` equals SoT `main_norm(main_proj(·))` |
| mHC head collapse (`hc_head`) | `test_deepseek_v4_component_parity.py` :: `test_hc_head_collapse_parity` | `collapse_hc_head` equals SoT `hc_head(·)` (PRE-norm collapse) |
| Base-logits head (`hc_head` → `norm` → `lm_head`) | `test_deepseek_v4_component_parity.py` :: `test_base_logits_from_hidden_parity` | full-vocab base logits equal the SoT head math |
| Markov head, serial correction | `test_deepseek_v4_component_parity.py` :: `test_markov_head_serial_correction_parity` | `sample_block` equals the SoT serial bias-then-sample loop |
| Confidence head | `test_deepseek_v4_component_parity.py` :: `TestDsv4ComponentParityWithConfidence::test_confidence_head_parity` | `compute_confidence` equals the SoT confidence on the post-`hc_head` PRE-norm tap |
| Non-causal full-block sparse attention + whole-block forward (through real `DeepseekV4AttnBackend`) | `test_deepseek_v4_block_forward_sot_parity.py` :: `test_non_causal_block_forward_matches_sot` (+ negative `test_causal_index_regression_diverges_from_sot`) | production block-forward base logits equal the SoT within fp8 tolerance; forcing the production index builder causal makes parity diverge (proves non-causality is load-bearing) |
| Per-row independence (dynamic batch) | `test_deepseek_v4_dynamic_batch.py` :: `test_mixed_length_batch_rows_are_independent` | a batched row's block-forward equals the same request run alone (mixed prompt + accept lengths) |
| Tensor-parallel sharding | `test_deepseek_v4_tp_parity.py` :: `test_tp2_block_forward_matches_tp1_on_q_pad_gap_config` | TP=2 base logits equal TP=1 on an `n_local_heads` q-pad-gap config (2-GPU) |
| Worker draft-block wiring | `test_deepseek_v4_worker_parity.py` :: `test_v4_decode_block_matches_sot_forward_spec` | the worker's V4 draft block equals the SoT `forward_spec` (harness-driven) |
| Ragged-verify cuda-graph geometry | `test_deepseek_v4_ragged_verify_graph_parity.py` :: `test_graph_verify_logits_match_eager_on_mixed_verify_lens` (+ negative `test_force_uniform_capture_seam_makes_parity_diverge`, `test_accept_length_floor_under_mixed_verify_lens`) | compact ragged-verify cuda-graph logits equal the eager forward on mixed `verify_lens`; the force-uniform capture seam diverges; accept length stays above the floor |

## dense (Qwen3 / Gemma4 DSpark draft) — component → protecting test

| Component (production symbol) | Test file :: method | What it pins against the SoT |
|---|---|---|
| Context KV injection (`kv_proj_only` + norms) | `test_dspark_model_parity.py` :: `test_ctx_kv_injection_parity` (Qwen3 + Gemma4) | SGLang ctx K/V equals the reference projections |
| Target-hidden projection (`fc` + `hidden_norm`) | `test_dspark_model_parity.py` :: `test_fc_hidden_norm_projection_parity` | `project_target_hidden` equals reference `hidden_norm(fc(·))` |
| Draft logits + greedy tokens (`lm_head`) | `test_dspark_model_parity.py` :: `test_draft_logits_greedy_token_parity` | draft logits (lm_head matmul) and greedy tokens match the reference |
| In-model Markov correction | `test_dspark_model_parity.py` :: `test_markov_head_parity_vanilla` | the markov bias matches the reference for the vanilla config |
| `draft_probs` row order (no anchor row) | `test_dspark_model_parity.py` :: `test_draft_probs_row_order_no_anchor_row` | row 0 is `q(s_0)`; there is no spurious anchor row |
| Gemma4 shared-K/V load (`attention_k_eq_v`) | `test_dspark_model_parity.py` :: `TestGemma4DSparkModelParityKEqV` | the shared-K/V weight-load branch matches the reference |
| Dense whole-block forward (through MHA backend) | `test_dense_block_forward_parity.py` :: `test_dense_block_forward_matches_sot` (Qwen3 + Gemma4 subclasses) | dense draft forward backbone hidden + logits match the SoT whole-block (granularity B) |
| Markov head variants (`VanillaMarkov` / `GatedMarkovHead` / `RNNHead`) | `test_dspark_markov_parity.py` :: `test_compute_step_bias_exact_match`, `test_apply_block_logits_exact_match`, `test_sample_block_greedy_tokens_exact_match` (per variant) + `test_rnn_state_carries_across_steps` + `TestBuildMarkovHead` | each head's step bias, block-logits application, and greedy sampling match the reference exactly; the RNN carries state across steps; `build_markov_head` dispatches the right type |
| Confidence head (`DSparkConfidenceHead`) | `test_dspark_confidence_parity.py` :: `test_with_markov_raw_logit_matches_reference`, `test_without_markov_matches_hidden_only_reference`, `test_markov_embed_stack_uses_prev_token_offset`, `TestBuildConfidenceHead`, `TestMissingConfidenceWeightsRaises` | the RAW logit matches `AcceptRatePredictor(cat([h, m]))`; the with/without-markov input wiring, the off-by-one markov-embed stack, the build dispatch, and the missing-weights guard all match the contract |

## Notes

- **Real dimensions.** Every test builds at the real DeepSeek-V4-Flash /
  DeepSpec per-tensor dimensions (dsv4: head_dim 512, hidden 4096, vocab 129280,
  q/o_lora 1024, moe_intermediate 2048, n_routed_experts 256, window 128,
  hc_mult 4, gamma 5, markov_rank 256); only the draft **stage count** is a
  reduced instance count, never a faked dimension.
- **Negative tests are part of the guardrail.** `test_deepseek_v4_block_forward_sot_parity`
  and `test_deepseek_v4_ragged_verify_graph_parity` each include a negative seam that
  must make parity diverge — otherwise the positive test could pass vacuously.
- **The reference is not collected by CI.** It lives under `test/manual/` (CI
  only requires registered tests under `test/registered/**`); these comparison
  tests import it as the numerical standard answer.
