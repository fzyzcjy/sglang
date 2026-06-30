# DSpark V4 (DeepSeek-V4-Flash DSpark) parity reference package.
#
# Holds both ends of the dsv4 draft parity check, fed identical weights + inputs:
#  * the EXTERNAL source-of-truth (SoT) oracle -- ``sot_attention.py`` (pure-torch
#    non-causal sparse-attention) and ``modeling.py`` (the full DSpark draft modeling
#    + DSpark heads + mHC math), with ``parity_fixture.py`` carrying the shared tiny
#    config and the SoT->SGLang weight-sync helper;
#  * the PRODUCTION block-forward harness -- ``sglang_block_forward_harness.py`` --
#    which drives the real SGLang dsv4 draft path through the real backend / KV pool
#    and compares its output against the SoT oracle above.
#
# The production path and the oracle derive the math independently, so a shared bug
# cannot make a wrong production path pass.
