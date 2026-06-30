# DSpark V4 production block-forward fixture package.
#
# The vendored SoT oracle now lives in the sibling ``deepseek_v4/`` package
# (``deepseek_v4/sot_attention.py`` for the attention oracle, ``deepseek_v4/modeling.py``
# for the full DSpark draft modeling). This package keeps only the GPU block-forward
# harness (``block_forward_harness.py``) that drives the PRODUCTION dsv4 path through
# the real backend / pool and compares against that SoT oracle.
