import unittest


from sglang.srt.speculative.dspark_utils import (
    DEFAULT_DSPARK_GAMMA,
    SUPPORTED_DSPARK_MARKOV_HEAD_TYPES,
    DSparkLengthContract,
    dspark_gamma_from_num_draft_tokens,
    make_dspark_length_contract,
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSparkLengthContract(CustomTestCase):
    def test_verify_num_draft_tokens_is_gamma_plus_one(self):
        """Verify window is exactly gamma+1 (plan §2)."""
        for gamma in (1, 3, 7, 16):
            contract = DSparkLengthContract(gamma=gamma)
            self.assertEqual(contract.verify_num_draft_tokens, gamma + 1)

    def test_validate_accepts_gamma_ge_one(self):
        """validate() should not raise for gamma >= 1."""
        for gamma in (1, 2, 7, 100):
            DSparkLengthContract(gamma=gamma).validate()

    def test_validate_rejects_gamma_zero(self):
        """validate() must raise ValueError for gamma == 0."""
        contract = DSparkLengthContract(gamma=0)
        with self.assertRaises(ValueError):
            contract.validate()

    def test_validate_rejects_negative_gamma(self):
        """validate() must raise ValueError for negative gamma."""
        contract = DSparkLengthContract(gamma=-1)
        with self.assertRaises(ValueError):
            contract.validate()

    def test_make_dspark_length_contract_returns_correct_contract(self):
        """make_dspark_length_contract produces a validated DSparkLengthContract."""
        contract = make_dspark_length_contract(gamma=7)
        self.assertIsInstance(contract, DSparkLengthContract)
        self.assertEqual(contract.verify_num_draft_tokens, 8)

    def test_make_dspark_length_contract_rejects_zero_gamma(self):
        """make_dspark_length_contract raises ValueError for gamma < 1."""
        with self.assertRaises(ValueError):
            make_dspark_length_contract(gamma=0)

    def test_make_dspark_length_contract_rejects_negative_gamma(self):
        """make_dspark_length_contract raises ValueError for negative gamma."""
        with self.assertRaises(ValueError):
            make_dspark_length_contract(gamma=-3)


class TestDSparkGammaFromNumDraftTokens(CustomTestCase):
    def test_round_trip_gamma_to_num_draft_tokens(self):
        """gamma -> num_draft_tokens -> gamma round-trips correctly."""
        for gamma in (1, 3, 7, 15):
            contract = make_dspark_length_contract(gamma=gamma)
            num_draft_tokens = contract.speculative_num_draft_tokens
            recovered = dspark_gamma_from_num_draft_tokens(num_draft_tokens)
            self.assertEqual(recovered, gamma)

    def test_num_draft_tokens_2_gives_gamma_1(self):
        """Minimum valid num_draft_tokens=2 maps to gamma=1."""
        self.assertEqual(dspark_gamma_from_num_draft_tokens(2), 1)

    def test_num_draft_tokens_8_gives_gamma_7(self):
        """Default num_draft_tokens=8 maps to default gamma=7."""
        self.assertEqual(dspark_gamma_from_num_draft_tokens(8), DEFAULT_DSPARK_GAMMA)

    def test_raises_for_num_draft_tokens_less_than_2(self):
        """dspark_gamma_from_num_draft_tokens raises ValueError for input < 2."""
        with self.assertRaises(ValueError):
            dspark_gamma_from_num_draft_tokens(1)

    def test_raises_for_num_draft_tokens_zero(self):
        """dspark_gamma_from_num_draft_tokens raises ValueError for input == 0."""
        with self.assertRaises(ValueError):
            dspark_gamma_from_num_draft_tokens(0)

    def test_raises_for_negative_num_draft_tokens(self):
        """dspark_gamma_from_num_draft_tokens raises ValueError for negative input."""
        with self.assertRaises(ValueError):
            dspark_gamma_from_num_draft_tokens(-1)


class TestParseDSparkDraftConfig(CustomTestCase):
    def _make_config_dict(self, **overrides) -> dict:
        base = {
            "num_hidden_layers": 4,
            "num_target_layers": 28,
            "block_size": 7,
            "target_layer_ids": [6, 13, 20, 27],
            "markov_rank": 32,
            "markov_head_type": "vanilla",
            "mask_token_id": 1,
        }
        base.update(overrides)
        return base

    def test_parse_vanilla_markov_config(self):
        """parse_dspark_draft_config correctly extracts vanilla markov fields."""
        cfg = self._make_config_dict()
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.markov_rank, 32)
        self.assertEqual(result.markov_head_type, "vanilla")
        self.assertTrue(result.require_markov())

    def test_parse_gated_markov_config(self):
        """parse_dspark_draft_config accepts gated markov_head_type."""
        cfg = self._make_config_dict(markov_head_type="gated")
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.markov_head_type, "gated")

    def test_parse_rnn_markov_config(self):
        """parse_dspark_draft_config accepts rnn markov_head_type."""
        cfg = self._make_config_dict(markov_head_type="rnn")
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.markov_head_type, "rnn")

    def test_gamma_is_block_size(self):
        """resolve_gamma returns block_size from config."""
        cfg = self._make_config_dict(block_size=5)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.resolve_gamma(default=None), 5)

    def test_resolve_gamma_default_when_not_set(self):
        """resolve_gamma returns provided default when gamma is None in config."""
        cfg = self._make_config_dict()
        cfg.pop("block_size", None)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertIsNone(result.gamma)
        self.assertEqual(result.resolve_gamma(default=7), 7)

    def test_require_markov_true_when_rank_positive(self):
        """require_markov() returns True for markov_rank > 0."""
        cfg = self._make_config_dict(markov_rank=16)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertTrue(result.require_markov())

    def test_require_markov_false_when_rank_zero(self):
        """require_markov() returns False for markov_rank == 0."""
        cfg = self._make_config_dict(markov_rank=0)
        cfg.pop("markov_head_type", None)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertFalse(result.require_markov())

    def test_raises_for_unsupported_markov_head_type(self):
        """parse_dspark_draft_config raises ValueError for unknown markov_head_type."""
        cfg = self._make_config_dict(markov_head_type="transformer")
        with self.assertRaises(ValueError):
            parse_dspark_draft_config(draft_hf_config=cfg)

    def test_raises_for_missing_markov_head_type_when_rank_positive(self):
        """parse_dspark_draft_config raises ValueError when markov_head_type absent but rank > 0."""
        cfg = self._make_config_dict()
        cfg.pop("markov_head_type", None)
        with self.assertRaises(ValueError):
            parse_dspark_draft_config(draft_hf_config=cfg)

    def test_raises_for_negative_markov_rank(self):
        """parse_dspark_draft_config raises ValueError for markov_rank < 0."""
        cfg = self._make_config_dict(markov_rank=-1)
        with self.assertRaises(ValueError):
            parse_dspark_draft_config(draft_hf_config=cfg)

    def test_all_supported_head_types_are_parseable(self):
        """All SUPPORTED_DSPARK_MARKOV_HEAD_TYPES parse without error."""
        for head_type in SUPPORTED_DSPARK_MARKOV_HEAD_TYPES:
            cfg = self._make_config_dict(markov_head_type=head_type)
            result = parse_dspark_draft_config(draft_hf_config=cfg)
            self.assertEqual(result.markov_head_type, head_type)

    def test_dspark_config_subdict_overrides_top_level(self):
        """dspark_config sub-dict fields take precedence over top-level fields."""
        cfg = self._make_config_dict(markov_rank=8)
        cfg["dspark_config"] = {"markov_rank": 64, "markov_head_type": "vanilla"}
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.markov_rank, 64)

    def test_mask_token_id_is_parsed(self):
        """mask_token_id field is captured from config."""
        cfg = self._make_config_dict(mask_token_id=42)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.mask_token_id, 42)

    def test_target_layer_ids_are_parsed(self):
        """target_layer_ids list is captured correctly."""
        layer_ids = [3, 10, 17, 24]
        cfg = self._make_config_dict(target_layer_ids=layer_ids)
        result = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(result.target_layer_ids, layer_ids)


if __name__ == "__main__":
    unittest.main()
