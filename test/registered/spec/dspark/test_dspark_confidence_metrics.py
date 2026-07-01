import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_confidence_metrics import (
    PerPositionConfidenceMetrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _cpu_metrics(gamma: int) -> PerPositionConfidenceMetrics:
    return PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cpu"))


class TestPerPositionConfidenceMetrics(CustomTestCase):
    def test_perfectly_calibrated_has_low_ece(self):
        """Constant survival matching the Bernoulli target rate gives ECE ~ 0."""
        torch.manual_seed(0)
        n = 40000
        survival = torch.full((n, 1), 0.3, dtype=torch.float64)
        prefix_mask = (torch.rand(n, 1) < 0.3).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        row = metrics.compute()[0]
        self.assertLess(row["ece"], 0.03)
        self.assertAlmostEqual(row["pred_mean"], 0.3, places=4)

    def test_overconfident_has_high_ece_and_pred_above_target(self):
        """Survival far above the target rate yields large ECE and pred_mean > target_mean."""
        torch.manual_seed(0)
        n = 40000
        survival = torch.full((n, 1), 0.9, dtype=torch.float64)
        prefix_mask = (torch.rand(n, 1) < 0.3).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        row = metrics.compute()[0]
        self.assertGreater(row["ece"], 0.4)
        self.assertGreater(row["pred_mean"], row["target_mean"])

    def test_separable_scores_give_auc_near_one(self):
        """Positives with high survival and negatives with low survival give AUC ~ 1."""
        torch.manual_seed(0)
        n = 20000
        pos = torch.rand(n, 1) * 0.3 + 0.7
        neg = torch.rand(n, 1) * 0.3
        survival = torch.cat([pos, neg], dim=0)
        prefix_mask = torch.cat([torch.ones(n, 1), torch.zeros(n, 1)], dim=0)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        self.assertGreater(metrics.compute()[0]["auc"], 0.99)

    def test_random_scores_give_auc_near_half(self):
        """Survival independent of the label gives AUC ~ 0.5."""
        torch.manual_seed(0)
        n = 40000
        survival = torch.rand(n, 1)
        prefix_mask = (torch.rand(n, 1) < 0.5).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        auc = metrics.compute()[0]["auc"]
        self.assertGreater(auc, 0.45)
        self.assertLess(auc, 0.55)

    def test_batched_update_matches_per_sample_update(self):
        """A single [bs, gamma] update accumulates the same histograms as row-by-row updates."""
        torch.manual_seed(0)
        bs, gamma = 32, 5
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        batched = _cpu_metrics(gamma=gamma)
        batched.update(survival=survival, prefix_mask=prefix_mask)

        per_sample = _cpu_metrics(gamma=gamma)
        for row_idx in range(bs):
            per_sample.update(
                survival=survival[row_idx : row_idx + 1],
                prefix_mask=prefix_mask[row_idx : row_idx + 1],
            )

        for name in (
            "coarse_count",
            "coarse_pred",
            "coarse_target",
            "fine_pos",
            "fine_neg",
            "brier_num",
        ):
            self.assertTrue(
                torch.allclose(
                    getattr(batched, name), getattr(per_sample, name), atol=1e-9
                ),
                msg=name,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_on_device_accumulation_matches_cpu(self):
        """Histograms accumulated on GPU reduce to the same metrics as the CPU reference."""
        torch.manual_seed(0)
        bs, gamma = 24, 4
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        cpu = _cpu_metrics(gamma=gamma)
        cpu.update(survival=survival, prefix_mask=prefix_mask)

        gpu = PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cuda"))
        gpu.update(survival=survival.cuda(), prefix_mask=prefix_mask.cuda())

        for cpu_row, gpu_row in zip(cpu.compute(), gpu.compute()):
            for key in ("ece", "auc", "brier", "pred_mean", "target_mean"):
                self.assertAlmostEqual(cpu_row[key], gpu_row[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
