import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator_offline import (
    OfflineBlockAcceptEstimateRecorder,
)
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator_online import (
    OnlineBlockAcceptEstimateRecorder as BlockAcceptEstimateRecorder,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_GAMMA = 3
_VOCAB = 8


class _FakeLayout:
    def __init__(self, verify_lens: torch.Tensor):
        self.verify_lens = verify_lens


def _reference_logprob(
    logits_row: torch.Tensor, token: int, temperature: float
) -> float:
    scaled = logits_row.to(torch.float32) / temperature
    return float(scaled[token] - torch.logsumexp(scaled, dim=-1))


def _make_recorder(tmp_dir: str) -> tuple[BlockAcceptEstimateRecorder, Path]:
    path = Path(tmp_dir) / "estimate.jsonl"
    recorder = BlockAcceptEstimateRecorder(path=str(path), gamma=_GAMMA, device="cpu")
    return recorder, path


def _observe(
    recorder: BlockAcceptEstimateRecorder,
    *,
    forward_ct: int,
    rid: str,
    drafts: list[int],
    corrected_logits: torch.Tensor,
    target_logits: torch.Tensor,
    verify_len: int,
    correct_len: int,
    bonus: int,
    seq_len: int,
    cap_trim: int = 0,
    temperature: float = 1.0,
) -> None:
    recorder.observe_verify_step(
        forward_ct=forward_ct,
        rids=[rid],
        draft_tokens=torch.tensor([drafts], dtype=torch.int64),
        corrected_logits=corrected_logits.unsqueeze(0),
        draft_temperatures=torch.tensor([temperature], dtype=torch.float32),
        greedy_mask=torch.tensor([False]),
        target_logits=target_logits,
        target_temperatures=torch.tensor([[temperature]], dtype=torch.float32),
        truncated_sampling_mask=None,
        logits_adjustments_are_noop=True,
        correct_len=torch.tensor([correct_len], dtype=torch.int32),
        cap_trim_lens=torch.tensor([cap_trim], dtype=torch.int32),
        bonus=torch.tensor([bonus], dtype=torch.int64),
        prefix_lens=torch.tensor([seq_len], dtype=torch.int64),
        layout=_FakeLayout(torch.tensor([verify_len], dtype=torch.int32)),
    )


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestBlockAcceptEstimateRecorder(CustomTestCase):
    def test_exact_block_when_rejected_inside_window(self):
        """A block rejected inside the verify window emits an exact record without q/pg."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected = torch.randn(_GAMMA, _VOCAB)
            target = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=[1, 2, 3],
                corrected_logits=corrected,
                target_logits=target,
                verify_len=3,
                correct_len=1,
                bonus=5,
                seq_len=10,
            )
            recorder._file.flush()

            records = _read_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["w"], 2)
            self.assertEqual(records[0]["cl"], 1)
            self.assertNotIn("q_lp", records[0])
            self.assertNotIn("pg", records[0])

    def test_censored_block_gathers_q_and_same_step_bonus_row_p(self):
        """A cap-hit block records trimmed q values and resolves its first trimmed offset against the bonus row."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected = torch.randn(_GAMMA, _VOCAB)
            target = torch.randn((_GAMMA + 1), _VOCAB)
            drafts = [1, 2, 3]
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=drafts,
                corrected_logits=corrected,
                target_logits=target,
                verify_len=2,
                correct_len=1,
                bonus=2,
                seq_len=10,
            )
            recorder._file.flush()

            records = _read_records(path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["w"], 1)
            self.assertEqual(record["cl"], 1)
            self.assertEqual(record["trimmed_tokens"], [2, 3])
            self.assertEqual(len(record["q_lp"]), 2)
            self.assertAlmostEqual(
                record["q_lp"][0],
                _reference_logprob(corrected[1], 2, 1.0),
                places=4,
            )
            self.assertAlmostEqual(
                record["q_lp"][1],
                _reference_logprob(corrected[2], 3, 1.0),
                places=4,
            )
            self.assertEqual(len(record["pg"]), 1)
            src_fct, offset, p_lp, draft_token, realized_token = record["pg"][0]
            self.assertEqual((src_fct, offset), (1, 2))
            self.assertEqual(draft_token, 2)
            self.assertEqual(realized_token, 2)
            self.assertAlmostEqual(
                p_lp, _reference_logprob(target[1], 2, 1.0), places=4
            )

    def test_pending_block_resolves_in_later_step(self):
        """A trimmed offset beyond this step's commits resolves against a later step's committed row."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected1 = torch.randn(_GAMMA, _VOCAB)
            target1 = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=[1, 2, 3],
                corrected_logits=corrected1,
                target_logits=target1,
                verify_len=2,
                correct_len=1,
                bonus=2,
                seq_len=10,
            )

            corrected2 = torch.randn(_GAMMA, _VOCAB)
            target2 = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=2,
                rid="r0",
                drafts=[3, 6, 7],
                corrected_logits=corrected2,
                target_logits=target2,
                verify_len=4,
                correct_len=3,
                bonus=4,
                seq_len=12,
            )
            recorder._file.flush()

            records = _read_records(path)
            self.assertEqual(len(records), 2)
            step2 = records[1]
            self.assertEqual(len(step2["pg"]), 1)
            src_fct, offset, p_lp, draft_token, realized_token = step2["pg"][0]
            self.assertEqual((src_fct, offset), (1, 3))
            self.assertEqual(draft_token, 3)
            self.assertEqual(realized_token, 3)
            self.assertAlmostEqual(
                p_lp, _reference_logprob(target2[0], 3, 1.0), places=4
            )
            self.assertEqual(recorder._states["r0"].pending, [])

    def test_divergence_drops_block_after_final_gather(self):
        """After a trimmed token mismatches the realized token, the block stops producing gathers."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected1 = torch.randn(_GAMMA, _VOCAB)
            target1 = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=[1, 2, 3],
                corrected_logits=corrected1,
                target_logits=target1,
                verify_len=2,
                correct_len=1,
                bonus=6,
                seq_len=10,
            )
            recorder._file.flush()

            records = _read_records(path)
            src_fct, offset, p_lp, draft_token, realized_token = records[0]["pg"][0]
            self.assertEqual(draft_token, 2)
            self.assertEqual(realized_token, 6)
            self.assertEqual(recorder._states["r0"].pending, [])

    def test_temperature_scales_logprobs(self):
        """Recorded logprobs use softmax(logits/T) with the per-request temperature."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected = torch.randn(_GAMMA, _VOCAB)
            target = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=[1, 2, 3],
                corrected_logits=corrected,
                target_logits=target,
                verify_len=2,
                correct_len=1,
                bonus=2,
                seq_len=10,
                temperature=0.7,
            )
            recorder._file.flush()

            record = _read_records(path)[0]
            self.assertAlmostEqual(
                record["q_lp"][0],
                _reference_logprob(corrected[1], 2, 0.7),
                places=4,
            )
            self.assertAlmostEqual(
                record["pg"][0][2],
                _reference_logprob(target[1], 2, 0.7),
                places=4,
            )

    def test_greedy_row_is_skipped_but_seq_len_bookkeeping_advances(self):
        """A greedy request emits no record while its expected-seq-len bookkeeping still advances."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected = torch.randn(_GAMMA, _VOCAB)
            target = torch.randn((_GAMMA + 1), _VOCAB)
            recorder.observe_verify_step(
                forward_ct=1,
                rids=["r0"],
                draft_tokens=torch.tensor([[1, 2, 3]], dtype=torch.int64),
                corrected_logits=corrected.unsqueeze(0),
                draft_temperatures=torch.tensor([1.0]),
                greedy_mask=torch.tensor([True]),
                target_logits=target,
                target_temperatures=torch.tensor([[1.0]]),
                truncated_sampling_mask=None,
                logits_adjustments_are_noop=True,
                correct_len=torch.tensor([2], dtype=torch.int32),
                cap_trim_lens=torch.tensor([0], dtype=torch.int32),
                bonus=torch.tensor([5], dtype=torch.int64),
                prefix_lens=torch.tensor([10], dtype=torch.int64),
                layout=_FakeLayout(torch.tensor([4], dtype=torch.int32)),
            )
            recorder._file.flush()

            self.assertEqual(path.read_text(), "")
            self.assertEqual(recorder._states["r0"].expected_seq_len, 13)

    def test_seq_len_discontinuity_drops_pending_blocks(self):
        """A seq-len jump (retraction signature) drops pending blocks instead of gathering at wrong rows."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected1 = torch.randn(_GAMMA, _VOCAB)
            target1 = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=1,
                rid="r0",
                drafts=[1, 2, 3],
                corrected_logits=corrected1,
                target_logits=target1,
                verify_len=2,
                correct_len=1,
                bonus=2,
                seq_len=10,
            )
            self.assertEqual(len(recorder._states["r0"].pending), 1)

            corrected2 = torch.randn(_GAMMA, _VOCAB)
            target2 = torch.randn((_GAMMA + 1), _VOCAB)
            _observe(
                recorder,
                forward_ct=2,
                rid="r0",
                drafts=[3, 6, 7],
                corrected_logits=corrected2,
                target_logits=target2,
                verify_len=4,
                correct_len=3,
                bonus=4,
                seq_len=11,
            )
            recorder._file.flush()

            records = _read_records(path)
            self.assertNotIn("pg", records[1])
            self.assertEqual(recorder._states["r0"].pending, [])
            self.assertEqual(recorder._discontinuity_drop_ct, 1)

    def test_truncated_sampling_row_is_excluded_while_clean_row_records(self):
        """A mixed batch records the pure-temperature row and excludes only the truncated-sampling row."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, path = _make_recorder(tmp)
            corrected = torch.randn(2, _GAMMA, _VOCAB)
            target = torch.randn(2 * (_GAMMA + 1), _VOCAB)
            recorder.observe_verify_step(
                forward_ct=1,
                rids=["r_clean", "r_top_p"],
                draft_tokens=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
                corrected_logits=corrected,
                draft_temperatures=torch.tensor([1.0, 1.0]),
                greedy_mask=torch.tensor([False, False]),
                target_logits=target,
                target_temperatures=torch.tensor([[1.0], [1.0]]),
                truncated_sampling_mask=torch.tensor([False, True]),
                logits_adjustments_are_noop=True,
                correct_len=torch.tensor([1, 1], dtype=torch.int32),
                cap_trim_lens=torch.tensor([0, 0], dtype=torch.int32),
                bonus=torch.tensor([2, 7], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                layout=_FakeLayout(torch.tensor([2, 2], dtype=torch.int32)),
            )
            recorder._file.flush()

            records = _read_records(path)
            self.assertEqual([r["rid"] for r in records], ["r_clean"])
            self.assertEqual(recorder._skipped_step_ct, 0)
            self.assertEqual(recorder._states["r_top_p"].pending, [])
            self.assertEqual(recorder._states["r_top_p"].expected_seq_len, 22)


def _offline_estimate(path: Path, gamma: int) -> tuple[float, float, int]:
    from collections import defaultdict

    blocks: list[dict] = []
    gathers: dict[tuple, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        blocks.append(rec)
        for src_fct, offset, p_lp, draft_token, realized_token in rec.get("pg", []):
            gathers[(rec["rid"], src_fct)].append(
                [offset, p_lp, draft_token, realized_token]
            )
    los: list[float] = []
    his: list[float] = []
    for rec in blocks:
        cl, w = rec["cl"], rec["w"]
        if "q_lp" not in rec:
            los.append(cl + 1.0)
            his.append(cl + 1.0)
            continue
        q_lps = rec["q_lp"]
        entries = {e[0]: e for e in gathers.get((rec["rid"], rec["fct"]), [])}
        base, prod, lo_extra, tail = w + 1.0, 1.0, 0.0, 0.0
        for offset in range(w + 1, gamma + 1):
            entry = entries.get(offset)
            if entry is None:
                tail = prod * (gamma - offset + 1)
                break
            _, p_lp, draft_token, realized_token = entry
            a = min(1.0, math.exp(p_lp - q_lps[offset - w - 1]))
            prod *= a
            lo_extra += prod
            if draft_token != realized_token:
                if offset < gamma:
                    tail = prod * (gamma - offset)
                break
        los.append(base + lo_extra)
        his.append(base + lo_extra + tail)
    n = len(los)
    return sum(los) / n, sum(his) / n, n


class TestOnlineCeilingEstimate(CustomTestCase):
    def test_online_estimate_matches_offline_aggregation(self):
        """The online rolling ceiling estimate equals the offline analyzer over the same blocks."""
        bs, steps = 4, 14
        gen = torch.Generator().manual_seed(11)
        seq = [50 + 3 * b for b in range(bs)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "est.jsonl"
            recorder = BlockAcceptEstimateRecorder(
                path=str(path),
                gamma=_GAMMA,
                device="cpu",
                online_log_interval=3,
                online_window_steps=2,
            )
            for t in range(steps):
                verify_lens, correct_lens, drafts, bonus, prefix = [], [], [], [], []
                for b in range(bs):
                    vl = int(torch.randint(1, _GAMMA + 2, (1,), generator=gen))
                    window = vl - 1
                    cl = int(torch.randint(0, window + 1, (1,), generator=gen))
                    row = torch.randint(0, _VOCAB, (_GAMMA,), generator=gen).tolist()
                    if cl < _GAMMA and int(torch.randint(0, 2, (1,), generator=gen)):
                        bt = row[cl]
                    else:
                        bt = int(torch.randint(0, _VOCAB, (1,), generator=gen))
                    verify_lens.append(vl)
                    correct_lens.append(cl)
                    drafts.append(row)
                    bonus.append(bt)
                    prefix.append(seq[b])
                    seq[b] += cl + 1
                recorder.observe_verify_step(
                    forward_ct=t + 1,
                    rids=[f"r{b}" for b in range(bs)],
                    draft_tokens=torch.tensor(drafts, dtype=torch.int64),
                    corrected_logits=torch.randn(bs, _GAMMA, _VOCAB, generator=gen),
                    draft_temperatures=torch.ones(bs),
                    greedy_mask=torch.zeros(bs, dtype=torch.bool),
                    target_logits=torch.randn(bs * (_GAMMA + 1), _VOCAB, generator=gen),
                    target_temperatures=torch.ones(bs),
                    truncated_sampling_mask=None,
                    logits_adjustments_are_noop=True,
                    correct_len=torch.tensor(correct_lens, dtype=torch.int32),
                    cap_trim_lens=torch.tensor(
                        [_GAMMA - (v - 1) for v in verify_lens], dtype=torch.int32
                    ),
                    bonus=torch.tensor(bonus, dtype=torch.int64),
                    prefix_lens=torch.tensor(prefix, dtype=torch.int64),
                    layout=_FakeLayout(torch.tensor(verify_lens, dtype=torch.int32)),
                )
            recorder.drain_pending_online()
            recorder._file.flush()

            off_lo, off_hi, off_n = _offline_estimate(path, _GAMMA)
            snap = recorder.online_estimate()
            self.assertIsNotNone(snap)
            self.assertEqual(snap.cumulative_blocks, off_n)
            self.assertAlmostEqual(snap.cumulative_lo, off_lo, places=6)
            self.assertAlmostEqual(snap.cumulative_hi, off_hi, places=6)
            self.assertGreater(off_n, bs)
            self.assertLessEqual(snap.window_blocks, snap.cumulative_blocks)

    def test_online_window_evicts_forward_passes_outside_horizon(self):
        """The rolling window only keeps blocks finalized within the last window_steps forward passes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "est.jsonl"
            recorder = BlockAcceptEstimateRecorder(
                path=str(path),
                gamma=_GAMMA,
                device="cpu",
                online_log_interval=1,
                online_window_steps=3,
            )
            target = torch.randn((_GAMMA + 1), _VOCAB)
            seq = 10
            for t in range(10):
                cl = t % 3
                _observe(
                    recorder,
                    forward_ct=t + 1,
                    rid="r0",
                    drafts=[1, 2, 3],
                    corrected_logits=torch.randn(_GAMMA, _VOCAB),
                    target_logits=target,
                    verify_len=_GAMMA + 1,
                    correct_len=cl,
                    bonus=5,
                    seq_len=seq,
                )
                seq += cl + 1
            snap = recorder.online_estimate()
            self.assertIsNotNone(snap)
            self.assertLessEqual(snap.window_horizon, 3)
            self.assertLessEqual(snap.window_blocks, 3)
            self.assertEqual(snap.cumulative_blocks, 10)

    def test_online_estimate_none_without_observations(self):
        """With no observed blocks yet, online_estimate and the log suffix are both None."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = _make_recorder(tmp)
            self.assertIsNone(recorder.online_estimate())
            self.assertIsNone(recorder.estimate_log_suffix())

    def test_estimate_log_suffix_reports_cumulative(self):
        """After observations the log suffix reports the cumulative mid estimate and bracket."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = _make_recorder(tmp)
            target = torch.randn((_GAMMA + 1), _VOCAB)
            seq = 10
            for t in range(6):
                cl = t % 3
                _observe(
                    recorder,
                    forward_ct=t + 1,
                    rid="r0",
                    drafts=[1, 2, 3],
                    corrected_logits=torch.randn(_GAMMA, _VOCAB),
                    target_logits=target,
                    verify_len=_GAMMA + 1,
                    correct_len=cl,
                    bonus=5,
                    seq_len=seq,
                )
                seq += cl + 1
            suffix = recorder.estimate_log_suffix()
            self.assertIsNotNone(suffix)
            self.assertIn("est uncap acc len", suffix)


class TestOfflineRecorderMatchesOnline(CustomTestCase):
    def test_offline_dump_matches_online_dump_over_random_stream(self):
        """The synchronous offline recorder writes byte-identical JSONL to the async online recorder."""
        bs, steps = 4, 16
        gen = torch.Generator().manual_seed(23)
        with tempfile.TemporaryDirectory() as tmp:
            online_path = Path(tmp) / "online.jsonl"
            offline_path = Path(tmp) / "offline.jsonl"
            online = BlockAcceptEstimateRecorder(
                path=str(online_path), gamma=_GAMMA, device="cpu"
            )
            offline = OfflineBlockAcceptEstimateRecorder(
                path=str(offline_path), gamma=_GAMMA
            )
            seq = [40 + 5 * b for b in range(bs)]
            for t in range(steps):
                verify_lens, correct_lens, drafts, bonus, prefix = [], [], [], [], []
                for b in range(bs):
                    vl = int(torch.randint(1, _GAMMA + 2, (1,), generator=gen))
                    window = vl - 1
                    cl = int(torch.randint(0, window + 1, (1,), generator=gen))
                    row = torch.randint(0, _VOCAB, (_GAMMA,), generator=gen).tolist()
                    if cl < _GAMMA and int(torch.randint(0, 2, (1,), generator=gen)):
                        bt = row[cl]
                    else:
                        bt = int(torch.randint(0, _VOCAB, (1,), generator=gen))
                    verify_lens.append(vl)
                    correct_lens.append(cl)
                    drafts.append(row)
                    bonus.append(bt)
                    prefix.append(seq[b])
                    seq[b] += cl + 1
                kwargs = dict(
                    forward_ct=t + 1,
                    rids=[f"r{b}" for b in range(bs)],
                    draft_tokens=torch.tensor(drafts, dtype=torch.int64),
                    corrected_logits=torch.randn(bs, _GAMMA, _VOCAB, generator=gen),
                    draft_temperatures=torch.ones(bs),
                    greedy_mask=torch.zeros(bs, dtype=torch.bool),
                    target_logits=torch.randn(bs * (_GAMMA + 1), _VOCAB, generator=gen),
                    target_temperatures=torch.ones(bs),
                    truncated_sampling_mask=None,
                    logits_adjustments_are_noop=True,
                    correct_len=torch.tensor(correct_lens, dtype=torch.int32),
                    cap_trim_lens=torch.tensor(
                        [_GAMMA - (v - 1) for v in verify_lens], dtype=torch.int32
                    ),
                    bonus=torch.tensor(bonus, dtype=torch.int64),
                    prefix_lens=torch.tensor(prefix, dtype=torch.int64),
                    layout=_FakeLayout(torch.tensor(verify_lens, dtype=torch.int32)),
                )
                online.observe_verify_step(**kwargs)
                offline.observe_verify_step(**kwargs)
            online.flush()
            offline.flush()

            online_recs = _read_records(online_path)
            offline_recs = _read_records(offline_path)
            self.assertEqual(len(online_recs), len(offline_recs))
            for a, b in zip(online_recs, offline_recs):
                self.assertEqual(a.keys(), b.keys())
                self.assertEqual(a["rid"], b["rid"])
                self.assertEqual(a["fct"], b["fct"])
                self.assertEqual(a["w"], b["w"])
                self.assertEqual(a["cl"], b["cl"])
                if "pg" in a:
                    self.assertEqual(len(a["pg"]), len(b["pg"]))
                    for ea, eb in zip(a["pg"], b["pg"]):
                        self.assertEqual(ea[0], eb[0])
                        self.assertEqual(ea[1], eb[1])
                        self.assertAlmostEqual(ea[2], eb[2], places=5)
                        self.assertEqual(ea[3], eb[3])
                        self.assertEqual(ea[4], eb[4])


if __name__ == "__main__":
    unittest.main()
