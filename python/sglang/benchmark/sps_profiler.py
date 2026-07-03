"""Off-diagonal DSpark SPS profiler: fit T(bs, K) = bias + alpha(bs) + theta(M).

Drives a RUNNING DSpark server (compact + SGLANG_RECORD_STEP_TIME=1). For each
(K-fraction, batch size) cell it:
  1. pins the verify budget K over HTTP:
     POST /set_internal_state {"server_args": {"dspark_force_budget_frac": f}}
  2. sends ONE sglang.benchmark.one_batch_server request (fixed batch, lockstep
     decode, no prefill churn), which returns the strict per-step iter_time
     from the server's step_time_dict.
Then it back-fits the additive cost table (two 1D lookups + a bias, zero
parametric form) and reports the fit residual (the additivity/separability
check) and each cell's step-time noise.

Example:
  python3 -m sglang.benchmark.sps_profiler --base-url http://127.0.0.1:31310 \
    --fracs 0.42 0.5 0.6 0.7 0.8 1.0 --bs 4 8 16 32 64 96 128 160 192 224 256 \
    --input-len 512 --output-len 256 --out sps_table_dsv4_b300.json
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time

import requests

DEFAULT_TIMEOUT = 60


def get_gamma(base_url: str) -> int:
    info = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT).json()
    st = (info.get("internal_states") or [{}])[0]
    rec = st.get("dspark_sps_record") or {}
    g = rec.get("verify_num_draft_tokens") or st.get("speculative_num_draft_tokens")
    if not g:
        raise RuntimeError(
            "Could not read verify_num_draft_tokens from /server_info; is this a "
            "DSpark server with SGLANG_DSPARK_ENABLE_SPS_RECORD=1?"
        )
    return int(g)


def set_frac(base_url: str, frac) -> None:
    r = requests.post(
        base_url + "/set_internal_state",
        json={"server_args": {"dspark_force_budget_frac": frac}},
        timeout=DEFAULT_TIMEOUT,
    ).json()
    # /set_internal_state fans out per rank -> a list of the `updated` flags
    # (bare bools), or [{"updated": ...}] on older builds; accept both.
    outs = r if isinstance(r, list) else [r]

    def _ok(o):
        return o.get("updated") if isinstance(o, dict) else bool(o)

    if not outs or not all(_ok(o) for o in outs):
        raise RuntimeError(f"set dspark_force_budget_frac={frac} rejected: {r}")


def one_batch(base_url: str, bs: int, input_len: int, output_len: int, temp: float):
    with tempfile.NamedTemporaryFile("r+", suffix=".jsonl", delete=False) as f:
        out = f.name
    requests.post(base_url + "/flush_cache", timeout=DEFAULT_TIMEOUT)
    cmd = [
        sys.executable,
        "-m",
        "sglang.benchmark.one_batch_server",
        "--model",
        "None",
        "--base-url",
        base_url,
        "--batch-size",
        str(bs),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--temperature",
        str(temp),
        "--result-filename",
        out,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    rows = [json.loads(l) for l in open(out) if l.strip()]
    return rows[-1] if rows else None


def ols_resid_backfit(cells: list, mbin_w: int = 64):
    # Zero-parameter additive lookup: T(bs, M) ~ bias + alpha(bs) + theta(M),
    # M = bs + K binned. Iterated median back-fit (median polish). Gauge:
    # alpha(min bs)=0, theta(min M)=0. Returns (bias, alpha{bs}, theta{mbin}, residuals%).
    def mbin(m):
        return round(m / mbin_w) * mbin_w

    bslist = sorted({c["bs"] for c in cells})
    mbins = sorted({mbin(c["M"]) for c in cells})
    alpha = {b: 0.0 for b in bslist}
    theta = {m: 0.0 for m in mbins}
    bias = statistics.median(c["T"] for c in cells)
    for _ in range(200):
        for b in bslist:
            alpha[b] = statistics.median(
                c["T"] - bias - theta[mbin(c["M"])] for c in cells if c["bs"] == b
            )
        bias += alpha[bslist[0]]
        s0 = alpha[bslist[0]]
        for b in bslist:
            alpha[b] -= s0
        for m in mbins:
            theta[m] = statistics.median(
                c["T"] - bias - alpha[c["bs"]] for c in cells if mbin(c["M"]) == m
            )
        bias += theta[mbins[0]]
        t0 = theta[mbins[0]]
        for m in mbins:
            theta[m] -= t0
    # Goodness of fit + per-component standard error.
    resid = [c["T"] - (bias + alpha[c["bs"]] + theta[mbin(c["M"])]) for c in cells]
    rel = [abs(r) / c["T"] * 100 for r, c in zip(resid, cells)]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
    tbar = statistics.fmean(c["T"] for c in cells)
    ss_tot = sum((c["T"] - tbar) ** 2 for c in cells)
    r2 = 1.0 - sum(r * r for r in resid) / ss_tot if ss_tot > 0 else float("nan")

    # SE of each probe's fitted level ~ stdev of its member residuals / sqrt(n).
    def probe_se(pred):
        se = {}
        for key in sorted({pred(c) for c in cells}):
            rs = [r for r, c in zip(resid, cells) if pred(c) == key]
            se[key] = (statistics.pstdev(rs) / (len(rs) ** 0.5)) if len(rs) > 1 else 0.0
        return se

    stats = {
        "rms_ms": rms * 1e3,
        "r2": r2,
        "n": len(cells),
        "alpha_se": probe_se(lambda c: c["bs"]),
        "theta_se": probe_se(lambda c: mbin(c["M"])),
    }
    return bias, alpha, theta, rel, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument(
        "--fracs", type=float, nargs="+", default=[0.42, 0.5, 0.6, 0.7, 0.8, 1.0]
    )
    p.add_argument(
        "--bs",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
    )
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", required=True, help="SpsAdditiveCostTable json path")
    p.add_argument("--cells-out", default="", help="optional per-cell jsonl dump")
    a = p.parse_args()

    gamma = get_gamma(a.base_url)  # = verify_num_draft_tokens (gamma+1)
    total = len(a.fracs) * len(a.bs)
    t0 = time.monotonic()
    print(
        f"[sps] verify_num_draft_tokens={gamma}  grid={len(a.fracs)}fracs x "
        f"{len(a.bs)}bs = {total} cells"
    )
    cells = []
    done = 0
    for fi, frac in enumerate(a.fracs):
        print(f"[sps] ===== frac={frac}  ({fi+1}/{len(a.fracs)}) =====")
        for bs in a.bs:
            done += 1
            el = time.monotonic() - t0
            eta = (el / done) * (total - done)
            # Re-pin K per cell: set_internal_state also resets the cumulative
            # spec-accept counters, so acc_length reads back as THIS cell's
            # average (used to derive iter_time = decode_wall / (out_len/acc)).
            set_frac(a.base_url, frac)
            r = one_batch(a.base_url, bs, a.input_len, a.output_len, a.temperature)
            budget = int(frac * bs * (gamma - 1))  # K above the 1-token floor
            M = bs + budget
            if not r or r.get("iter_time", -1) <= 0:
                print(
                    f"[sps] [{done:>3}/{total}] frac={frac} bs={bs}: "
                    f"no iter_time (skipped)  elapsed={el/60:.1f}m eta={eta/60:.1f}m"
                )
                continue
            cell = {
                "bs": bs,
                "frac": frac,
                "M": M,
                "T": r["iter_time"],
                "acc_length": r.get("acc_length"),
                "n_steps": r.get("n_decode_steps"),
            }
            cells.append(cell)
            if a.cells_out:
                open(a.cells_out, "a").write(json.dumps(cell) + "\n")
            print(
                f"[sps] [{done:>3}/{total}] frac={frac} bs={bs:>4} M={M:>5} "
                f"iter={r['iter_time']*1e3:7.3f}ms acc={r.get('acc_length')} "
                f"n={r.get('n_decode_steps')}  elapsed={el/60:.1f}m eta={eta/60:.1f}m"
            )

    if len(cells) < 4:
        sys.exit("[sps] too few cells to fit")

    bias, alpha, theta, rel, stats = ols_resid_backfit(cells)
    bslist = sorted(alpha)
    mbins = sorted(theta)
    print(f"\n[sps] === additive fit  T = bias + alpha(bs) + theta(M) ===")
    print(f"  n_cells={stats['n']}  bias={bias*1e3:.2f}ms")
    print(
        f"  GOODNESS OF FIT: R^2={stats['r2']:.4f}  RMS residual={stats['rms_ms']:.3f}ms  "
        f"| rel-resid median={statistics.median(rel):.2f}% "
        f"p90={sorted(rel)[int(0.9*len(rel))-1]:.2f}% max={max(rel):.2f}%"
    )
    print(
        "  alpha(bs) ms (+-SE): "
        + " ".join(
            f"{b}:{alpha[b]*1e3:.1f}+-{stats['alpha_se'][b]*1e3:.2f}" for b in bslist
        )
    )
    print(
        "  theta(M)  ms (+-SE): "
        + " ".join(
            f"{m}:{theta[m]*1e3:.1f}+-{stats['theta_se'][m]*1e3:.2f}" for m in mbins
        )
    )

    table = {
        "bias_seconds": bias,
        "bs_probes": bslist,
        "alpha_seconds": [alpha[b] for b in bslist],
        "m_probes": mbins,  # indexed on M = bs + K (total verify tokens)
        "theta_seconds": [theta[m] for m in mbins],
    }
    with open(a.out, "w") as f:
        json.dump(table, f, indent=1)
    print(f"\n[sps] wrote SpsAdditiveCostTable -> {a.out}")


if __name__ == "__main__":
    main()
