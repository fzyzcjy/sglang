import statistics

import numpy as np


def ols_resid_backfit(cells: list, mbin_w: int = 64):
    # Additive step-cost model T(bs, M) ~ bias + alpha(bs) + theta(M), M = bs + K
    # binned to mbin_w. Ordinary least squares over the two-way indicator design
    # (bias + per-bs + per-Mbin dummies, gauge alpha(min bs)=0, theta(min M)=0)
    # via a min-norm SVD solve, so the confounded null space (disjoint M ranges
    # across bs) resolves to small sensible offsets instead of the local minimum
    # an iterated median polish lands in. Returns (bias, alpha{bs}, theta{mbin},
    # residuals%, stats).
    def mbin(m):
        return round(m / mbin_w) * mbin_w

    bslist = sorted({c["bs"] for c in cells})
    mbins = sorted({mbin(c["M"]) for c in cells})
    bs_col = {b: i for i, b in enumerate(bslist[1:])}
    m_col = {m: i for i, m in enumerate(mbins[1:])}
    num_cols = 1 + len(bs_col) + len(m_col)

    design = np.zeros((len(cells), num_cols))
    target = np.array([c["T"] for c in cells], dtype=float)
    for row, c in enumerate(cells):
        design[row, 0] = 1.0
        if c["bs"] in bs_col:
            design[row, 1 + bs_col[c["bs"]]] = 1.0
        if mbin(c["M"]) in m_col:
            design[row, 1 + len(bs_col) + m_col[mbin(c["M"])]] = 1.0

    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    bias = float(beta[0])
    alpha = {bslist[0]: 0.0}
    for b in bslist[1:]:
        alpha[b] = float(beta[1 + bs_col[b]])
    theta = {mbins[0]: 0.0}
    for m in mbins[1:]:
        theta[m] = float(beta[1 + len(bs_col) + m_col[m]])

    resid = [c["T"] - (bias + alpha[c["bs"]] + theta[mbin(c["M"])]) for c in cells]
    rel = [abs(r) / c["T"] * 100 for r, c in zip(resid, cells)]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
    tbar = statistics.fmean(c["T"] for c in cells)
    ss_tot = sum((c["T"] - tbar) ** 2 for c in cells)
    r2 = 1.0 - sum(r * r for r in resid) / ss_tot if ss_tot > 0 else float("nan")

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
