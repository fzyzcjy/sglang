import statistics


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
