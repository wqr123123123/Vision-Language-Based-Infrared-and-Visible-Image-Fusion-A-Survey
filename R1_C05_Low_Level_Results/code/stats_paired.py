#!/usr/bin/env python3
"""Paired statistics and average tie ranks for the 4.1 fusion benchmark.

Reads each method's per-image metrics.csv from the result tree, aligns methods
on common image IDs, and writes:

  stats/descriptive.csv                 per method/dataset/metric n/mean/sd/CI
  stats/paired_pairwise_<dataset>.csv   paired mean difference, bootstrap CI,
                                        sign-flip permutation p, Holm-adjusted p,
                                        Cohen's d_z, n_common
  stats/ranks_<dataset>.csv             average tie ranks and s = 1-(r-1)/(M-1)

Methods are only compared on their common image IDs, never on their own subsets.
Deterministic: bootstrap seed 20261026, 10,000 resamples both procedures.
"""
import argparse, os
from pathlib import Path

import numpy as np, pandas as pd

REV = Path(__file__).resolve().parent.parent
# Per-image metrics come from the authoritative table (SF already corrected for
# every method), not from the result directories, which mix corrected SF in the
# rerun methods with the legacy uint8 SF everywhere else.
RES = REV / 'per_image'
OUT = Path.cwd() / 'paired_results'
# MetaFusion is ranked now that its fused images exist for all three datasets
# (code/run_metafusion_4_1.py); only the 0.5x Text-DiFuse modulated variant is out.
EXCLUDE = {'Text-DiFuse-Modulated'}
METRICS = ['EN', 'SD', 'SF', 'AG', 'MI', 'SSIM', 'CC', 'SCD']
DATASETS = ['TNO', 'LLVIP', 'RoadScene']
SEED = 20261026
N_BOOT = 10000
N_PERM = 10000
MIN_N = 30


def load(dataset):
    """Return {method: pd.DataFrame(index=image base)} for one dataset."""
    out = {}
    for f in sorted((RES).glob(f'*/{dataset}.csv')):
        if f.parent.name in EXCLUDE:
            continue
        df = pd.read_csv(f)
        if 'Image' not in df.columns:
            continue
        df['base'] = df['Image'].astype(str).map(lambda x: os.path.splitext(os.path.basename(x))[0])
        df = df.drop_duplicates('base').set_index('base')
        out[f.parent.name] = df
    return out


# Resample matrices are generated once per sample size and reused across tests
# (common random numbers). This is deterministic and keeps 10,000 x 10,000
# resamples affordable; it does not change any test's null distribution.
_SIGN, _IDX = {}, {}


def _sign(n):
    if n not in _SIGN:
        _SIGN[n] = np.random.default_rng(SEED + n).choice([-1.0, 1.0], size=(N_PERM, n))
    return _SIGN[n]


def _idx(n):
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(SEED + 7919 + n).integers(0, n, size=(N_BOOT, n))
    return _IDX[n]


def boot_ci(x, rng=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    means = x[_idx(n)].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_p(d, rng=None):
    """Two-sided sign-flip permutation p-value for mean(d)=0."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    stats = np.abs(_sign(n) @ d) / n
    return float((np.sum(stats >= obs) + 1) / (N_PERM + 1))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def average_ranks(df, metric, ascending=True):
    """Average tie rank (1 = best) and s = 1-(r-1)/(M-1)."""
    name = df['method'].astype(str).values if 'method' in df else None  # noqa: F841
    vals = df.set_index('method')[metric]
    r = vals.rank(ascending=ascending, method='average')
    M = len(r)
    s = 1 - (r - 1) / (M - 1) if M > 1 else r * 0
    return pd.DataFrame({'metric': metric, 'method': r.index, 'mean_value': vals.values,
                         'rank': r.values, 's': s.values, 'M': M})


def main():
    global N_BOOT, N_PERM
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--perm', type=int, default=N_PERM)
    ap.add_argument('--min-n', type=int, default=MIN_N)
    a = ap.parse_args()
    N_BOOT, N_PERM = a.boot, a.perm
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    desc_rows, rank_rows, skip_rows = [], [], []
    for ds in DATASETS:
        data = load(ds)
        if not data:
            continue
        # descriptive + ranks
        for m in METRICS:
            vals = {name: df[m].dropna() for name, df in data.items() if m in df.columns}
            t = pd.DataFrame([{'method': k, 'mean': v.mean(), 'n': len(v)} for k, v in vals.items()])
            for _, row in t.iterrows():
                v = vals[row['method']].values.astype(float)
                lo, hi = boot_ci(v, rng)
                desc_rows.append({'dataset': ds, 'metric': m, 'method': row['method'],
                                  'n': len(v), 'mean': row['mean'], 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                  'ci_low': lo, 'ci_high': hi})
            # ranks only over methods sharing the largest per-image coverage in
            # this dataset, so a mean from a 30-image subset is never ranked
            # against a mean from 250 images.
            nmax = int(t['n'].max())
            if nmax >= a.min_n:
                sub = t[t['n'] == nmax]
                rk = average_ranks(sub.rename(columns={'mean': m}), m, ascending=False)
                rk.insert(0, 'dataset', ds)
                rk['n_ranked'] = nmax
                rank_rows.append(rk)

        # pairwise paired tests
        pairs = []
        names = sorted(data)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = data[names[i]], data[names[j]]
                common = A.index.intersection(B.index)
                if len(common) < a.min_n:
                    skip_rows.append({'dataset': ds, 'method_a': names[i], 'method_b': names[j],
                                      'n_common': len(common), 'reason': f'n_common < {a.min_n}'})
                    continue
                for m in METRICS:
                    if m not in A.columns or m not in B.columns:
                        continue
                    xa = A.loc[common, m].astype(float).values
                    xb = B.loc[common, m].astype(float).values
                    ok = np.isfinite(xa) & np.isfinite(xb)
                    if ok.sum() < a.min_n:
                        continue
                    d = xa[ok] - xb[ok]
                    lo, hi = boot_ci(d, rng)
                    sd = d.std(ddof=1)
                    pairs.append({'dataset': ds, 'metric': m, 'method_a': names[i], 'method_b': names[j],
                                  'n_common': int(ok.sum()), 'mean_a': float(xa[ok].mean()),
                                  'mean_b': float(xb[ok].mean()), 'mean_diff': float(d.mean()),
                                  'ci_low': lo, 'ci_high': hi, 'p_perm': perm_p(d, rng),
                                  'd_z': float(d.mean() / sd) if sd > 0 else 0.0})
        if pairs:
            p = pd.DataFrame(pairs)
            p['p_holm'] = np.nan
            for m, g in p.groupby('metric'):
                p.loc[g.index, 'p_holm'] = holm(g['p_perm'].values)
            p['sig_holm_0.05'] = p['p_holm'] < 0.05
            p.to_csv(OUT / f'paired_pairwise_{ds}.csv', index=False)
            nsig = int(p['sig_holm_0.05'].sum())
            print(f'{ds}: {len(p)} paired comparisons, {nsig} significant after Holm')

    pd.DataFrame(desc_rows).to_csv(OUT / 'descriptive.csv', index=False)
    # always write the file, even with zero rows: with every method at full
    # coverage there are no exclusions, but readers expect the file to exist
    pd.DataFrame(skip_rows, columns=['dataset', 'method_a', 'method_b',
                                     'n_common', 'reason']).to_csv(
        OUT / 'paired_exclusions.csv', index=False)
    print(f'{len(skip_rows)} pairs skipped for insufficient common IDs')
    if rank_rows:
        pd.concat(rank_rows).to_csv(OUT / 'ranks_all.csv', index=False)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
