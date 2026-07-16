#!/usr/bin/env python3
"""
dynfc_zygosity_split.py
-----------------------
RECORD the unmasking: the within-family dynamic-fluidity signal split by zygosity.
Reproduces pooled / MZ-only / DZ-only Mundlak within-family decomposition, plus the
oriented MZ- and DZ-discordant contrasts (permutation, Wilcoxon, jackknife).

Inferential quantity is the raw-coef p-value (standardization-invariant).
Standardized beta uses sd(Severity_WF)/sd(Y) to match the lab's Mundlak convention.

Outputs (kept as record):
  mundlak_DynamicFC_zygosity_split.csv
  dynfc_discordant_deltas.csv

Env: pandas, numpy, scipy, statsmodels  (conda env dtiproject)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ---------------------------------------------------------------
BASE   = os.environ.get("PROJECT_ROOT", ".")
MASTER = os.path.join(BASE, 'MASTER_ROI_METRICS_DTI_FBA.csv')
DFC    = os.path.join(BASE, 'rsfMRI_Tier3_Dynamic_Metrics.csv')
OUT_SPLIT = os.path.join(BASE, 'mundlak_DynamicFC_zygosity_split.csv')
OUT_DELTA = os.path.join(BASE, 'dynfc_discordant_deltas.csv')
N_PERM = 20000
SEED   = 1

METRICS = ['Global_Dynamic_Fluidity', 'Dynamic_Fluidity_Reward',
           'Dynamic_Fluidity_Salience', 'Dynamic_Fluidity_DMN',
           'Dynamic_Fluidity_Olfactory']
COVARS  = ['Age_in_Yrs', 'C(Gender)', 'SSAGA_TB_Still_Smoking']
# ---------------------------------------------------------------------------


def load():
    m = pd.read_csv(MASTER)[['Subject', 'TwinPairID', 'ZygosityGT1', 'Severity',
                             'Age_in_Yrs', 'Gender', 'SSAGA_TB_Still_Smoking']]
    d = pd.read_csv(DFC)
    return m.merge(d, on='Subject')


def mundlak(df, metric):
    """Within/between Mundlak fit; returns standardized betas + raw-coef p."""
    sub = df.dropna(subset=[metric, 'Severity']).copy()
    pm = sub.groupby('TwinPairID')['Severity'].transform('mean')
    sub['Severity_BF'] = pm
    sub['Severity_WF'] = sub['Severity'] - pm
    sub['Y'] = sub[metric]
    sd_y, sd_wf, sd_bf = sub['Y'].std(), sub['Severity_WF'].std(), sub['Severity_BF'].std()
    formula = "Y ~ Severity_BF + Severity_WF + " + " + ".join(COVARS)
    mod = smf.mixedlm(formula, sub, groups=sub['TwinPairID']).fit(reml=False)
    return {
        'N': len(sub),
        'Beta_BF': mod.params['Severity_BF'] * (sd_bf / sd_y) if sd_bf > 0 else np.nan,
        'P_BF':    mod.pvalues['Severity_BF'],
        'Beta_WF': mod.params['Severity_WF'] * (sd_wf / sd_y) if sd_wf > 0 else np.nan,
        'P_WF':    mod.pvalues['Severity_WF'],
    }


def run_group(df, label):
    rows = []
    for met in METRICS:
        r = mundlak(df, met)
        r.update({'Metric': met, 'Group': label})
        rows.append(r)
    out = pd.DataFrame(rows)
    out['FDR_BF'] = stats.false_discovery_control(out['P_BF'])
    out['FDR_WF'] = stats.false_discovery_control(out['P_WF'])
    return out


def oriented_deltas(df, zyg, metric):
    """Higher-severity minus lower-severity co-twin, discordant pairs only."""
    sub = df[df['ZygosityGT1'] == zyg]
    dl = []
    for _, g in sub.groupby('TwinPairID'):
        if len(g) != 2:
            continue
        g = g.sort_values('Severity')
        lo, hi = g.iloc[0], g.iloc[1]
        if hi['Severity'] == lo['Severity']:
            continue
        dl.append(hi[metric] - lo[metric])
    return np.asarray(dl, float)


def signflip_p(d, rng, n=N_PERM):
    obs = abs(d.mean())
    hits = sum(abs((d * rng.choice([-1, 1], size=len(d))).mean()) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    print(f"N merged = {len(df)}")

    # --- zygosity-split Mundlak ---
    split = pd.concat([run_group(df, 'Pooled'),
                       run_group(df[df.ZygosityGT1 == 'MZ'], 'MZ_only'),
                       run_group(df[df.ZygosityGT1 == 'DZ'], 'DZ_only')],
                      ignore_index=True)
    split = split[['Group', 'Metric', 'N', 'Beta_BF', 'P_BF', 'FDR_BF',
                   'Beta_WF', 'P_WF', 'FDR_WF']].round(4)
    split.to_csv(OUT_SPLIT, index=False)
    print(f"\nSaved {OUT_SPLIT}")
    print(split.to_string(index=False))

    # --- oriented discordant contrasts ---
    drows = []
    for zyg in ['MZ', 'DZ']:
        for met in METRICS:
            d = oriented_deltas(df, zyg, met)
            if len(d) < 3:
                continue
            jk = np.array([np.delete(d, i).mean() for i in range(len(d))])
            drows.append({
                'Zygosity': zyg, 'Metric': met, 'N_disc': len(d),
                'MeanDelta': round(d.mean(), 5), 'SD': round(d.std(ddof=1), 5),
                'Frac_pos': round((d > 0).mean(), 2),
                'Perm_p': round(signflip_p(d, rng), 4),
                'Wilcoxon_p': round(stats.wilcoxon(d).pvalue, 4) if np.any(d != 0) else np.nan,
                'JK_min': round(jk.min(), 5), 'JK_max': round(jk.max(), 5),
            })
    deltas = pd.DataFrame(drows)
    deltas.to_csv(OUT_DELTA, index=False)
    print(f"\nSaved {OUT_DELTA}")
    print(deltas.to_string(index=False))

    # --- verdict ---
    mz = split[(split.Group == 'MZ_only')]
    dz = split[(split.Group == 'DZ_only')]
    print("\n=== VERDICT ===")
    print(f"MZ-only WF significant metrics (p<0.05): {(mz.P_WF < 0.05).sum()} / {len(mz)}")
    print(f"DZ-only WF significant metrics (p<0.05): {(dz.P_WF < 0.05).sum()} / {len(dz)}")
    print("If MZ~0 and DZ carries it -> within-family signal is genetic, not exposure.")


if __name__ == '__main__':
    main()
