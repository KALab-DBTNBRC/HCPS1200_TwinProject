#!/usr/bin/env python3
"""
dynfc_checks.py
---------------
Three checks on the surrogate-corrected (excess) within-family dynamic-fluidity signal,
to decide whether the within-MZ association is real or a parametric artefact.

CHECK 1  motion robustness of excess fluidity         (needs rsfMRI_motion_FD.csv)
CHECK 2  leverage: which discordant pairs drive it
CHECK 3  permutation test of the Mundlak within-family coefficient  <-- the decisive one

Run:  python3 dynfc_checks.py
Env:  pandas numpy scipy statsmodels
"""
import os, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

BASE   = os.environ.get("PROJECT_ROOT", ".")
MASTER = os.path.join(BASE, 'MASTER_ROI_METRICS_DTI_FBA.csv')
EXCESS = os.path.join(BASE, 'rsfMRI_Tier3_Dynamic_Excess.csv')
MOTION = os.path.join(BASE, 'rsfMRI_motion_FD.csv')
N_PERM = 10000
SEED   = 11
COVARS = ['Age_in_Yrs', 'C(Gender)', 'SSAGA_TB_Still_Smoking']
METRICS = ['Global_Dynamic_Fluidity','Dynamic_Fluidity_Reward','Dynamic_Fluidity_Salience',
           'Dynamic_Fluidity_DMN','Dynamic_Fluidity_Olfactory']


def base_df():
    m = pd.read_csv(MASTER)[['Subject','TwinPairID','ZygosityGT1','Severity',
                             'Age_in_Yrs','Gender','SSAGA_TB_Still_Smoking']]
    e = pd.read_csv(EXCESS)
    return m.merge(e, on='Subject')


def add_within_between(df):
    pm = df.groupby('TwinPairID')['Severity'].transform('mean')
    df = df.copy(); df['Severity_BF'] = pm; df['Severity_WF'] = df['Severity'] - pm
    return df


def check1_motion():
    print("\n" + "="*64 + "\nCHECK 1 - motion robustness of excess fluidity\n" + "="*64)
    if not os.path.exists(MOTION):
        print(f"  motion file not found ({MOTION}); run on server.")
        return
    df = base_df().merge(pd.read_csv(MOTION), on='Subject')
    for c in METRICS:
        r_sf,_ = stats.pearsonr(df['Severity'], df[c])
        r_mf,_ = stats.pearsonr(df['Mean_FD'], df[c])
        from numpy.linalg import lstsq
        X = np.c_[np.ones(len(df)), df['Mean_FD']]
        res_s = df['Severity'] - X @ lstsq(X, df['Severity'], rcond=None)[0]
        res_y = df[c]        - X @ lstsq(X, df[c],        rcond=None)[0]
        rp,pp = stats.pearsonr(res_s, res_y)
        print(f"  {c:26s} sev~exc r={r_sf:+.3f} | mot~exc r={r_mf:+.3f} | sev~exc|mot r={rp:+.3f} p={pp:.3f}")
    df = add_within_between(df)
    print("  -- MZ-only within-family beta with Mean_FD covariate --")
    for c in METRICS:
        sub = df[df.ZygosityGT1=='MZ'].dropna(subset=[c,'Mean_FD']).copy(); sub['Y']=sub[c]
        sd=sub['Y'].std(); sdw=sub['Severity_WF'].std()
        mod = smf.mixedlm("Y ~ Severity_BF + Severity_WF + Mean_FD + "+" + ".join(COVARS),
                          sub, groups=sub['TwinPairID']).fit(reml=False)
        print(f"    {c:26s} betaWF={mod.params['Severity_WF']*sdw/sd:+.3f} p={mod.pvalues['Severity_WF']:.3f}")


def check2_leverage():
    print("\n" + "="*64 + "\nCHECK 2 - leverage: which discordant pairs drive within-MZ\n" + "="*64)
    df = add_within_between(base_df())
    mz = df[df.ZygosityGT1=='MZ']
    diffs = []
    for pid,g in mz.groupby('TwinPairID'):
        if len(g)==2:
            d = abs(g['Severity'].iloc[0]-g['Severity'].iloc[1])
            if d>0: diffs.append(d)
    diffs = np.array(diffs)
    print(f"  MZ discordant pairs: {len(diffs)}  | |dSeverity|=1: {(diffs==1).sum()}  |dSeverity|=2: {(diffs==2).sum()}")
    disc_ids = [pid for pid,g in mz.groupby('TwinPairID')
                if len(g)==2 and g['Severity'].iloc[0]!=g['Severity'].iloc[1]]
    for c in ['Global_Dynamic_Fluidity','Dynamic_Fluidity_DMN']:
        def fit(exclude=None):
            sub = mz if exclude is None else mz[mz.TwinPairID!=exclude]
            sub = sub.dropna(subset=[c]).copy(); sub['Y']=sub[c]
            sd=sub['Y'].std(); sdw=sub['Severity_WF'].std()
            mod = smf.mixedlm("Y ~ Severity_BF + Severity_WF + "+" + ".join(COVARS),
                              sub, groups=sub['TwinPairID']).fit(reml=False)
            return mod.params['Severity_WF']*sdw/sd, mod.pvalues['Severity_WF']
        b0,p0 = fit()
        jk = [(pid,)+fit(pid) for pid in disc_ids]
        ps = [p for _,_,p in jk]; bs=[b for _,b,_ in jk]
        worst = max(jk, key=lambda t:t[2])
        print(f"  {c:26s} full betaWF={b0:+.3f} p={p0:.3f} | drop-1 p range [{min(ps):.3f},{max(ps):.3f}] "
              f"beta range [{min(bs):+.3f},{max(bs):+.3f}]")
        print(f"      most influential pair (removal -> p={worst[2]:.3f})")


def check3_perm():
    print("\n" + "="*64 + "\nCHECK 3 - permutation test of within-family coefficient (decisive)\n" + "="*64)
    rng = np.random.default_rng(SEED)
    df = add_within_between(base_df())
    for group in ['Pooled','MZ_only']:
        sub0 = df if group=='Pooled' else df[df.ZygosityGT1=='MZ']
        print(f"  -- {group} --")
        for c in METRICS:
            sub = sub0.dropna(subset=[c]).copy(); sub['Y']=sub[c]
            sd=sub['Y'].std(); sdw=sub['Severity_WF'].std()
            def beta_wf(d):
                m = smf.mixedlm("Y ~ Severity_BF + Severity_WF + "+" + ".join(COVARS),
                                d, groups=d['TwinPairID']).fit(reml=False)
                return m.params['Severity_WF']*sdw/sd, m.pvalues['Severity_WF']
            b_obs, p_param = beta_wf(sub)
            null = np.empty(N_PERM)
            pair_ids = sub['TwinPairID'].values
            wf = sub['Severity_WF'].values.copy()
            uniq = sub['TwinPairID'].unique()
            for k in range(N_PERM):
                flip = {pid:s for pid,s in zip(uniq, rng.choice([-1,1], len(uniq)))}
                sub['Severity_WF'] = wf * np.array([flip[p] for p in pair_ids])
                try: null[k] = beta_wf(sub)[0]
                except Exception: null[k] = np.nan
            sub['Severity_WF'] = wf
            p_perm = (np.sum(np.abs(null) >= abs(b_obs)) + 1)/(np.sum(~np.isnan(null))+1)
            flag = "  <-- parametric sig but perm NULL" if (p_param<0.05 and p_perm>=0.05) else ""
            print(f"    {c:26s} betaWF={b_obs:+.3f} param_p={p_param:.3f} PERM_p={p_perm:.3f}{flag}")


if __name__ == '__main__':
    check1_motion()
    check2_leverage()
    check3_perm()
