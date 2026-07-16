import os
"""
NCT Q3: AUD associations with controllability
Four-step pipeline mirroring B1's analytic logic, applied to NCT metrics.

Step 1  Population LME       360 nodes x 4 metrics
        AC ~ Severity + covariates + (1|TwinPairID)
Step 2  Mundlak BF/WF        all 360 nodes (spotlight included regardless)
Step 3  MZ discordant pairs  80 MZ pairs, perm test (parallelised, node-level)
        + DZ sign-reversal check
Step 4  Spotlight summary    pre-specified olfactory/insular nodes

LME methodology matches B1 FunctionalLME.py exactly:
  - Gender_Label categorical (reference = Female)
  - Rare Race/Ethnicity levels (<5) collapsed to 'Other_Unknown'
  - C(Race) + C(Ethnicity) categorical terms
  - optimizer = 'cg' (robust to convergence warnings)

Outputs -> NCT_inputs/Controllability/aud_analysis/
"""

import warnings, logging, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

CTRL_DIR  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs/Controllability"))
R_INPUTS  = CTRL_DIR / 'r_inputs'
NCT_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
BEH_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/network_roi_metrics_FINAL.csv"))
OUT_DIR   = CTRL_DIR / 'aud_analysis'
OUT_DIR.mkdir(exist_ok=True)

N_WORKERS_LME  = 25
N_WORKERS_DISC = 10
N_PERM         = 5000
ALPHA_FDR      = 0.05

COVARIATES = [
    'Age_in_Yrs', 'Gender_Label',
    'SSAGA_TB_Still_Smoking', 'SSAGA_Times_Used_Illicits',
    'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc',
]
CAT_COVS   = ['Race', 'Ethnicity']

SPOTLIGHT = [
    'R_Pir_ROI','L_Pir_ROI', 'R_52_ROI','L_52_ROI',
    'R_RI_ROI','L_RI_ROI',   'R_EC_ROI','L_EC_ROI',
    'R_AAIC_ROI','L_AAIC_ROI','R_OFC_ROI','L_OFC_ROI',
    'R_25_ROI','L_25_ROI',
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()])
log = logging.getLogger('NCT_Q3')


def load_data():
    log.info('Loading controllability arrays...')
    arrays = {}
    for name in ['MIND_AC_T3', 'SC_AC_T3', 'MIND_MC', 'SC_MC']:
        arr = np.loadtxt(str(R_INPUTS / f'{name}.csv'), delimiter=',').astype(np.float64)
        arrays[name] = arr

    subj_idx   = pd.read_csv(str(CTRL_DIR / 'subject_index.csv'))
    node_df    = pd.read_csv(str(NCT_DIR  / 'node_names.csv'))
    beh        = pd.read_csv(str(BEH_CSV))
    node_names = node_df['Region'].tolist()

    subj_idx['subj_id'] = subj_idx['subj_id'].astype(int)
    beh['Subject']      = beh['Subject'].astype(int)

    df = subj_idx.merge(
        beh[['Subject','TwinPairID','ZygosityGT1','Severity',
             'Age_in_Yrs','Gender','Race','Ethnicity',
             'SSAGA_TB_Still_Smoking','SSAGA_Times_Used_Illicits',
             'SSAGA_Mj_Times_Used','FamHist_Combined_DrgAlc',
             'Total_Wine_7days','Total_Hard_Liquor_7days',
             'Total_Beer_Wine_Cooler_7days']],
        left_on='subj_id', right_on='Subject', how='left'
    )

    df['Gender_Label'] = pd.Categorical(
        df['Gender'].map({0: 'Female', 1: 'Male'}),
        categories=['Female', 'Male'])

    for col in ['Race', 'Ethnicity']:
        counts = df[col].value_counts()
        rare   = counts[counts < 5].index
        df[col] = df[col].replace(rare, 'Other_Unknown').astype(str)

    return arrays, df, node_names


def build_formula(outcome, mundlak=False):
    base = COVARIATES + [f'C({c})' for c in CAT_COVS]
    sev  = ['Severity_BF', 'Severity_WF'] if mundlak else ['Severity']
    return f'{outcome} ~ ' + ' + '.join(sev + base)


def lme_one_node(args):
    node_i, node_name, metric_name, node_vals, df_beh = args
    try:
        d = df_beh.copy()
        d['_y'] = node_vals
        d = d.dropna(subset=['_y', 'Severity', 'TwinPairID'])
        if len(d) < 50:
            return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                    'Beta_Severity': np.nan, 'P_Value': np.nan,
                    'N': len(d), 'Status': 'INSUFFICIENT_DATA'}

        mod = smf.mixedlm(build_formula('_y'), d, groups=d['TwinPairID'])
        res = mod.fit(method='cg', maxiter=500, disp=False)
        return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                'Beta_Severity': round(float(res.params.get('Severity', np.nan)), 6),
                'P_Value':       round(float(res.pvalues.get('Severity', np.nan)), 6),
                'N': int(res.nobs), 'Status': 'OK'}
    except Exception as exc:
        return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                'Beta_Severity': np.nan, 'P_Value': np.nan, 'N': 0,
                'Status': f'FAILED: {str(exc)[:40]}'}


def run_population_lme(arrays, df, node_names):
    log.info('STEP 1: Population LME')
    all_lme = {}
    for metric_name, arr in arrays.items():
        t0 = time.time()
        args = [(i, node_names[i], metric_name, arr[:, i], df)
                for i in range(len(node_names))]
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS_LME) as pool:
            for fut in as_completed([pool.submit(lme_one_node, a) for a in args]):
                results.append(fut.result())

        res_df = pd.DataFrame(results).sort_values('node_i').reset_index(drop=True)
        res_df['FDR_q']   = _fdr(res_df['P_Value'])
        res_df['Sig_FDR'] = res_df['FDR_q'] < ALPHA_FDR

        res_df.to_csv(str(OUT_DIR / f'lme_population_{metric_name}.csv'), index=False)
        all_lme[metric_name] = res_df
    return all_lme


def mundlak_one_node(args):
    node_i, node_name, metric_name, node_vals, df_beh = args
    try:
        d = df_beh.copy()
        d['_y'] = node_vals
        d = d.dropna(subset=['_y', 'Severity', 'TwinPairID'])
        if len(d) < 50:
            return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                    'Beta_BF': np.nan, 'P_BF': np.nan,
                    'Beta_WF': np.nan, 'P_WF': np.nan,
                    'N': len(d), 'Status': 'INSUFFICIENT_DATA'}

        d['Severity_BF'] = d.groupby('TwinPairID')['Severity'].transform('mean')
        d['Severity_WF'] = d['Severity'] - d['Severity_BF']

        mod = smf.mixedlm(build_formula('_y', mundlak=True), d, groups=d['TwinPairID'])
        res = mod.fit(method='cg', maxiter=500, disp=False)
        return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                'Beta_BF': round(float(res.params.get('Severity_BF', np.nan)), 6),
                'P_BF':    round(float(res.pvalues.get('Severity_BF', np.nan)), 6),
                'Beta_WF': round(float(res.params.get('Severity_WF', np.nan)), 6),
                'P_WF':    round(float(res.pvalues.get('Severity_WF', np.nan)), 6),
                'N': int(res.nobs), 'Status': 'OK'}
    except Exception as exc:
        return {'node_i': node_i, 'Region': node_name, 'Metric': metric_name,
                'Beta_BF': np.nan, 'P_BF': np.nan,
                'Beta_WF': np.nan, 'P_WF': np.nan,
                'N': 0, 'Status': f'FAILED: {str(exc)[:40]}'}


def run_mundlak(arrays, df, node_names):
    log.info('STEP 2: Mundlak BF/WF Decomposition')
    all_mundlak = {}
    for metric_name, arr in arrays.items():
        t0 = time.time()
        args = [(i, node_names[i], metric_name, arr[:, i], df)
                for i in range(len(node_names))]
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS_LME) as pool:
            for fut in as_completed([pool.submit(mundlak_one_node, a) for a in args]):
                results.append(fut.result())

        res_df = pd.DataFrame(results).sort_values('node_i').reset_index(drop=True)
        res_df['FDR_q_BF'] = _fdr(res_df['P_BF'])
        res_df['FDR_q_WF'] = _fdr(res_df['P_WF'])
        res_df['Sig_BF']   = res_df['FDR_q_BF'] < ALPHA_FDR
        res_df['Sig_WF']   = res_df['FDR_q_WF'] < ALPHA_FDR

        res_df.to_csv(str(OUT_DIR / f'mundlak_{metric_name}.csv'), index=False)
        all_mundlak[metric_name] = res_df
    return all_mundlak


def _disc_one_node(node_i, node_name, vals, mz_hi, mz_lo, delta_sev,
                   dz_hi, dz_lo, dz_dsev, spotlight_set):
    """Process one node: MZ Spearman, permutation, DZ check, jackknife."""
    delta_y = vals[mz_hi] - vals[mz_lo]
    valid   = ~np.isnan(delta_y) & ~np.isnan(delta_sev)

    if valid.sum() < 5:
        return {'node_i': node_i, 'Region': node_name,
                'MZ_Spearman_rho': np.nan, 'MZ_Perm_p': np.nan,
                'DZ_rho': np.nan, 'Sign_Reversal': False,
                'JK_Stability': 'INSUFFICIENT', 'N_MZ_Pairs': int(valid.sum()),
                'Spotlight': node_name in spotlight_set}

    dy_v, ds_v = delta_y[valid], delta_sev[valid]
    rho, _ = stats.spearmanr(ds_v, dy_v)

    rng  = np.random.default_rng(42)
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        signs   = rng.choice([-1, 1], size=valid.sum())
        null[p] = stats.spearmanr(ds_v, dy_v * signs)[0]
    perm_p = float((np.abs(null) >= np.abs(rho)).mean())

    dz_delta = vals[dz_hi] - vals[dz_lo]
    dz_valid = ~np.isnan(dz_delta) & ~np.isnan(dz_dsev)
    dz_rho   = (stats.spearmanr(dz_dsev[dz_valid], dz_delta[dz_valid])[0]
                if dz_valid.sum() >= 3 else np.nan)

    jk = []
    for drop in range(valid.sum()):
        m = np.ones(valid.sum(), dtype=bool); m[drop] = False
        if m.sum() >= 4:
            jk.append(stats.spearmanr(ds_v[m], dy_v[m])[0])
    jk_stable = ('STABLE' if jk and all(np.sign(r) == np.sign(rho) for r in jk)
                 else 'UNSTABLE')

    return {'node_i': node_i, 'Region': node_name,
            'MZ_Spearman_rho': round(float(rho), 4),
            'MZ_Perm_p': round(perm_p, 4),
            'DZ_rho': round(float(dz_rho), 4) if not np.isnan(dz_rho) else np.nan,
            'Sign_Reversal': (not np.isnan(dz_rho)) and (np.sign(rho) != np.sign(dz_rho)),
            'JK_Stability': jk_stable,
            'N_MZ_Pairs': int(valid.sum()),
            'Spotlight': node_name in spotlight_set}


def _build_pair_index(df, zyg_prefix):
    """Return (hi_rows, lo_rows, delta_sev) for pairs of a given zygosity."""
    sub = df[df['ZygosityGT1'].str.startswith(zyg_prefix)].copy()
    sub = sub.groupby('TwinPairID').filter(lambda g: len(g) == 2)
    sub = sub.sort_values(['TwinPairID', 'Severity'], ascending=[True, False])
    subj_to_row = {int(r['subj_id']): int(r['row'])
                   for _, r in df.iterrows() if pd.notna(r['subj_id'])}
    hi, lo, dsev = [], [], []
    for pid, g in sub.groupby('TwinPairID'):
        g = g.reset_index(drop=True)
        r0 = subj_to_row.get(int(g.iloc[0]['subj_id']))
        r1 = subj_to_row.get(int(g.iloc[1]['subj_id']))
        if r0 is None or r1 is None:
            continue
        hi.append(r0); lo.append(r1)
        dsev.append(int(g.iloc[0]['Severity']) - int(g.iloc[1]['Severity']))
    return np.array(hi), np.array(lo), np.array(dsev, dtype=float)


def mz_discordant(arrays, df, node_names):
    log.info('STEP 3: MZ Discordant Pairs')
    mz_hi, mz_lo, delta_sev = _build_pair_index(df, 'MZ')
    dz_hi, dz_lo, dz_dsev    = _build_pair_index(df, 'DZ')
    n_disc = int((delta_sev != 0).sum())

    spotlight_set = set(SPOTLIGHT)
    all_disc = {}

    for metric_name, arr in arrays.items():
        t0 = time.time()

        worker = partial(_disc_one_node,
                         mz_hi=mz_hi, mz_lo=mz_lo, delta_sev=delta_sev,
                         dz_hi=dz_hi, dz_lo=dz_lo, dz_dsev=dz_dsev,
                         spotlight_set=spotlight_set)

        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS_DISC) as pool:
            futs = [pool.submit(worker, i, node_names[i], arr[:, i])
                    for i in range(len(node_names))]
            for fut in as_completed(futs):
                results.append(fut.result())

        res_df = pd.DataFrame(results).sort_values('node_i').reset_index(drop=True)
        res_df['Metric']    = metric_name
        res_df['N_Disc']    = n_disc
        res_df['FDR_q_MZ']  = _fdr(res_df['MZ_Perm_p'])
        res_df['Sig_MZ']    = res_df['FDR_q_MZ'] < ALPHA_FDR

        res_df.to_csv(str(OUT_DIR / f'mz_discordant_{metric_name}.csv'), index=False)
        all_disc[metric_name] = res_df
    return all_disc


def spotlight_summary(lme_results, mundlak_results, disc_results):
    log.info('STEP 4: Spotlight Summary')
    rows = []
    for metric_name in lme_results:
        lme = lme_results[metric_name].set_index('Region')
        mnd = mundlak_results[metric_name].set_index('Region')
        dsc = disc_results[metric_name].set_index('Region')
        for node in SPOTLIGHT:
            if node not in lme.index:
                continue
            rows.append({
                'Region': node, 'Metric': metric_name,
                'LME_Beta':  lme.at[node, 'Beta_Severity'],
                'LME_P':     lme.at[node, 'P_Value'],
                'LME_FDR_q': lme.at[node, 'FDR_q'],
                'BF_Beta':   mnd.at[node, 'Beta_BF'], 'BF_P': mnd.at[node, 'P_BF'],
                'WF_Beta':   mnd.at[node, 'Beta_WF'], 'WF_P': mnd.at[node, 'P_WF'],
                'MZ_rho':    dsc.at[node, 'MZ_Spearman_rho'],
                'MZ_Perm_p': dsc.at[node, 'MZ_Perm_p'],
                'DZ_rho':    dsc.at[node, 'DZ_rho'],
                'Sign_Reversal': dsc.at[node, 'Sign_Reversal'],
                'JK_Stability':  dsc.at[node, 'JK_Stability'],
            })
    spot_df = pd.DataFrame(rows)
    spot_df.to_csv(str(OUT_DIR / 'spotlight_aud_summary.csv'), index=False)
    return spot_df


def _fdr(pvals):
    from statsmodels.stats.multitest import multipletests
    pv    = pd.to_numeric(pvals, errors='coerce').values
    out   = np.full(len(pv), np.nan)
    valid = ~np.isnan(pv)
    if valid.sum() > 0:
        out[valid] = multipletests(pv[valid], method='fdr_bh')[1]
    return out


def main():
    t0 = time.time()
    log.info('NCT Q3: AUD Associations with Controllability')

    arrays, df, node_names = load_data()

    lme_results     = run_population_lme(arrays, df, node_names)
    mundlak_results = run_mundlak(arrays, df, node_names)
    disc_results    = mz_discordant(arrays, df, node_names)
    spotlight_summary(lme_results, mundlak_results, disc_results)

    log.info(f'Q3 COMPLETE   Wall: {timedelta(seconds=int(time.time()-t0))}')
    log.info(f'Outputs -> {OUT_DIR}/')
    log.info('Next: nct_control_energy.py (Q4)')


if __name__ == '__main__':
    main()
