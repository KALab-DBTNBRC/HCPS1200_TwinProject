import os
"""
NCT Q4: Minimum control energy
What is the minimum structural energy cost to drive
the brain from rest (x0=0) into a target network activation state x_T,
and does that cost differ by AUD severity or show heritability?

Theory (Gu et al. 2015; Karrer et al. 2020):
  System   dx/dt = A x + B u,   B = I (all nodes controllable)
  Gramian  W_c(T) = integral 0..T e^{At} BB' e^{A't} dt
  Min energy from x0=0 to x_T:
      E_min(x_T) = x_T' W_c(T)^{-1} x_T

For symmetric A = V Lambda V':
  W_c(T) = V diag[(e^{2 lambda T}-1)/(2 lambda)] V'
  W_c(T)^{-1} = V diag[2 lambda/(e^{2 lambda T}-1)] V'         (lambda->0 limit: 1/T)
  E_min = sum_i (V' x_T)_i^2 * 2*lambda_i/(e^{2*lambda_i*T}-1)

Two target families:
  (A) NETWORK targets  -- cortical Reward / Olfactory / Salience / DMN
  (B) NODE targets     -- unit impulse at each of 360 nodes
                          E_node[i] = W_c(T)^{-1}[i,i]

Both substrates (MIND, SC-log), T = 1, 3, 5.

Downstream tests on network energies (primary T=3):
  1. Substrate comparison (MIND vs SC energy correlation across subjects)
  2. AUD LME: log(E) ~ Severity + covariates + (1|TwinPairID)
  3. MZ discordant: Delta log(E) ~ Delta Severity

Outputs -> NCT_inputs/Controllability/control_energy/
"""

import warnings, logging, time
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

CTRL_DIR  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs/Controllability"))
NCT_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
MIND_NORM = NCT_DIR / 'MIND_norm'
SC_NORM   = NCT_DIR / 'SC_norm'
TARGETS   = CTRL_DIR / 'targets' / 'target_vectors.npy'
BEH_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/network_roi_metrics_FINAL.csv"))
OUT_DIR   = CTRL_DIR / 'control_energy'
OUT_DIR.mkdir(exist_ok=True)

N_NODES       = 360
T_PRIMARY     = 3
T_ALL         = [1, 3, 5]
EIG_EPS       = 1e-10
NETWORK_ORDER = ['Reward', 'Olfactory', 'Salience', 'DMN']
N_WORKERS_LME = 40
N_WORKERS_DISC= 10
N_PERM        = 5000
ALPHA_FDR     = 0.05

COVARIATES = [
    'Age_in_Yrs', 'Gender_Label',
    'SSAGA_TB_Still_Smoking', 'SSAGA_Times_Used_Illicits',
    'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc',
]
CAT_COVS = ['Race', 'Ethnicity']

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('NCT_Q4')


def compute_energies(eigvals, eigvecs, targets, T):
    """
    Network + node minimum control energies for all subjects at horizon T.
    """
    N, M = eigvals.shape

    twoLT = 2.0 * eigvals * T
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inv_coeff = np.where(np.abs(eigvals) > EIG_EPS,
                             2.0 * eigvals / np.expm1(twoLT),
                             1.0 / T)

    proj = np.einsum('nmi,km->nki', eigvecs, targets)
    net_E = np.einsum('nki,ni->nk', proj**2, inv_coeff)

    node_E = np.einsum('nji,ni->nj', eigvecs**2, inv_coeff)

    return net_E, node_E


def load_stack(directory, tag, subject_ids):
    arrays = []
    for sid in subject_ids:
        p = directory / f'{sid}_{tag}_norm.npy'
        arrays.append(np.load(str(p)).astype(np.float64))
    return np.stack(arrays, axis=0)


def load_metadata():
    subj_idx = pd.read_csv(str(CTRL_DIR / 'subject_index.csv'))
    beh      = pd.read_csv(str(BEH_CSV))
    subj_idx['subj_id'] = subj_idx['subj_id'].astype(int)
    beh['Subject']      = beh['Subject'].astype(int)
    df = subj_idx.merge(
        beh[['Subject','TwinPairID','ZygosityGT1','Severity',
             'Age_in_Yrs','Gender','Race','Ethnicity',
             'SSAGA_TB_Still_Smoking','SSAGA_Times_Used_Illicits',
             'SSAGA_Mj_Times_Used','FamHist_Combined_DrgAlc']],
        left_on='subj_id', right_on='Subject', how='left')
    df['Gender_Label'] = pd.Categorical(
        df['Gender'].map({0:'Female',1:'Male'}), categories=['Female','Male'])
    for col in ['Race','Ethnicity']:
        counts = df[col].value_counts()
        rare   = counts[counts < 5].index
        df[col] = df[col].replace(rare, 'Other_Unknown').astype(str)
    return subj_idx, df


def _fdr(pvals):
    from statsmodels.stats.multitest import multipletests
    pv = pd.to_numeric(pvals, errors='coerce').values
    out = np.full(len(pv), np.nan); v = ~np.isnan(pv)
    if v.sum() > 0: out[v] = multipletests(pv[v], method='fdr_bh')[1]
    return out


def lme_energy(df, energy_col):
    base = COVARIATES + [f'C({c})' for c in CAT_COVS]
    formula = f'{energy_col} ~ Severity + ' + ' + '.join(base)
    d = df.dropna(subset=[energy_col,'Severity','TwinPairID'])
    mod = smf.mixedlm(formula, d, groups=d['TwinPairID'])
    res = mod.fit(method='cg', maxiter=500, disp=False)
    d2 = d.copy()
    d2['Severity_BF'] = d2.groupby('TwinPairID')['Severity'].transform('mean')
    d2['Severity_WF'] = d2['Severity'] - d2['Severity_BF']
    f2 = f'{energy_col} ~ Severity_BF + Severity_WF + ' + ' + '.join(base)
    res2 = smf.mixedlm(f2, d2, groups=d2['TwinPairID']).fit(method='cg', disp=False)
    return {
        'Beta_Severity': float(res.params.get('Severity', np.nan)),
        'P_Severity':    float(res.pvalues.get('Severity', np.nan)),
        'Beta_BF': float(res2.params.get('Severity_BF', np.nan)),
        'P_BF':    float(res2.pvalues.get('Severity_BF', np.nan)),
        'Beta_WF': float(res2.params.get('Severity_WF', np.nan)),
        'P_WF':    float(res2.pvalues.get('Severity_WF', np.nan)),
        'N': int(res.nobs),
    }


def discordant_energy(df, energy_col):
    """MZ discordant Spearman + permutation for one energy column."""
    mz = df[df['ZygosityGT1'].str.startswith('MZ')]
    mz = mz.groupby('TwinPairID').filter(lambda g: len(g)==2)
    mz = mz.sort_values(['TwinPairID','Severity'], ascending=[True,False])
    dsev, dE = [], []
    for pid, g in mz.groupby('TwinPairID'):
        g = g.reset_index(drop=True)
        dsev.append(g.iloc[0]['Severity'] - g.iloc[1]['Severity'])
        dE.append(g.iloc[0][energy_col] - g.iloc[1][energy_col])
    dsev, dE = np.array(dsev,float), np.array(dE,float)
    valid = ~np.isnan(dE) & ~np.isnan(dsev)
    if valid.sum() < 5:
        return {'MZ_rho': np.nan, 'MZ_perm_p': np.nan, 'N_pairs': int(valid.sum())}
    rho,_ = stats.spearmanr(dsev[valid], dE[valid])
    rng = np.random.default_rng(42)
    null = np.array([stats.spearmanr(dsev[valid],
                     dE[valid]*rng.choice([-1,1],size=valid.sum()))[0]
                     for _ in range(N_PERM)])
    perm_p = float((np.abs(null) >= abs(rho)).mean())
    return {'MZ_rho': round(float(rho),4), 'MZ_perm_p': round(perm_p,4),
            'N_pairs': int(valid.sum())}


def main():
    t0 = time.time()
    log.info('NCT Q4: Minimum Control Energy')

    subj_idx, df = load_metadata()
    subject_ids  = subj_idx['subj_id'].astype(str).tolist()
    N = len(subject_ids)

    targets = np.load(str(TARGETS))

    log.info('Loading + eigendecomposing MIND...')
    MIND = load_stack(MIND_NORM, 'MIND', subject_ids)
    mind_w, mind_v = np.linalg.eigh(MIND); del MIND

    log.info('Loading + eigendecomposing SC...')
    SC = load_stack(SC_NORM, 'SC', subject_ids)
    sc_w, sc_v = np.linalg.eigh(SC); del SC

    energy_summary = []
    network_energy_store = {}

    for T in T_ALL:
        for subname, (w, v) in [('MIND', (mind_w, mind_v)),
                                 ('SC',   (sc_w,   sc_v))]:
            net_E, node_E = compute_energies(w, v, targets, T)

            if subname == 'SC':
                net_E_save  = np.log(net_E)
                node_E_save = np.log(node_E)
                tag = 'log'
            else:
                net_E_save, node_E_save, tag = net_E, node_E, 'raw'

            np.save(str(OUT_DIR / f'{subname}_network_energy_T{T}.npy'), net_E_save)
            np.save(str(OUT_DIR / f'{subname}_node_energy_T{T}.npy'),    node_E_save)

            if T == T_PRIMARY:
                network_energy_store[subname] = net_E_save

            for k, net in enumerate(NETWORK_ORDER):
                col = net_E_save[:, k]
                energy_summary.append({
                    'Substrate': subname, 'T': T, 'Network': net,
                    'Energy_mean': round(float(col.mean()),4),
                    'Energy_sd':   round(float(col.std()),4),
                    'Energy_min':  round(float(col.min()),4),
                    'Energy_max':  round(float(col.max()),4),
                    'Transform':   tag,
                })

    pd.DataFrame(energy_summary).to_csv(
        str(OUT_DIR / 'energy_summary.csv'), index=False)

    energy_df = subj_idx.copy()
    for subname in ['MIND','SC']:
        for k, net in enumerate(NETWORK_ORDER):
            energy_df[f'{subname}_E_{net}'] = network_energy_store[subname][:,k]
    energy_df['subj_id'] = energy_df['subj_id'].astype(int)
    full = energy_df.merge(df, on='subj_id', how='left', suffixes=('','_b'))
    energy_df.to_csv(str(OUT_DIR / 'subject_network_energy.csv'), index=False)

    sub_rows = []
    for net in NETWORK_ORDER:
        m = energy_df[f'MIND_E_{net}']
        s = energy_df[f'SC_E_{net}']
        r,p = stats.pearsonr(m, s)
        sub_rows.append({'Network': net, 'Pearson_r': round(r,4), 'p': round(p,6)})
    pd.DataFrame(sub_rows).to_csv(str(OUT_DIR/'energy_substrate_comparison.csv'), index=False)

    aud_rows = []
    for subname in ['MIND','SC']:
        for net in NETWORK_ORDER:
            col = f'{subname}_E_{net}'
            try:
                lme = lme_energy(full, col)
                disc = discordant_energy(full, col)
                aud_rows.append({'Substrate':subname,'Network':net, **lme, **disc})
            except Exception as exc:
                log.error(f'  {subname} {net}: {exc}')
                aud_rows.append({'Substrate':subname,'Network':net,
                                 'Status':str(exc)[:50]})

    aud_df = pd.DataFrame(aud_rows)
    for pcol, qcol in [('P_Severity','FDR_Severity'),('P_BF','FDR_BF'),
                       ('P_WF','FDR_WF'),('MZ_perm_p','FDR_MZ')]:
        if pcol in aud_df:
            aud_df[qcol] = _fdr(aud_df[pcol])
    aud_df.to_csv(str(OUT_DIR / 'energy_aud_association.csv'), index=False)

    for subname, ac_file in [('MIND','MIND_AC_T3.npy'), ('SC','SC_AC_T3.npy')]:
        ac = np.load(str(CTRL_DIR / ac_file))
        ne = np.load(str(OUT_DIR / f'{subname}_node_energy_T{T_PRIMARY}.npy'))
        if subname=='SC': ac = np.log(ac)
        ac_grp = ac.mean(0); ne_grp = ne.mean(0)
        r,_ = stats.pearsonr(ac_grp, ne_grp)
        rs,_= stats.spearmanr(ac_grp, ne_grp)
        log.info(f'  {subname}: node-E vs AC  Pearson r={r:.4f}  Spearman rho={rs:.4f}')

    wall = timedelta(seconds=int(time.time()-t0))
    log.info(f'Q4 COMPLETE   Wall: {wall}   Outputs -> {OUT_DIR}/')
    log.info('Next: nct_optimal_control.py (Q5)')


if __name__ == '__main__':
    main()
