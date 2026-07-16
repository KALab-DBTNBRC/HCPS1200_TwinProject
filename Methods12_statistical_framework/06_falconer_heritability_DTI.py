"""
falconer_heritability_DTI.py

Corresponds to Methods: ACE identification diagnostics for the DTI tensor
metrics (MD/RD/AD/FA), used specifically because the DZ correlations for
diffusivity in two networks (reward, limbic-olfactory) are NEGATIVE --
a value the standard ACE model cannot accommodate, making the ML variance
components built on them artefactual (see Results 3.8: "the DZ
correlations were negative... which the ACE model cannot accommodate,
so the variance components built upon them are artefactual").

Falconer method-of-moments: rMZ, rDZ -> A = 2*(rMZ-rDZ), C = 2*rDZ-rMZ.
This classical, assumption-light estimator is what identifies the
negative-rDZ networks as non-estimable under ACE (A implied negative or
undefined), which is exactly the diagnostic this project's Results text
reports -- this script is the source of that diagnosis, not just a
sanity check on an otherwise-trusted ML fit.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

CSV_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv")
OUT_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "DTI_TENSOR_FALCONER_ACE.csv")

TENSOR_METRICS = [f'{tm}_{net}' for tm in ['MD', 'FA', 'RD', 'AD']
                  for net in ['Reward', 'Salience', 'DMN', 'Olfactory']]


POTENTIAL_COVS = [
    'Age_in_Yrs', 'Gender', 'SSAGA_TB_Still_Smoking', 'Race', 'Ethnicity',
    'SSAGA_Times_Used_Illicits', 'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc',
]


def get_valid_covariates(df, target_cols, potential_covs):
    """Only include a covariate if it varies within the exact sub-sample
    entering this model -- the same dynamic-filtering convention used
    throughout this project's population-association scripts. This is
    the actual paper strategy for 'the covariate set' referenced in
    Methods (Heritability/ACE): the same general set, not a reduced one."""
    clean_df = df.dropna(subset=target_cols + potential_covs).copy()
    valid_covs = []
    for cov in potential_covs:
        if clean_df[cov].nunique() > 1:
            valid_covs.append(cov)
    return valid_covs, clean_df


def build_pair_matrix(df, metric):
    valid_covs, d = get_valid_covariates(df, [metric], POTENTIAL_COVS)
    d = d.copy()
    # Race/Ethnicity are categorical -- dummy-encode for the OLS residualisation
    X = d[valid_covs].copy()
    for cat_col in ['Race', 'Ethnicity']:
        if cat_col in X.columns:
            X = pd.get_dummies(X, columns=[cat_col], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = d[metric]
    resid = sm.OLS(y, X).fit().resid
    d['resid_z'] = (resid - resid.mean()) / resid.std()
    mz_pairs, dz_pairs = [], []
    for pid, g in d.groupby('TwinPairID'):
        if len(g) != 2:
            continue
        zyg = g['ZygosityGT1'].iloc[0]
        vals = g['resid_z'].values
        if zyg == 'MZ':
            mz_pairs.append(vals)
        elif zyg == 'DZ':
            dz_pairs.append(vals)
    return np.array(mz_pairs), np.array(dz_pairs)


def ace_negloglik(params, mz, dz):
    a2, c2 = params
    e2 = 1 - a2 - c2
    if a2 < 0 or c2 < 0 or e2 <= 1e-6:
        return 1e10
    nll = 0.0
    for group, cov_within in [(mz, a2 + c2), (dz, 0.5 * a2 + c2)]:
        if len(group) == 0:
            continue
        Sigma = np.array([[1.0, cov_within], [cov_within, 1.0]])
        try:
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return 1e10
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return 1e10
        n = len(group)
        quad = np.einsum('ij,jk,ik->i', group, Sigma_inv, group).sum()
        nll += 0.5 * n * (2 * np.log(2 * np.pi) + logdet) + 0.5 * quad
    return nll


def dense_grid_refit(mz, dz, step=0.02):
    best_nll, best_params = np.inf, None
    for a2 in np.arange(0.0, 1.0, step):
        for c2 in np.arange(0.0, 1.0 - a2 + step, step):
            if a2 + c2 > 1:
                continue
            nll = ace_negloglik([a2, c2], mz, dz)
            if nll < best_nll:
                best_nll, best_params = nll, (a2, c2)
    res = minimize(ace_negloglik, x0=list(best_params), args=(mz, dz), method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    a2, c2 = res.x
    return max(0, min(1, a2)), max(0, min(1 - a2, c2)), res.fun


def falconer_moments(mz, dz):
    """Classical method-of-moments: rMZ, rDZ from Pearson correlation."""
    rMZ = np.corrcoef(mz[:, 0], mz[:, 1])[0, 1] if len(mz) > 1 else np.nan
    rDZ = np.corrcoef(dz[:, 0], dz[:, 1])[0, 1] if len(dz) > 1 else np.nan
    A = 2 * (rMZ - rDZ)
    C = 2 * rDZ - rMZ
    return rMZ, rDZ, A, C


def fit_reduced_ae(mz, dz):
    best = None
    for a2_0 in [0.3, 0.5, 0.7]:
        res = minimize(lambda p: ace_negloglik([p[0], 0.0], mz, dz), x0=[a2_0], method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        if best is None or res.fun < best.fun:
            best = res
    return max(0, min(1, best.x[0])), best.fun


def bootstrap_ci(mz, dz, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    a_boot, c_boot = [], []
    n_mz, n_dz = len(mz), len(dz)
    for _ in range(n_boot):
        mz_b = mz[rng.integers(0, n_mz, n_mz)] if n_mz > 0 else mz
        dz_b = dz[rng.integers(0, n_dz, n_dz)] if n_dz > 0 else dz
        try:
            a2_b, c2_b, _ = dense_grid_refit(mz_b, dz_b, step=0.05)
            a_boot.append(a2_b)
            c_boot.append(c2_b)
        except Exception:
            continue
    a_boot, c_boot = np.array(a_boot), np.array(c_boot)
    return (np.percentile(a_boot, 2.5), np.percentile(a_boot, 97.5),
            np.percentile(c_boot, 2.5), np.percentile(c_boot, 97.5))


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, low_memory=False)
    results = []
    print(f"{'Metric':<14} {'A(ML)':>7} {'C(ML)':>7} {'rMZ':>7} {'rDZ':>7} {'A(Falc)':>8} {'C(Falc)':>8} {'LRT_p_corr':>10}")
    for metric in TENSOR_METRICS:
        mz, dz = build_pair_matrix(df, metric)
        a2, c2, nll_full = dense_grid_refit(mz, dz)
        a2_red, nll_red = fit_reduced_ae(mz, dz)
        lrt_stat = max(0, 2 * (nll_red - nll_full))
        p_corrected = 0.5 * (1 - st.chi2.cdf(lrt_stat, df=1))
        rMZ, rDZ, A_falc, C_falc = falconer_moments(mz, dz)
        a_lo, a_hi, c_lo, c_hi = bootstrap_ci(mz, dz, n_boot=200)
        results.append({
            'Metric': metric, 'N_MZ': len(mz), 'N_DZ': len(dz),
            'A_pct': round(a2, 4), 'A_boot_lo': round(a_lo, 4), 'A_boot_hi': round(a_hi, 4),
            'C_pct': round(c2, 4), 'C_boot_lo': round(c_lo, 4), 'C_boot_hi': round(c_hi, 4),
            'rMZ': round(rMZ, 4), 'rDZ': round(rDZ, 4),
            'A_Falconer': round(A_falc, 4), 'C_Falconer': round(C_falc, 4),
            'DZ_negative_nonestimable': bool(rDZ < 0),
            'LRT_stat': round(lrt_stat, 4), 'LRT_p_boundary_corrected': round(p_corrected, 4),
        })
        flag = "  <-- rDZ negative, non-estimable under ACE" if rDZ < 0 else ""
        print(f"{metric:<14} {a2:>7.3f} {c2:>7.3f} {rMZ:>7.3f} {rDZ:>7.3f} {A_falc:>8.3f} {C_falc:>8.3f} {p_corrected:>10.4f}{flag}")

    res_df = pd.DataFrame(results)
    _, q, _, _ = multipletests(res_df['LRT_p_boundary_corrected'], method='fdr_bh')
    res_df['FDR_q_boundary_corrected'] = np.round(q, 4)
    res_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    res_df['A_ML_vs_Falconer_diff'] = (res_df['A_pct'] - res_df['A_Falconer']).abs()
    print(f"\nMax |A(ML) - A(Falconer)| across all 16 tensor metrics: {res_df['A_ML_vs_Falconer_diff'].max():.4f}")
    n_nonestimable = res_df['DZ_negative_nonestimable'].sum()
    print(f"Networks with negative rDZ (non-estimable under ACE): {n_nonestimable}/16")
    print("(Falconer is a completely independent, assumption-light method-of-moments estimator --")
    print(" it is what identifies the negative-rDZ, non-estimable networks directly.)")
