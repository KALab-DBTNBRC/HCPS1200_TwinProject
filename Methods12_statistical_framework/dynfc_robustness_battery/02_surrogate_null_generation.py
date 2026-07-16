#!/usr/bin/env python3
"""
dynfc_surrogate_null.py
-----------------------
Multivariate phase-randomized (MVPR) surrogate null for dynamic fluidity.
Surrogates preserve each subject's static FC and power spectra (cross-spectral
density) and destroy only genuine non-stationary dynamics -> the principled null
"no dynamics beyond what static FC + spectra imply" (Prichard & Theiler 1994;
Liegeois et al. 2017, NeuroImage).

The fluidity computation here is COPIED VERBATIM from DynamicFluidity.py so the
surrogate path cannot diverge from the real path. If DynamicFluidity.py changes,
re-sync the WINDOW/STEP/network_indices block below.

Outputs (kept as record):
  dynfc_surrogate_subject.csv     per-subject obs / null_mean / null_sd / z / p_emp / excess (per metric)
  dynfc_surrogate_summary.csv     per-metric group test (mean obs vs null, frac subjects p<.05)
  rsfMRI_Tier3_Dynamic_Excess.csv per-subject EXCESS fluidity (same column names) -> feeds script 3

Env: numpy, pandas, scipy  (conda env dtiproject). Heavy compute -> uses all cores.
"""
import os, glob, time
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings; warnings.filterwarnings('ignore')

# ---- CONFIG (match DynamicFluidity.py) ------------------------------------
BASE_TS   = os.environ.get("TSDIR", os.path.join(os.environ.get("PROJECT_ROOT", "."), "Native_Timeseries"))
OUT_DIR   = os.environ.get("PROJECT_ROOT", ".")
N_SURR    = int(os.environ.get('NSURR', '500'))     # drop to 200 for a fast first pass
RUN_LEN   = int(os.environ.get('RUNLEN', '1200'))   # HCP rest run length; per-run surrogating if T % RUN_LEN == 0
MAX_WORKERS = int(os.environ.get('WORKERS', '70'))
SEED_BASE = 12345

WINDOW_SIZE, STEP_SIZE = 100, 20
network_indices = {
    'Reward':   [140,320,122,302,138,318,115,295,141,321,142,322,143,323,216,396,214,394,215,395,161,341,162,342,111,291,230,410,22,47,23,48,15,40,16,41,17,42,18,43,11,36,12,37,13,38,14,39,19,44,20,45],
    'Salience': [161,341,162,342,159,339,107,287,109,289,90,270,129,309,130,310,131,311,163,343,165,345,164,344,19,44,20,45,9,34],
    'DMN':      [211,391,212,392,85,265,83,263,84,264,64,244,115,295,122,302,138,318,77,257,80,260,200,380,193,373,178,358,176,356,177,357,205,385,1,26,2,27,3,28,4,29],
    'Olfactory':[160,340,168,348,169,349,181,361,222,402,143,323,216,396,142,322,214,394,19,44,20,45,1,26,2,27,3,28,4,29],
}
METRICS = ['Global_Dynamic_Fluidity','Dynamic_Fluidity_Reward','Dynamic_Fluidity_Salience',
           'Dynamic_Fluidity_DMN','Dynamic_Fluidity_Olfactory']
# ---------------------------------------------------------------------------

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def compute_fluidity(ts):
    """VERBATIM logic from DynamicFluidity.py. ts: (n_trs, 410)."""
    n_trs, n_parcels = ts.shape
    n_windows = (n_trs - WINDOW_SIZE) // STEP_SIZE
    if n_windows <= 0:
        return None
    triu = np.triu_indices(n_parcels, k=1)
    edge_dyn = np.empty((n_windows, triu[0].size))
    for w in range(n_windows):
        s = w * STEP_SIZE
        edge_dyn[w] = fisher_z(np.corrcoef(ts[s:s+WINDOW_SIZE].T))[triu]
    edge_var = np.var(edge_dyn, axis=0)
    out = {'Global_Dynamic_Fluidity': float(np.mean(edge_var))}
    var_mat = np.zeros((n_parcels, n_parcels)); var_mat[triu] = edge_var
    var_mat += var_mat.T
    for net, ids in network_indices.items():
        idx = [i-1 for i in ids if 0 <= i-1 < n_parcels]
        sub = var_mat[np.ix_(idx, idx)]
        out[f'Dynamic_Fluidity_{net}'] = float(np.nanmean(sub[np.triu_indices(len(idx), k=1)]))
    return out

def mvpr_block(X, rng):
    """Phase-randomize one (T, N) block, shared phases across columns -> preserves cross-spectrum."""
    T = X.shape[0]
    Xf = np.fft.rfft(X, axis=0)
    phi = rng.uniform(0, 2*np.pi, Xf.shape[0]); phi[0] = 0.0
    if T % 2 == 0: phi[-1] = 0.0
    return np.fft.irfft(Xf * np.exp(1j*phi)[:, None], n=T, axis=0)

def mvpr_surrogate(ts, rng):
    """Per-run surrogating when divisible, else whole-series."""
    T = ts.shape[0]
    if RUN_LEN > 0 and T % RUN_LEN == 0 and T // RUN_LEN > 1:
        return np.vstack([mvpr_block(ts[r*RUN_LEN:(r+1)*RUN_LEN], rng) for r in range(T // RUN_LEN)])
    return mvpr_block(ts, rng)

def process_subject(fp):
    sid = int(os.path.basename(fp).split('_')[0])
    try:
        ts = np.load(fp)
        obs = compute_fluidity(ts)
        if obs is None:
            return None
        rng = np.random.default_rng(SEED_BASE + sid)
        null = {m: np.empty(N_SURR) for m in METRICS}
        for k in range(N_SURR):
            f = compute_fluidity(mvpr_surrogate(ts, rng))
            for m in METRICS:
                null[m][k] = f[m]
        row = {'Subject': sid}
        for m in METRICS:
            nm, ns = null[m].mean(), null[m].std(ddof=1)
            row[f'{m}_obs'] = obs[m]
            row[f'{m}_nullmean'] = nm
            row[f'{m}_nullsd'] = ns
            row[f'{m}_z'] = (obs[m]-nm)/ns if ns > 0 else np.nan
            row[f'{m}_p'] = (np.sum(null[m] >= obs[m]) + 1)/(N_SURR + 1)   # one-sided obs>null
            row[f'{m}_excess'] = obs[m] - nm
        return row
    except Exception as e:
        print(f"[{sid}] FAILED: {e}")
        return None

def main():
    files = sorted(glob.glob(os.path.join(BASE_TS, '*.npy')))
    print(f"{time.strftime('%H:%M:%S')}  {len(files)} subjects, N_SURR={N_SURR}, workers={MAX_WORKERS}")
    if not files:
        raise SystemExit(f"No .npy in {BASE_TS}")
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_subject, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r: rows.append(r)
            if i % 20 == 0: print(f"  {i}/{len(files)} done")
    df = pd.DataFrame(rows).sort_values('Subject')
    df.to_csv(os.path.join(OUT_DIR, 'dynfc_surrogate_subject.csv'), index=False)

    # per-metric summary + group test (observed vs own null mean, paired)
    srows = []
    for m in METRICS:
        obs, nm = df[f'{m}_obs'], df[f'{m}_nullmean']
        t, p = stats.wilcoxon(obs - nm)
        srows.append({'Metric': m,
                      'Mean_obs': round(obs.mean(), 5),
                      'Mean_null': round(nm.mean(), 5),
                      'Mean_excess': round((obs-nm).mean(), 5),
                      'Frac_subj_p<.05': round((df[f'{m}_p'] < 0.05).mean(), 3),
                      'Group_wilcoxon_p': p})
    summ = pd.DataFrame(srows)
    summ.to_csv(os.path.join(OUT_DIR, 'dynfc_surrogate_summary.csv'), index=False)

    # excess table (same column names) -> feeds script 3
    excess = df[['Subject'] + [f'{m}_excess' for m in METRICS]].copy()
    excess.columns = ['Subject'] + METRICS
    excess.to_csv(os.path.join(OUT_DIR, 'rsfMRI_Tier3_Dynamic_Excess.csv'), index=False)

    print("\nSaved: dynfc_surrogate_subject.csv, dynfc_surrogate_summary.csv, rsfMRI_Tier3_Dynamic_Excess.csv")
    print(summ.to_string(index=False))
    print("\nInterpretation: Mean_excess>0 with high Frac_subj_p<.05 => fluidity exceeds the")
    print("stationary sampling floor (genuine dynamics). Then run script 3 on the excess table.")

if __name__ == '__main__':
    main()
