import os
"""
NCT Q1: Controllability Atlas
Produces all group-level summary outputs from the 8 pre-computed
controllability arrays (MIND/SC x AC T1/T3/T5 / MC).

Outputs (all -> NCT_inputs/controllability/summary/):
  group_stats.csv              360 nodes x all 8 metrics, mean + SD
  hub_table_{metric}.csv       all 360 nodes ranked by group mean
  substrate_comparison.csv     Pearson/Spearman r MIND vs SC for AC & MC
  ac_mc_correlations.csv       within-substrate AC-MC cross-correlations
  individual_diffs.csv         per-subject global mean AC, SD, Severity
  spotlight_nodes.csv          key olfactory, reward, assoc-cortex nodes
  nct_q1_report.txt            human-readable summary

Primary focus: T=3 for AC. T=1 and T=5 reported as sensitivity.
SC arrays log-transformed throughout (log-normal distribution of SIFT2 weights).
"""

import logging, time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

CTRL_DIR  = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs/Controllability"))
NCT_DIR   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "NCT_inputs"))
BEH_CSV   = Path(os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables/network_roi_metrics_FINAL.csv"))
OUT_DIR   = CTRL_DIR / 'summary'
OUT_DIR.mkdir(exist_ok=True)

N_NODES = 360

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()])
log = logging.getLogger('NCT_Q1')

SPOTLIGHT = {
    'Olfactory_primary': [
        'R_Pir_ROI','L_Pir_ROI',
        'R_52_ROI','L_52_ROI',
        'R_RI_ROI','L_RI_ROI',
        'R_EC_ROI','L_EC_ROI',
        'R_PreS_ROI','L_PreS_ROI',
    ],
    'OFC_reward': [
        'R_OFC_ROI','L_OFC_ROI',
        'R_pOFC_ROI','L_pOFC_ROI',
        'R_Iai_ROI','L_Iai_ROI',
        'R_AAIC_ROI','L_AAIC_ROI',
        'R_25_ROI','L_25_ROI',
        'R_a24pr_ROI','L_a24pr_ROI',
    ],
    'Association_hubs': [
        'R_PGi_ROI','L_PGi_ROI',
        'R_PGp_ROI','L_PGp_ROI',
        'R_PH_ROI','L_PH_ROI',
        'R_46_ROI','L_46_ROI',
        'R_45_ROI','L_45_ROI',
        'R_47l_ROI','L_47l_ROI',
    ],
}


def load_all():
    log.info('Loading controllability arrays...')
    subj_idx = pd.read_csv(str(CTRL_DIR / 'subject_index.csv'))
    node_df  = pd.read_csv(str(NCT_DIR  / 'node_names.csv'))
    node_names = node_df['Region'].tolist()
    beh = pd.read_csv(str(BEH_CSV))

    arrays = {}
    for tag, fname in [
        ('MIND_AC_T1', 'MIND_AC_T1.npy'),
        ('MIND_AC_T3', 'MIND_AC_T3.npy'),
        ('MIND_AC_T5', 'MIND_AC_T5.npy'),
        ('MIND_MC',    'MIND_MC.npy'),
        ('SC_AC_T1',   'SC_AC_T1.npy'),
        ('SC_AC_T3',   'SC_AC_T3.npy'),
        ('SC_AC_T5',   'SC_AC_T5.npy'),
        ('SC_MC',      'SC_MC.npy'),
    ]:
        arr = np.load(str(CTRL_DIR / fname)).astype(np.float64)
        assert arr.shape == (len(subj_idx), N_NODES), \
            f'{fname}: expected {(len(subj_idx), N_NODES)}, got {arr.shape}'
        arrays[tag] = arr

    return arrays, subj_idx, node_names, beh


def log_transform_sc(arrays):
    """
    Apply log to all SC arrays in-place (returns new dict).
    Minimum SC value floor > 0 confirmed (audit: min=3.0 for AC,
    min=0.13 for MC), so no epsilon shift needed.
    """
    out = {}
    for k, v in arrays.items():
        if k.startswith('SC'):
            out[k] = np.log(v)
        else:
            out[k] = v
    return out


def compute_group_stats(arrays, node_names):
    rows = {'Region': node_names}
    for tag, arr in arrays.items():
        rows[f'{tag}_mean'] = arr.mean(axis=0)
        rows[f'{tag}_sd']   = arr.std(axis=0)
    return pd.DataFrame(rows)


def hub_table(arr, node_names, tag):
    group_mean = arr.mean(axis=0)
    group_sd   = arr.std(axis=0)
    group_sem  = group_sd / np.sqrt(arr.shape[0])
    cv         = group_sd / np.abs(group_mean)
    df = pd.DataFrame({
        'Region':     node_names,
        'Mean':       group_mean,
        'SD':         group_sd,
        'SEM':        group_sem,
        'CV':         cv,
        'Rank':       0,
    }).sort_values('Mean', ascending=False).reset_index(drop=True)
    df['Rank'] = np.arange(1, len(df) + 1)
    df['Metric'] = tag
    return df


def substrate_comparison(arrays):
    rows = []
    pairs = [
        ('MIND_AC_T1', 'SC_AC_T1', 'AC_T1'),
        ('MIND_AC_T3', 'SC_AC_T3', 'AC_T3'),
        ('MIND_AC_T5', 'SC_AC_T5', 'AC_T5'),
        ('MIND_MC',    'SC_MC',    'MC'),
    ]
    for mk, sk, label in pairs:
        m_gm = arrays[mk].mean(axis=0)
        s_gm = arrays[sk].mean(axis=0)
        pr, pp = stats.pearsonr(m_gm, s_gm)
        sr, sp = stats.spearmanr(m_gm, s_gm)
        rows.append({
            'Metric': label,
            'Pearson_r':    round(pr, 4),
            'Pearson_p':    round(pp, 6),
            'Spearman_rho': round(sr, 4),
            'Spearman_p':   round(sp, 6),
            'N_nodes': N_NODES,
        })
    return pd.DataFrame(rows)


def ac_mc_correlations(arrays):
    rows = []
    for substrate in ['MIND', 'SC']:
        ac_gm = arrays[f'{substrate}_AC_T3'].mean(axis=0)
        mc_gm = arrays[f'{substrate}_MC'].mean(axis=0)
        pr, _ = stats.pearsonr(ac_gm, mc_gm)
        sr, _ = stats.spearmanr(ac_gm, mc_gm)
        subj_r = [stats.pearsonr(arrays[f'{substrate}_AC_T3'][i],
                                  arrays[f'{substrate}_MC'][i])[0]
                  for i in range(arrays[f'{substrate}_AC_T3'].shape[0])]
        rows.append({
            'Substrate':            substrate,
            'Group_mean_Pearson_r': round(pr, 4),
            'Group_mean_Spearman':  round(sr, 4),
            'SubjLevel_r_mean':     round(np.mean(subj_r), 4),
            'SubjLevel_r_SD':       round(np.std(subj_r), 4),
        })
    return pd.DataFrame(rows)


def individual_diffs(arrays, subj_idx, beh):
    df = subj_idx.copy()
    for tag in ['MIND_AC_T3', 'SC_AC_T3', 'MIND_MC', 'SC_MC']:
        arr = arrays[tag]
        df[f'{tag}_global_mean'] = arr.mean(axis=1)
        df[f'{tag}_global_sd']   = arr.std(axis=1)
        df[f'{tag}_range']       = arr.max(axis=1) - arr.min(axis=1)

    beh_sub = beh[['Subject','Severity','Age_in_Yrs','Gender',
                   'Total_Wine_7days','FamHist_Combined_DrgAlc']].copy()
    df['subj_id'] = df['subj_id'].astype(int)
    beh_sub['Subject'] = beh_sub['Subject'].astype(int)
    df = df.merge(beh_sub, left_on='subj_id', right_on='Subject', how='left')

    for tag in ['MIND_AC_T3', 'SC_AC_T3']:
        col = f'{tag}_global_mean'
        valid = df.dropna(subset=[col, 'Severity'])
        r, p = stats.spearmanr(valid[col], valid['Severity'])
        log.info(f'  Global {tag} ~ Severity: Spearman rho={r:.4f}  p={p:.4f}')

    return df


def spotlight_table(arrays, node_names):
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    rows = []
    for net, regions in SPOTLIGHT.items():
        for reg in regions:
            if reg not in name_to_idx:
                continue
            idx = name_to_idx[reg]
            row = {'Network': net, 'Region': reg}
            for tag in ['MIND_AC_T3', 'MIND_MC', 'SC_AC_T3', 'SC_MC']:
                arr = arrays[tag]
                row[f'{tag}_mean'] = round(arr[:, idx].mean(), 6)
                row[f'{tag}_sd']   = round(arr[:, idx].std(),  6)
            rows.append(row)
    return pd.DataFrame(rows)


def bilateral_symmetry(arrays, node_names):
    rh = {n.replace('R_', ''): i for i, n in enumerate(node_names) if n.startswith('R_')}
    lh = {n.replace('L_', ''): i for i, n in enumerate(node_names) if n.startswith('L_')}
    shared = sorted(set(rh) & set(lh))

    rows = []
    for tag in ['MIND_AC_T3', 'SC_AC_T3']:
        gm = arrays[tag].mean(axis=0)
        rh_vals = np.array([gm[rh[k]] for k in shared])
        lh_vals = np.array([gm[lh[k]] for k in shared])
        r, p = stats.pearsonr(rh_vals, lh_vals)
        ratio = rh_vals.mean() / lh_vals.mean() if lh_vals.mean() > 0 else np.nan
        rows.append({
            'Metric':      tag,
            'N_homologues': len(shared),
            'Pearson_r':   round(r, 4),
            'P_value':     round(p, 8),
            'RH_mean':     round(rh_vals.mean(), 4),
            'LH_mean':     round(lh_vals.mean(), 4),
            'RH_LH_ratio': round(ratio, 4),
        })
    return pd.DataFrame(rows)


def write_report(group_stats, substrate_cmp, ac_mc_corr, bilateral, hub_tables, out_path):
    lines = [
        'NCT Q1: CONTROLLABILITY ATLAS -- SUMMARY REPORT',
        '=' * 60,
        f'N subjects: 238   N nodes: {N_NODES}   Substrates: MIND, SC(log)',
        '',
    ]
    for tag in ['MIND_AC_T3', 'SC_AC_T3', 'MIND_MC', 'SC_MC']:
        m = group_stats[f'{tag}_mean']
        lines.append(f'  {tag:<14}  mean={m.mean():.4f}  min={m.min():.4f}  max={m.max():.4f}')

    for _, row in substrate_cmp.iterrows():
        lines.append(f'  MIND vs SC({row.Metric}): Pearson r={row.Pearson_r}  Spearman rho={row.Spearman_rho}')

    for tag in ['MIND_AC_T3', 'SC_AC_T3', 'MIND_MC', 'SC_MC']:
        ht = hub_tables[tag]
        lines += ['', f'-- TOP-10 HUBS: {tag} --']
        for _, row in ht.head(10).iterrows():
            lines.append(f'  {int(row.Rank):>3}.  {row.Region:<30}  {row.Mean:.4f}')

    out_path.write_text('\n'.join(lines))


def main():
    t0 = time.time()
    log.info('NCT Q1: Controllability Atlas')

    arrays, subj_idx, node_names, beh = load_all()
    arrays = log_transform_sc(arrays)

    group_stats = compute_group_stats(arrays, node_names)
    group_stats.to_csv(str(OUT_DIR / 'group_stats.csv'), index=False)

    hub_tables = {}
    for tag in ['MIND_AC_T3', 'SC_AC_T3', 'MIND_MC', 'SC_MC',
                'MIND_AC_T1', 'MIND_AC_T5', 'SC_AC_T1', 'SC_AC_T5']:
        ht = hub_table(arrays[tag], node_names, tag)
        hub_tables[tag] = ht
        ht.to_csv(str(OUT_DIR / f'hub_table_{tag}.csv'), index=False)

    sub_cmp = substrate_comparison(arrays)
    sub_cmp.to_csv(str(OUT_DIR / 'substrate_comparison.csv'), index=False)

    ac_mc = ac_mc_correlations(arrays)
    ac_mc.to_csv(str(OUT_DIR / 'ac_mc_correlations.csv'), index=False)

    ind_diff = individual_diffs(arrays, subj_idx, beh)
    ind_diff.to_csv(str(OUT_DIR / 'individual_diffs.csv'), index=False)

    spot = spotlight_table(arrays, node_names)
    spot.to_csv(str(OUT_DIR / 'spotlight_nodes.csv'), index=False)

    bilat = bilateral_symmetry(arrays, node_names)
    bilat.to_csv(str(OUT_DIR / 'bilateral_symmetry.csv'), index=False)

    write_report(group_stats, sub_cmp, ac_mc, bilat, hub_tables, OUT_DIR / 'nct_q1_report.txt')

    wall = timedelta(seconds=int(time.time() - t0))
    log.info(f'Q1 COMPLETE   Wall time : {wall}   Outputs -> {OUT_DIR}/')
    log.info('Next: nct_heritability.R (Q2)')


if __name__ == '__main__':
    main()
