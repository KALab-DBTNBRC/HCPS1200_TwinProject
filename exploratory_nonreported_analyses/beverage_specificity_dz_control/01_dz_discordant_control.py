"""
dz_discordant_control.py
========================
Genetic specificity control for the MZ discordant beverage findings.

DZ twins share ~50% of genetic variance (and 0% of non-shared genetic
variance beyond that). If the beverage-specific white matter effect
is driven by the full genetic control afforded by MZ pairs, DZ
discordant pairs should show attenuated effects.

This analysis:
1. Identifies DZ discordant pairs (one Severity=0, one Severity>0)
2. Computes within-pair deltas identically to the MZ analysis
3. Runs the same Spearman correlations with permutation-based p-values
4. Directly compares MZ vs DZ effect sizes

Interpretation guide:
    MZ |r| >> DZ |r|  → genetic control amplifies the signal
                         → supports causal interpretation
    MZ |r| ≈ DZ |r|   → effect does not require full genetic control
                         → causal claim requires qualification
    DZ |r| > MZ |r|   → unexpected, investigate data quality

Inputs:  network_roi_metrics_FINAL.csv
Outputs: dz_control_results.csv
         dz_mz_comparison.csv
         dz_control_summary.txt
         dz_control_plots/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PATHS─────────────────
CSV     = os.path.join(os.environ.get("PROJECT_ROOT", "."), "twintables", "network_roi_metrics_FINAL.csv")
OUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "dzdiscordantcontrol")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)

N_PERMUTATIONS = 10_000
RANDOM_SEED    = 42

TARGETS = [
    ('Total_Wine_7days',            'FD_Olfactory',   'Wine × FD Olfactory'),
    ('Total_Wine_7days',            'FD_Reward',      'Wine × FD Reward'),
    ('Total_Hard_Liquor_7days',     'FD_Olfactory',   'Hard Liquor × FD Olfactory'),
    ('Total_Hard_Liquor_7days',     'FD_Reward',      'Hard Liquor × FD Reward'),
    ('Total_Beer_Wine_Cooler_7days','FDC_Salience',   'Beer × FDC Salience'),
]

MZ_RESULTS = {
    'Wine × FD Olfactory':         {'r': -0.658, 'p': 0.0016, 'n': 20},
    'Wine × FD Reward':            {'r': -0.594, 'p': 0.0058, 'n': 20},
    'Hard Liquor × FD Olfactory':  {'r': -0.465, 'p': 0.0390, 'n': 20},
    'Hard Liquor × FD Reward':     {'r': -0.445, 'p': 0.0490, 'n': 20},
    'Beer × FDC Salience':         {'r': -0.474, 'p': 0.0346, 'n': 20},
}

def build_delta_df(df, zygosity):
    zyg_df = df[df['ZygosityGT1'] == zygosity].copy()
    pairs  = zyg_df.groupby('TwinPairID')

    delta_rows = []
    pair_ids   = []
    sev_pairs  = []   

    bev_cols    = ['Total_Wine_7days', 'Total_Hard_Liquor_7days',
                   'Total_Beer_Wine_Cooler_7days', 'Total_Malt_Liquor_7days']
    metric_cols = [c for c in df.columns if any(
        c.startswith(m) for m in ['FD_', 'FC_', 'FDC_'])]

    for pair_id, group in pairs:
        if len(group) != 2:
            continue
        sevs = group['Severity'].values
        if not (0 in sevs and any(s > 0 for s in sevs)):
            continue

        h = group[group['Severity'] == 0].iloc[0]
        s = group[group['Severity'] >  0].iloc[0]

        row = {}
        for col in bev_cols + metric_cols:
            if col in group.columns:
                h_val = float(h[col]) if pd.notnull(h[col]) else 0.0
                s_val = float(s[col]) if pd.notnull(s[col]) else 0.0
                row[f'Delta_{col}'] = s_val - h_val

        delta_rows.append(row)
        pair_ids.append(pair_id)
        sev_pairs.append(int(s['Severity']))

    delta_df = pd.DataFrame(delta_rows, index=pair_ids)
    delta_df.index.name = 'TwinPairID'
    delta_df['Severity_Affected'] = sev_pairs
    return delta_df

def permutation_spearman(x, y, n_perm=10_000, seed=42):
    rng      = np.random.default_rng(seed)
    r_obs, _ = stats.spearmanr(x, y)

    null_r = np.empty(n_perm)
    for i in range(n_perm):
        signs    = rng.choice([-1, 1], size=len(x))
        x_perm   = x * signs
        r_null, _ = stats.spearmanr(x_perm, y)
        null_r[i] = r_null

    p_perm = np.mean(np.abs(null_r) >= np.abs(r_obs))
    return r_obs, p_perm, null_r

def cohens_q(r1, r2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    return z1 - z2

def main():
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv(CSV, low_memory=False)

    mz_delta = build_delta_df(df, 'MZ')
    dz_delta = build_delta_df(df, 'DZ')

    print(f'MZ discordant pairs: {len(mz_delta)}')
    print(f'DZ discordant pairs: {len(dz_delta)}\n')

    if len(dz_delta) < 4:
        print('WARNING: Fewer than 4 DZ discordant pairs. Analysis underpowered.')
        print('Report n and present as descriptive only.\n')

    dz_results  = []
    summary_lines = []

    for bev_col, metric_col, label in TARGETS:
        dbev = f'Delta_{bev_col}'
        dmet = f'Delta_{metric_col}'

        print(f'=== {label} ===')

        # MZ
        mz_valid = mz_delta[[dbev, dmet]].dropna()
        if len(mz_valid) < 4:
            print(f'  MZ: insufficient data (n={len(mz_valid)})')
        else:
            r_mz, p_mz, _ = permutation_spearman(
                mz_valid[dbev].values, mz_valid[dmet].values,
                n_perm=N_PERMUTATIONS, seed=RANDOM_SEED)
            print(f'  MZ: r = {r_mz:.3f}, perm p = {p_mz:.4f}, n = {len(mz_valid)}')

        if label in MZ_RESULTS:
            r_mz_known = MZ_RESULTS[label]['r']
            n_mz_known = MZ_RESULTS[label]['n']
        else:
            r_mz_known = r_mz if len(mz_valid) >= 4 else np.nan
            n_mz_known = len(mz_valid)

        # DZ
        dz_valid = dz_delta[[dbev, dmet]].dropna()
        n_dz     = len(dz_valid)

        if n_dz < 4:
            print(f'  DZ: n={n_dz} — descriptive only, no inference')
            r_dz  = float(stats.spearmanr(dz_valid[dbev], dz_valid[dmet])[0]) \
                    if n_dz >= 3 else np.nan
            p_dz  = np.nan
            null_dz = np.array([])
        else:
            r_dz, p_dz, null_dz = permutation_spearman(
                dz_valid[dbev].values, dz_valid[dmet].values,
                n_perm=N_PERMUTATIONS, seed=RANDOM_SEED)

        # BUG FIX: Pre-format the p-value string 
        p_dz_str = f"{p_dz:.4f}" if not np.isnan(p_dz) else "n/a"
        print(f'  DZ: r = {r_dz:.3f}, perm p = {p_dz_str}, n = {n_dz}')

        # COMPARISON
        if not np.isnan(r_dz) and not np.isnan(r_mz_known):
            q  = cohens_q(r_mz_known, r_dz)
            attenuation = (abs(r_mz_known) - abs(r_dz)) / abs(r_mz_known) * 100
            pattern = ('MZ > DZ — genetic control amplifies signal'
                       if abs(r_mz_known) > abs(r_dz) + 0.10
                       else 'MZ ≈ DZ — genetic control minimal'
                       if abs(abs(r_mz_known) - abs(r_dz)) <= 0.10
                       else 'DZ > MZ — investigate')
            print(f'  Cohen q = {q:.3f}, attenuation = {attenuation:.1f}%')
            print(f'  Pattern: {pattern}')
        else:
            attenuation = np.nan
            q = np.nan
            pattern = 'Insufficient DZ n for comparison'

        print()

        summary_lines.append(
            f'{label}: MZ r={r_mz_known:.3f} (n={n_mz_known}) | '
            f'DZ r={r_dz:.3f} (n={n_dz}, p={p_dz_str}) | '
            f'Attenuation={attenuation:.1f}% | {pattern}'
        )

        dz_results.append({
            'Analysis': label, 'Beverage': bev_col, 'Metric': metric_col,
            'MZ_r': r_mz_known, 'MZ_n': n_mz_known,
            'DZ_r': r_dz, 'DZ_n': n_dz, 'DZ_perm_p': p_dz,
            'Cohens_q': q, 'Attenuation_pct': attenuation,
            'Pattern': pattern
        })

        # FIGURE
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0A0A0A')
        fig.patch.set_facecolor('#0A0A0A')

        ax = axes[0]
        ax.set_facecolor('#0A0A0A')
        if len(mz_valid) >= 3:
            ax.scatter(mz_valid[dbev], mz_valid[dmet],
                       color='#2980B9', alpha=0.85, s=65,
                       edgecolors='white', linewidths=0.5, zorder=3,
                       label=f'MZ pairs (n={len(mz_valid)})')
            slope, intercept, *_ = stats.linregress(mz_valid[dbev], mz_valid[dmet])
            xr = np.linspace(mz_valid[dbev].min(), mz_valid[dbev].max(), 100)
            ax.plot(xr, slope*xr+intercept, '--', color='#2980B9', linewidth=1.5, alpha=0.7)
        ax.axhline(0, color='#444', linewidth=0.5)
        ax.axvline(0, color='#444', linewidth=0.5)
        ax.set_title(f'MZ Discordant  ρ = {r_mz_known:.3f}', color='white', fontsize=11)
        ax.set_xlabel('Δ Beverage (affected−ctrl)', color='#AAAAAA', fontsize=9)
        ax.set_ylabel(f'Δ {metric_col}', color='#AAAAAA', fontsize=9)
        ax.tick_params(colors='#AAAAAA')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

        ax2 = axes[1]
        ax2.set_facecolor('#0A0A0A')
        if len(dz_valid) >= 3:
            ax2.scatter(dz_valid[dbev], dz_valid[dmet],
                        color='#E67E22', alpha=0.85, s=65,
                        edgecolors='white', linewidths=0.5, zorder=3,
                        label=f'DZ pairs (n={n_dz})')
            if n_dz >= 4:
                slope2, intercept2, *_ = stats.linregress(dz_valid[dbev], dz_valid[dmet])
                xr2 = np.linspace(dz_valid[dbev].min(), dz_valid[dbev].max(), 100)
                ax2.plot(xr2, slope2*xr2+intercept2, '--', color='#E67E22', linewidth=1.5, alpha=0.7)
        ax2.axhline(0, color='#444', linewidth=0.5)
        ax2.axvline(0, color='#444', linewidth=0.5)
        r_dz_str_plot = f'{r_dz:.3f}' if not np.isnan(r_dz) else 'n/a'
        ax2.set_title(f'DZ Discordant  ρ = {r_dz_str_plot}  (n={n_dz})', color='white', fontsize=11)
        ax2.set_xlabel('Δ Beverage (affected−ctrl)', color='#AAAAAA', fontsize=9)
        ax2.set_ylabel(f'Δ {metric_col}', color='#AAAAAA', fontsize=9)
        ax2.tick_params(colors='#AAAAAA')
        for sp in ax2.spines.values(): sp.set_edgecolor('#333')

        ax3 = axes[2]
        ax3.set_facecolor('#0A0A0A')
        bars_data = [r_mz_known, r_dz if not np.isnan(r_dz) else 0]
        bar_labels = [f'MZ\n(n={n_mz_known})', f'DZ\n(n={n_dz})']
        bar_colors = ['#2980B9', '#E67E22']
        b = ax3.bar(bar_labels, bars_data, color=bar_colors, alpha=0.85,
                    width=0.5, edgecolor='white', linewidth=0.5)
        ax3.axhline(0, color='white', linewidth=0.5)
        for bar, val in zip(b, bars_data):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     val - 0.04 if val < 0 else val + 0.02,
                     f'{val:.3f}', ha='center', va='top' if val < 0 else 'bottom',
                     color='white', fontsize=11, fontweight='bold')
        att_str = f'{attenuation:.1f}% attenuation' if not np.isnan(attenuation) else ''
        ax3.set_title(f'MZ vs DZ Effect Size\n{att_str}', color='white', fontsize=11)
        ax3.set_ylabel('Spearman ρ', color='#AAAAAA', fontsize=9)
        ax3.tick_params(colors='#AAAAAA')
        for sp in ax3.spines.values(): sp.set_edgecolor('#333')
        ax3.set_ylim(min(bars_data) - 0.2, max(bars_data) + 0.2)

        fig.suptitle(f'Genetic Specificity Control: {label}', color='white', fontsize=13, y=1.02)
        plt.tight_layout(pad=2)

        fname = label.replace(' × ', '_').replace(' ', '_') + '_dz_control.png'
        fig.savefig(os.path.join(OUT_DIR, 'plots', fname),
                    dpi=300, bbox_inches='tight', facecolor='#0A0A0A', edgecolor='none')
        plt.close()
        print(f'  Figure saved: plots/{fname}\n')

    pd.DataFrame(dz_results).to_csv(os.path.join(OUT_DIR, 'dz_control_results.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'dz_control_summary.txt'), 'w') as f:
        f.write('DZ DISCORDANT PAIR CONTROL — SUMMARY\n')
        f.write('=' * 70 + '\n\n')
        f.write('INTERPRETATION:\n')
        f.write('  MZ > DZ attenuation (>20%)  → genetic control amplifies signal\n')
        f.write('  MZ ≈ DZ (attenuation <10%)  → effect independent of genetic control\n')
        f.write('\nRESULTS:\n')
        for line in summary_lines:
            f.write(line + '\n')

    print('=' * 70)
    print('DZ CONTROL SUMMARY')
    print('=' * 70)
    for line in summary_lines:
        print(line)

if __name__ == '__main__':
    main()