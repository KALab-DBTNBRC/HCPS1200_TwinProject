"""
dynfc_excess_reanalysis.py
--------------------------
The load-bearing robustness step. Re-runs the population LME (severity) and the
zygosity-split Mundlak decomposition on SURROGATE-CORRECTED excess fluidity
(observed - stationary-null mean) from script 2.

Decision rule:
  * population severity assoc AND DZ-within effect persist on excess  -> genetic
    signal is above the sampling-noise floor; report as in Fig 4.
  * they evaporate on excess  -> the dynamic result was sampling variability;
    demote dynamics to a null alongside static FC.

Set METRIC_FILE to the excess table; defaults to rsfMRI_Tier3_Dynamic_Excess.csv.
(Point it at rsfMRI_Tier3_Dynamic_Metrics.csv to re-confirm raw-metric numbers.)

Outputs:
  dynfc_excess_population_LME.csv
  mundlak_DynamicFC_excess_zygosity_split.csv

Env: pandas, numpy, scipy, statsmodels
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

BASE        = os.environ.get("PROJECT_ROOT", ".")
MASTER      = os.path.join(BASE, 'MASTER_ROI_METRICS_DTI_FBA.csv')
METRIC_FILE = os.environ.get('METRIC_FILE', os.path.join(BASE, 'rsfMRI_Tier3_Dynamic_Excess.csv'))
TAG         = os.environ.get('TAG', 'excess')

METRICS = ['Global_Dynamic_Fluidity','Dynamic_Fluidity_Reward','Dynamic_Fluidity_Salience',
           'Dynamic_Fluidity_DMN','Dynamic_Fluidity_Olfactory']
COVARS  = ['Age_in_Yrs', 'C(Gender)', 'SSAGA_TB_Still_Smoking']


def load():
    m = pd.read_csv(MASTER)[['Subject','TwinPairID','ZygosityGT1','Severity',
                             'Age_in_Yrs','Gender','SSAGA_TB_Still_Smoking']]
    d = pd.read_csv(METRIC_FILE)
    return m.merge(d, on='Subject')


def population_lme(df, metric):
    sub = df.dropna(subset=[metric, 'Severity']).copy()
    sub['Y'] = sub[metric]
    sd_y, sd_s = sub['Y'].std(), sub['Severity'].std()
    mod = smf.mixedlm("Y ~ Severity + " + " + ".join(COVARS), sub,
                      groups=sub['TwinPairID']).fit(reml=False)
    ci = mod.conf_int().loc['Severity']
    return {'Metric': metric, 'N': len(sub),
            'Beta': mod.params['Severity']*sd_s/sd_y,
            'CI_lo': ci[0]*sd_s/sd_y, 'CI_hi': ci[1]*sd_s/sd_y,
            'P': mod.pvalues['Severity']}


def mundlak(df, metric):
    sub = df.dropna(subset=[metric, 'Severity']).copy()
    pm = sub.groupby('TwinPairID')['Severity'].transform('mean')
    sub['Severity_BF'], sub['Severity_WF'] = pm, sub['Severity'] - pm
    sub['Y'] = sub[metric]
    sd_y, sd_wf, sd_bf = sub['Y'].std(), sub['Severity_WF'].std(), sub['Severity_BF'].std()
    mod = smf.mixedlm("Y ~ Severity_BF + Severity_WF + " + " + ".join(COVARS), sub,
                      groups=sub['TwinPairID']).fit(reml=False)
    return {'Metric': metric, 'N': len(sub),
            'Beta_BF': mod.params['Severity_BF']*sd_bf/sd_y if sd_bf > 0 else np.nan,
            'P_BF': mod.pvalues['Severity_BF'],
            'Beta_WF': mod.params['Severity_WF']*sd_wf/sd_y if sd_wf > 0 else np.nan,
            'P_WF': mod.pvalues['Severity_WF']}


def split_group(df, label):
    out = pd.DataFrame([dict(mundlak(df, m), Group=label) for m in METRICS])
    out['FDR_BF'] = stats.false_discovery_control(out['P_BF'])
    out['FDR_WF'] = stats.false_discovery_control(out['P_WF'])
    return out


def main():
    df = load()
    print(f"[{TAG}] N merged = {len(df)}  (metric file: {os.path.basename(METRIC_FILE)})")

    pop = pd.DataFrame([population_lme(df, m) for m in METRICS])
    pop['FDR_q'] = stats.false_discovery_control(pop['P'])
    pop = pop.round(4)
    pop.to_csv(os.path.join(BASE, f'dynfc_{TAG}_population_LME.csv'), index=False)
    print("\nPopulation LME:\n", pop.to_string(index=False))

    split = pd.concat([split_group(df, 'Pooled'),
                       split_group(df[df.ZygosityGT1 == 'MZ'], 'MZ_only'),
                       split_group(df[df.ZygosityGT1 == 'DZ'], 'DZ_only')],
                      ignore_index=True)
    split = split[['Group','Metric','N','Beta_BF','P_BF','FDR_BF','Beta_WF','P_WF','FDR_WF']].round(4)
    split.to_csv(os.path.join(BASE, f'mundlak_DynamicFC_{TAG}_zygosity_split.csv'), index=False)
    print("\nZygosity split:\n", split.to_string(index=False))

    mz = split[split.Group == 'MZ_only']; dz = split[split.Group == 'DZ_only']
    print("\n=== VERDICT ON EXCESS ===")
    print(f"population severity assoc surviving FDR: {(pop.FDR_q < 0.05).sum()}/{len(pop)}")
    print(f"MZ-only WF p<0.05: {(mz.P_WF < 0.05).sum()}/{len(mz)} | DZ-only WF p<0.05: {(dz.P_WF < 0.05).sum()}/{len(dz)}")
    print("Persisting population + DZ-carried, MZ-null pattern => genuine genetic signal above noise floor.")


if __name__ == '__main__':
    main()
