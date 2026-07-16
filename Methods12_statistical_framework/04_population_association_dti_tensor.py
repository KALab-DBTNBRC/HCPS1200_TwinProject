#!/usr/bin/env python3
"""
lmemodel_dti_standardized.py
==============================
Standardized-beta version of lmemodel_dti.py (the reconstruction),
same relationship as lmemodel_standardized.py is to lmemodel.py.

"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
import os

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "MASTER_ROI_METRICS_DTI_FBA.csv")
OUTPUT_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "DTIvsFBA", "lme_results_FDR_DTI_STANDARDIZED.csv")

POTENTIAL_COVS = [
    'Severity', 'Age_in_Yrs', 'Gender', 'SSAGA_TB_Still_Smoking',
    'Race', 'Ethnicity', 'SSAGA_Times_Used_Illicits',
    'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc'
]
METRICS = ['MD', 'FA', 'RD', 'AD']
NETWORKS = ['Reward', 'Salience', 'DMN', 'Olfactory']

CONTINUOUS_COVS_TO_STANDARDIZE = [
    'Age_in_Yrs', 'SSAGA_Times_Used_Illicits',
    'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc'
]


def get_valid_covariates(df, target_cols, potential_covs):
    clean_df = df.dropna(subset=target_cols + potential_covs).copy()
    valid_covs = []
    for cov in potential_covs:
        if cov == 'Severity':
            continue
        if clean_df[cov].nunique() > 1:
            if cov in ['Race', 'Ethnicity']:
                valid_covs.append(f"C({cov})")
            else:
                valid_covs.append(cov)
    return valid_covs, clean_df


def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)


def run_standardized_dti_lme_model():
    if not os.path.exists(CSV_PATH):
        print(f"CRITICAL ERROR: Could not find {CSV_PATH}.")
        return

    df = pd.read_csv(CSV_PATH, low_memory=False)
    all_results = []

    print(f"Starting STANDARDIZED DTI LME models on N={len(df)} subjects...")
    for metric in METRICS:
        for net in NETWORKS:
            target = f"{metric}_{net}"
            if target not in df.columns:
                print(f"  > Skipping {target} (Column missing)")
                continue

            valid_covs, df_model = get_valid_covariates(df, [target], POTENTIAL_COVS)
            valid_covs.insert(0, 'Severity')

            df_model = df_model.copy()
            df_model[target] = zscore(df_model[target])
            df_model['Severity'] = zscore(df_model['Severity'])
            for cov in CONTINUOUS_COVS_TO_STANDARDIZE:
                if cov in df_model.columns:
                    df_model[cov] = zscore(df_model[cov])

            formula = f"{target} ~ " + " + ".join(valid_covs)
            try:
                model = smf.mixedlm(formula, df_model, groups=df_model["TwinPairID"]).fit()
                ci = model.conf_int(alpha=0.05).loc['Severity']
                all_results.append({
                    'Metric': metric,
                    'Network': net,
                    'Standardized_Beta': model.params['Severity'],
                    'CI_lower': ci[0],
                    'CI_upper': ci[1],
                    'P_Value': model.pvalues['Severity'],
                    'N': len(df_model)
                })
                print(f"  [SUCCESS] {target} analyzed. std_beta={model.params['Severity']:.4f} "
                      f"[{ci[0]:.4f}, {ci[1]:.4f}]")
            except Exception as e:
                print(f"  [FAILED] {target} | Error: {e}")
                continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        rejected, q_values, _, _ = multipletests(
            results_df['P_Value'].values, alpha=0.05, method='fdr_bh')
        results_df['FDR_q_value'] = q_values
        results_df['Survives_FDR'] = rejected
        results_df = results_df.sort_values(by='FDR_q_value')
        results_df.to_csv(OUTPUT_PATH, index=False)
        print("\n--- STANDARDIZED DTI LME RESULTS (FDR CORRECTED) ---")
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        print(f"\nSaved: {OUTPUT_PATH}")
    else:
        print("\nProcess finished with zero results.")


if __name__ == "__main__":
    run_standardized_dti_lme_model()
