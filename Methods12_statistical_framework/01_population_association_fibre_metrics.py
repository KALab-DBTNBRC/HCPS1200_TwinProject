import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
import os

warnings.filterwarnings("ignore")

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

def run_ultimate_lme_model():
    csv_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "network_roi_metrics_FINAL.csv")
    
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: Could not find {csv_path}. Ensure the file exists.")
        return
        
    df = pd.read_csv(csv_path, low_memory=False)

    # DEFINITIVE COVARIATE LIST 
    potential_covs = [
        'Severity', 'Age_in_Yrs', 'Gender', 'SSAGA_TB_Still_Smoking',
        'Race', 'Ethnicity', 'SSAGA_Times_Used_Illicits', 
        'SSAGA_Mj_Times_Used', 'FamHist_Combined_DrgAlc'
    ]

    metrics = ['FDC', 'FD', 'FC']
    networks = ['Reward', 'Salience', 'DMN', 'Olfactory']
    all_results = []

    print(f"Starting LME models on N={len(df)} subjects...")

    for metric in metrics:
        for net in networks:
            target = f"{metric}_{net}"
            if target not in df.columns:
                print(f"  > Skipping {target} (Column missing)")
                continue

            valid_covs, df_model = get_valid_covariates(df, [target], potential_covs)
            
            # Ensure Severity is the primary regressor
            valid_covs.insert(0, 'Severity')

            formula = f"{target} ~ " + " + ".join(valid_covs)

            try:
                # FIT THE MIXED EFFECTS MODEL
                model = smf.mixedlm(formula, df_model, groups=df_model["TwinPairID"]).fit()
                
                all_results.append({
                    'Metric': metric, 
                    'Network': net,
                    'Ethanol_Beta': model.params['Severity'],
                    'P_Value': model.pvalues['Severity'], 
                    'N': len(df_model)
                })
                print(f"  [SUCCESS] {target} analyzed.")
                
            except Exception as e:
                print(f"  [FAILED] {target} | Error: {e}")
                continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # --- APPLY FDR CORRECTION ---
        p_values = results_df['P_Value'].values
        
        # Benjamini-Hochberg FDR correction
        rejected, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        results_df['FDR_q_value'] = q_values
        results_df['Survives_FDR'] = rejected
        
        # Sort by best q-value
        results_df = results_df.sort_values(by='FDR_q_value')

        output_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "final_stats", "lme_results_FDR.csv")
        results_df.to_csv(output_path, index=False)
        print("\n--- FINAL LME RESULTS (FDR CORRECTED) ---")
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        print(f"\nFinal LME results saved: {output_path}")
    else:
        print("\nProcess finished with zero results. Check your CSV headers.")

if __name__ == "__main__":
    run_ultimate_lme_model()