import os
import glob
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION & PATHS
BASE_DIR = os.environ.get("PROJECT_ROOT", ".")
INPUT_DIR = os.path.join(BASE_DIR, "Native_Timeseries")
OUTPUT_CSV = os.path.join(BASE_DIR, "Tables/rsfMRI_Tier2_Decomposed_Metrics.csv")

MAX_WORKERS = 30

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

# DECOMPOSED TOPOLOGICAL INDICES
networks = {
    'Reward_Cortical': [140, 320, 122, 302, 138, 318, 115, 295, 141, 321, 142, 322, 143, 323, 216, 396, 214, 394, 215, 395, 161, 341, 162, 342, 111, 291, 230, 410],
    'Reward_Subcortical': [22, 47, 23, 48, 15, 40, 16, 41, 17, 42, 18, 43, 11, 36, 12, 37, 13, 38, 14, 39, 19, 44, 20, 45],
    
    'Olfactory_Cortical': [160, 340, 168, 348, 169, 349, 181, 361, 222, 402, 143, 323, 216, 396, 142, 322, 214, 394],
    'Olfactory_Subcortical': [19, 44, 20, 45, 1, 26, 2, 27, 3, 28, 4, 29],
    
    'DMN_Cortical': [211, 391, 212, 392, 85, 265, 83, 263, 84, 264, 64, 244, 115, 295, 122, 302, 138, 318, 77, 257, 80, 260, 200, 380, 193, 373, 178, 358, 176, 356, 177, 357, 205, 385],
    'DMN_Subcortical': [1, 26, 2, 27, 3, 28, 4, 29],
    
    'Salience_Cortical': [161, 341, 162, 342, 159, 339, 107, 287, 109, 289, 90, 270, 129, 309, 130, 310, 131, 311, 163, 343, 165, 345, 164, 344],
    'Salience_Subcortical': [19, 44, 20, 45, 9, 34]
}

# TARGETED HUB NODES FOR IFOF/UNCINATE AXIS
olf_primary = [160, 340, 168, 348, 169, 349]  # Piriform, EC, PreS
rew_gateway = [143, 323, 216, 396, 22, 47, 23, 48, 19, 44, 20, 45]  # OFC, NAc, AMY

# Combine for legacy extraction
full_networks = {
    'Reward': networks['Reward_Cortical'] + networks['Reward_Subcortical'],
    'Olfactory': networks['Olfactory_Cortical'] + networks['Olfactory_Subcortical'],
    'DMN': networks['DMN_Cortical'] + networks['DMN_Subcortical'],
    'Salience': networks['Salience_Cortical'] + networks['Salience_Subcortical']
}

# WORKER FUNCTION
def extract_decomposed_fc(filepath):
    sid = os.path.basename(filepath).split('_')[0]
    try:
        ts_410 = np.load(filepath)
        corr_matrix = np.corrcoef(ts_410.T)
        z_matrix = fisher_z(corr_matrix)
        
        metrics = {'Subject': int(sid)}
        
        # 1. Intra-Network Decompositions
        for net in ['Reward', 'Olfactory', 'DMN', 'Salience']:
            idx_ctx = [i - 1 for i in networks[f'{net}_Cortical'] if 0 <= (i - 1) < 410]
            idx_sub = [i - 1 for i in networks[f'{net}_Subcortical'] if 0 <= (i - 1) < 410]
            idx_full = idx_ctx + idx_sub
            
            # Cortical-Cortical (PRIMARY METRIC)
            ctx_submat = z_matrix[np.ix_(idx_ctx, idx_ctx)]
            np.fill_diagonal(ctx_submat, np.nan)
            metrics[f'{net}_FC_Cortical'] = np.nanmean(ctx_submat)
            
            # Subcortical-Subcortical (SUPPLEMENTARY)
            sub_submat = z_matrix[np.ix_(idx_sub, idx_sub)]
            np.fill_diagonal(sub_submat, np.nan)
            metrics[f'{net}_FC_Subcortical'] = np.nanmean(sub_submat)
            
            # Cortico-Subcortical (SUPPLEMENTARY)
            ctx_sub_cross = z_matrix[np.ix_(idx_ctx, idx_sub)]
            metrics[f'{net}_FC_CorticoSubcortical'] = np.nanmean(ctx_sub_cross)
            
            # Full Network (LEGACY)
            full_submat = z_matrix[np.ix_(idx_full, idx_full)]
            np.fill_diagonal(full_submat, np.nan)
            metrics[f'{net}_FC_Full'] = np.nanmean(full_submat)
            
            # Segregation (CLEAN FOR DMN/SALIENCE ONLY)
            outside_idx = [i for i in range(410) if i not in idx_ctx]
            between_ctx = z_matrix[np.ix_(idx_ctx, outside_idx)]
            metrics[f'{net}_Segregation_Cortical'] = (metrics[f'{net}_FC_Cortical'] - np.nanmean(between_ctx)) / metrics[f'{net}_FC_Cortical']

        # 2. Targeted Olfactory-to-Reward Gateway FC (IFOF Axis - CORE RESULT)
        idx_olf_p = [i - 1 for i in olf_primary if 0 <= (i - 1) < 410]
        idx_rew_g = [i - 1 for i in rew_gateway if 0 <= (i - 1) < 410]
        targeted_cross = z_matrix[np.ix_(idx_olf_p, idx_rew_g)]
        metrics['Inter_FC_OlfPrimary_RewGateway'] = np.nanmean(targeted_cross)
        
        # 3. Legacy General Inter-Network FC (SUPPLEMENTARY)
        idx_rew_full = [i - 1 for i in full_networks['Reward'] if 0 <= (i - 1) < 410]
        idx_olf_full = [i - 1 for i in full_networks['Olfactory'] if 0 <= (i - 1) < 410]
        inter_rew_olf = z_matrix[np.ix_(idx_rew_full, idx_olf_full)]
        metrics['Inter_FC_Olfactory_Reward'] = np.nanmean(inter_rew_olf)
        
        return sid, "SUCCESS", metrics
        
    except Exception as e:
        return sid, "FAILED", str(e)

# 4. EXECUTION
if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] Starting Phase 1: Targeted Extraction...")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_decomposed_fc, f): f for f in files}
        for future in as_completed(futures):
            sid, status, output = future.result()
            if status == "SUCCESS": results.append(output)

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"--- EXTRACTION COMPLETE ---")
        print(f"Saved refined metrics for discordant analysis: {OUTPUT_CSV}")